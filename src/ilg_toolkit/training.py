"""In-memory fitting with sequential, equally weighted regional contributions."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import NamedTuple, cast, final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .config import TrainingConfig
from .data import ObservationPartition, PairwiseObservations, RegionBatch
from .mlpe import (
    MLPEConfig,
    MLPEHead,
    finalize_mlpe_fit,
    mlpe_ml_negative_log_likelihood,
    profiled_mlpe_ml_fit,
)
from .mlpe.likelihood import MLPEFit
from .model import CalibratedModel
from .models import AbstractDistanceModel
from .resistance import ResistanceSolverContext

RegionCollection = RegionBatch | Sequence[RegionBatch] | Mapping[str, RegionBatch]
ObservationCollection = (
    PairwiseObservations | Sequence[PairwiseObservations] | Mapping[str, PairwiseObservations]
)
PartitionCollection = (
    ObservationPartition
    | Sequence[ObservationPartition | None]
    | Mapping[str, ObservationPartition | None]
    | None
)
Encoder = AbstractDistanceModel
Parameters = tuple[Encoder, dict[str, jax.Array] | None]


@final
class EpochRecord(eqx.Module):
    """Inference objectives after an update; zero is initialization.

    The aggregate is the arithmetic mean of regional, per-pair means.
    """

    epoch: int
    training_loss: float
    validation_loss: float | None = None
    training_by_region: dict[str, float] = eqx.field(default_factory=dict)
    validation_by_region: dict[str, float] = eqx.field(default_factory=dict)


@final
class TrainingState(eqx.Module):
    """Complete fitting result and continuation state, returned directly by ``fit``.

    The latest model, variance parameters and Adam state describe the latest
    epoch. ``model`` selects ``best_model``, which retains its own encoder and
    regional calibrations. MLPE ``raw_variances`` maps each region identity to
    its two nuisance parameters; direct fits use ``None``. Progress derives
    from history, and the input contract records structure rather than values.
    RNG uses a serializable legacy uint32 key.
    """

    raw_variances: dict[str, jax.Array] | None
    optimizer_state: optax.OptState
    rng_key: jax.Array
    config: TrainingConfig
    input_contract: dict
    history: tuple[EpochRecord, ...]
    latest_model: CalibratedModel
    best_model: CalibratedModel
    selected_epoch: int

    @property
    def model(self) -> CalibratedModel:
        """The model selected by the declared validation or final-epoch policy."""
        return self.best_model

    @property
    def encoder(self) -> Encoder:
        return self.latest_model.encoder

    @property
    def epoch(self) -> int:
        return self.history[-1].epoch

    @property
    def step(self) -> int:
        return self.epoch

    @property
    def region_names(self) -> tuple[str, ...]:
        return tuple(self.latest_model.training_pairs)

    @property
    def selection(self) -> str:
        return "validation" if self.latest_model.validation_pairs else "final"

    @property
    def best_loss(self) -> float:
        record = self.history[self.selected_epoch]
        return record.training_loss if record.validation_loss is None else record.validation_loss


class _TrainingPayload(NamedTuple):
    """Named numerical inputs passed intact through JAX objective kernels."""

    features: jax.Array
    nodes: jax.Array
    pairs: tuple[jax.Array, jax.Array]
    targets: jax.Array


class _RegionalInput(NamedTuple):
    region: RegionBatch
    observations: PairwiseObservations
    partition: ObservationPartition | None


@dataclass(frozen=True)
class _RegionalBatch:
    """Keep each region's observations and solver state separate from its shared encoder."""

    region: RegionBatch
    observations: PairwiseObservations
    payload: _TrainingPayload
    context: ResistanceSolverContext | None = None


def _prepared(
    region: RegionBatch,
    observations: PairwiseObservations,
    partition: ObservationPartition | None = None,
    *,
    role: str = "training",
    objective: str = "direct_log1p",
) -> _TrainingPayload:
    if partition is not None and partition.role != role:
        raise ValueError(f"Expected a {role} partition, got {partition.role}")
    pairs, values = observations.aligned_pairs(region, partition)
    if objective == "direct_log1p" and (
        observations.target.kind != "dissimilarity" or jnp.any(values < 0)
    ):
        raise ValueError(
            "direct_log1p requires nonnegative dissimilarities; relatedness is unsupported"
        )
    if objective == "mlpe":
        if observations.target.kind != "dissimilarity":
            raise ValueError("Population MLPE requires declared dissimilarities")
        kinds = region.sampling_unit_kinds
        assert kinds is not None
        if any(kind != "population" for kind in kinds):
            raise ValueError("Population MLPE does not support individual sampling units")
        if role == "training" and len(values) < 3:
            raise ValueError("MLPE training requires at least three observed pairs")
        if role == "training":
            degree = jnp.bincount(jnp.concatenate(pairs), length=len(region.sampling_unit_ids))
            if degree.max() <= 1:
                raise ValueError("MLPE variances are unidentifiable: pairs share no endpoints")
    targets = observations.target.forward(values)
    return _TrainingPayload(
        features=region.feature_array,
        nodes=region.pixel_nodes,
        pairs=pairs,
        targets=targets.astype(jnp.float64) if objective == "mlpe" else targets,
    )


def _align_collection(values: object, names: Sequence[str], description: str) -> tuple[object, ...]:
    """Align one labelled collection without assigning it a statistical role."""
    if isinstance(values, Mapping):
        if set(values) != set(names):
            raise ValueError(f"{description} mapping keys must match region names exactly")
        return tuple(values[name] for name in names)
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        if len(values) != len(names):
            raise ValueError(f"{description} sequence must align with all supplied regions")
        return tuple(values)
    if len(names) == 1:
        return (values,)
    raise ValueError(f"{description} must align with the supplied region collection")


def _normalize_inputs(
    regions: RegionCollection,
    observations: ObservationCollection,
    partitions: PartitionCollection,
) -> tuple[_RegionalInput, ...]:
    """Align explicit collections, then sort stable region identities for RNG order."""
    if isinstance(regions, RegionBatch):
        ordered = [regions]
    elif isinstance(regions, Mapping):
        ordered = list(regions.values())
        if any(
            not isinstance(value, RegionBatch) or key != value.name
            for key, value in regions.items()
        ):
            raise ValueError("Region mapping keys must match each prepared region's name")
    elif isinstance(regions, Sequence) and not isinstance(regions, (str, bytes)):
        ordered = list(regions)
    else:
        raise ValueError("Provide a prepared region or a nonempty region sequence/mapping")
    if not ordered or any(not isinstance(region, RegionBatch) for region in ordered):
        raise ValueError("Regions must be a nonempty collection of RegionBatch inputs")
    names = [region.name for region in ordered]
    if len(set(names)) != len(names):
        raise ValueError("Region names must be unique; duplicate names cannot share a fit")

    observed = _align_collection(observations, names, "Observations")
    selected = (
        (None,) * len(ordered)
        if partitions is None
        else _align_collection(partitions, names, "Partitions")
    )
    inputs = []
    for region, observation, partition in zip(ordered, observed, selected, strict=True):
        if not isinstance(observation, PairwiseObservations):
            raise ValueError("Observations contain unsupported input types")
        if partition is not None and not isinstance(partition, ObservationPartition):
            raise ValueError("Partitions contain unsupported input types")
        inputs.append(_RegionalInput(region, observation, partition))
    return tuple(sorted(inputs, key=lambda item: item.region.name))


def _regional_loss(
    encoder: Encoder,
    payload: _TrainingPayload,
    *,
    context: ResistanceSolverContext | None = None,
    inference: bool = True,
    key: jax.Array | None = None,
) -> jax.Array:
    predictions = _selected_scores(encoder, payload, context=context, inference=inference, key=key)
    return jnp.mean(jnp.square(jnp.log1p(predictions) - jnp.log1p(payload.targets)))


def _selected_scores(
    encoder: Encoder,
    payload: _TrainingPayload,
    *,
    context: ResistanceSolverContext | None = None,
    inference: bool = True,
    key: jax.Array | None = None,
) -> jax.Array:
    scores = encoder.predict_distances(
        payload.features, payload.nodes, context=context, inference=inference, key=key
    )
    return scores[payload.pairs]


def _joint_loss(
    parameters: Parameters,
    payload: _TrainingPayload,
    *,
    region_name: str,
    config: TrainingConfig,
    context: ResistanceSolverContext | None = None,
    key: jax.Array | None = None,
) -> jax.Array:
    encoder, raw_variances = parameters
    if config.objective == "direct_log1p":
        return _regional_loss(encoder, payload, context=context, inference=False, key=key)
    assert raw_variances is not None
    scores = _selected_scores(encoder, payload, context=context, inference=False, key=key).astype(
        jnp.float64
    )
    nll, _ = profiled_mlpe_ml_fit(
        scores,
        payload.targets,
        *payload.pairs,
        n_populations=len(payload.nodes),
        raw_variances=raw_variances[region_name],
        variance_floor=config.mlpe_variance_floor,
        jitter=config.mlpe_jitter,
    )
    return nll / len(scores)


def _joint_value_and_grad(
    parameters: Parameters,
    payload: _TrainingPayload,
    *,
    region_name: str,
    config: TrainingConfig,
    context: ResistanceSolverContext | None = None,
    key: jax.Array | None = None,
) -> tuple[jax.Array, Parameters]:
    value, gradient = eqx.filter_value_and_grad(_joint_loss)(
        parameters,
        payload,
        region_name=region_name,
        config=config,
        context=context,
        key=key,
    )
    bad = ~jnp.isfinite(value)
    for leaf in jax.tree.leaves(gradient):
        if eqx.is_inexact_array(leaf):
            bad = bad | jnp.any(~jnp.isfinite(leaf))
    return eqx.error_if(
        (value, gradient),
        bad,
        "Nonfinite objective or gradient; check MLPE score variation, "
        "identifiable endpoints and variance conditioning",
    )


def _refresh_mlpe(
    encoder: Encoder,
    raw: jax.Array,
    payload: _TrainingPayload,
    *,
    config: TrainingConfig,
    context: ResistanceSolverContext | None = None,
) -> MLPEFit:
    scores = _selected_scores(encoder, payload, context=context).astype(jnp.float64)
    return finalize_mlpe_fit(
        scores,
        payload.targets,
        *payload.pairs,
        n_populations=len(payload.nodes),
        raw_variances=raw,
        variance_floor=config.mlpe_variance_floor,
        jitter=config.mlpe_jitter,
    )


def _frozen_validation_loss(
    encoder: Encoder,
    raw: jax.Array,
    moments: tuple[jax.Array, jax.Array, jax.Array],
    payload: _TrainingPayload,
    *,
    config: TrainingConfig,
    context: ResistanceSolverContext | None = None,
) -> jax.Array:
    scores = _selected_scores(encoder, payload, context=context).astype(jnp.float64)
    center, scale, beta = moments
    nll = mlpe_ml_negative_log_likelihood(
        scores,
        payload.targets,
        *payload.pairs,
        n_populations=len(payload.nodes),
        fixed_effects=beta,
        raw_variances=raw,
        score_center=center,
        score_scale=scale,
        variance_floor=config.mlpe_variance_floor,
        jitter=config.mlpe_jitter,
    )
    return nll / len(scores)


_compiled_loss = eqx.filter_jit(_regional_loss)
_compiled_joint_gradient = eqx.filter_jit(_joint_value_and_grad)
_compiled_refresh = eqx.filter_jit(_refresh_mlpe)
_compiled_validation = eqx.filter_jit(_frozen_validation_loss)


def _labelled_pairs(batch: _RegionalBatch) -> tuple[tuple[str, str], ...]:
    return tuple(
        (
            min(batch.region.sampling_unit_ids[i], batch.region.sampling_unit_ids[j]),
            max(batch.region.sampling_unit_ids[i], batch.region.sampling_unit_ids[j]),
        )
        for i, j in zip(*batch.payload.pairs, strict=True)
    )


def _input_contract(
    training_inputs: Sequence[_RegionalInput], validation_inputs: Sequence[_RegionalInput]
) -> dict:
    """Cheap resume structure; unchanged measurement contents remain caller responsibility."""
    return {
        role: [
            {
                "region_name": region.name,
                "sampling_unit_ids": list(region.sampling_unit_ids),
                "sampling_unit_kinds": list(region.sampling_unit_kinds or ()),
                "feature_shape": list(region.feature_array.shape),
                "feature_dtype": str(region.feature_array.dtype),
                "position_dtype": str(cast(jax.Array, region.grid_positions).dtype),
                "observation_dtype": str(observations.values.dtype),
            }
            for region, observations, _ in inputs
        ]
        for role, inputs in (("training", training_inputs), ("validation", validation_inputs))
    }


def _initial_raw_variances(
    batches: Sequence[_RegionalBatch], config: TrainingConfig
) -> dict[str, jax.Array]:
    values = {}
    for batch in batches:
        if config.mlpe_initial_variances is None:
            variance = jnp.maximum(
                jnp.var(batch.payload.targets, ddof=1),
                config.mlpe_variance_floor * 100,
            )
            pair = jnp.maximum(variance * jnp.asarray([0.25, 0.5]), config.mlpe_variance_floor * 10)
        else:
            pair = jnp.asarray(config.mlpe_initial_variances, dtype=jnp.float64)
        positive = pair - config.mlpe_variance_floor
        # log(expm1(x)) without overflowing for large user-supplied initial variances.
        raw = positive + jnp.log(-jnp.expm1(-positive))
        values[batch.region.name] = raw
    return values


def _prepare_batches(
    training_inputs: Sequence[_RegionalInput],
    validation_inputs: Sequence[_RegionalInput],
    encoder: Encoder,
    config: TrainingConfig,
) -> tuple[tuple[_RegionalBatch, ...], tuple[_RegionalBatch, ...]]:
    """Prepare independent regional targets and reuse graph-shape solver contexts."""
    contexts: dict[tuple[int, int], ResistanceSolverContext | None] = {}

    def make_batch(inputs: _RegionalInput, role: str) -> _RegionalBatch:
        region, observations, partition = inputs
        payload = _prepared(region, observations, partition, role=role, objective=config.objective)
        shape = (region.feature_array.shape[0], region.feature_array.shape[1])
        if shape not in contexts:
            contexts[shape] = encoder.prepare(region.feature_array)
        context = contexts[shape]
        return _RegionalBatch(region, observations, payload, context)

    training = tuple(make_batch(inputs, "training") for inputs in training_inputs)
    validation = tuple(make_batch(inputs, "validation") for inputs in validation_inputs)
    training_by_name = {batch.region.name: batch for batch in training}
    for batch in validation:
        training_batch = training_by_name.get(batch.region.name)
        if training_batch is None:
            if config.objective == "mlpe":
                raise ValueError(
                    "MLPE validation requires a training calibration for the same region"
                )
            continue
        if set(_labelled_pairs(training_batch)) & set(_labelled_pairs(batch)):
            raise ValueError("Training and validation observations overlap within the same region")
        if not np.array_equal(training_batch.region.feature_array, batch.region.feature_array):
            raise ValueError(
                "Training and validation inputs for the same region must share a landscape"
            )
        locations = dict(
            zip(
                training_batch.region.sampling_unit_ids,
                cast(jax.Array, training_batch.region.grid_positions),
                strict=True,
            )
        )
        if any(
            label in locations and not np.array_equal(locations[label], position)
            for label, position in zip(
                batch.region.sampling_unit_ids,
                cast(jax.Array, batch.region.grid_positions),
                strict=True,
            )
        ):
            raise ValueError("Sampling-unit locations disagree within the same region")
    return training, validation


def _checked_call[T](
    function: Callable[..., T],
    *args,
    batch: _RegionalBatch,
    epoch: int,
    training_config: TrainingConfig,
    **kwargs,
) -> T:
    """Attach the regional numerical context without changing a failed calculation."""
    try:
        return jax.block_until_ready(function(*args, **kwargs))
    except RuntimeError as error:
        raise RuntimeError(
            f"{batch.region.name}: {training_config.objective} calculation failed "
            f"at epoch {epoch}; "
            f"check finite scores, gradients, score variation, identifiable pair endpoints "
            f"and variance conditioning; "
            f"solver={batch.context.config if batch.context is not None else None}. {error}"
        ) from error


def _apply_gradient(
    parameters: Parameters,
    optimizer_state: optax.OptState,
    gradient: Parameters,
    *,
    optimizer: optax.GradientTransformation,
) -> tuple[Parameters, optax.OptState]:
    # Optax accepts registered Equinox pytrees; its array-container alias omits Modules.
    updates, optimizer_state = optimizer.update(
        cast(optax.Updates, gradient), optimizer_state, cast(optax.Params, parameters)
    )
    parameters = eqx.apply_updates(parameters, updates)
    bad = jnp.array(False)
    for leaf in jax.tree.leaves((parameters, optimizer_state)):
        if eqx.is_inexact_array(leaf):
            bad = bad | jnp.any(~jnp.isfinite(leaf))
    return eqx.error_if((parameters, optimizer_state), bad, "Nonfinite optimizer update")


def _evaluate_regions(
    parameters: Parameters,
    batches: Sequence[_RegionalBatch],
    heads: Mapping[str, MLPEHead],
    config: TrainingConfig,
    epoch: int,
) -> tuple[float, dict[str, float]]:
    """Score with frozen training calibration and equal regional weighting."""
    evaluate = _compiled_loss if config.jit else _regional_loss
    validate = _compiled_validation if config.jit else _frozen_validation_loss
    values = {}
    encoder, raw_variances = parameters
    for batch in batches:
        if config.objective == "mlpe":
            assert raw_variances is not None
            head = heads[batch.region.name]
            moments = (
                jnp.asarray(head.score_center),
                jnp.asarray(head.score_scale),
                jnp.asarray((head.intercept, head.slope)),
            )
            value = _checked_call(
                validate,
                encoder,
                raw_variances[batch.region.name],
                moments,
                batch.payload,
                config=config,
                context=batch.context,
                batch=batch,
                epoch=epoch,
                training_config=config,
            )
        else:
            value = _checked_call(
                evaluate,
                encoder,
                batch.payload,
                context=batch.context,
                batch=batch,
                epoch=epoch,
                training_config=config,
            )
        if not np.isfinite(float(value)):
            raise FloatingPointError(
                f"{batch.region.name}: nonfinite {config.objective} objective at epoch {epoch}; "
                "check scores and variance conditioning"
            )
        values[batch.region.name] = float(value)
    return float(np.mean(list(values.values()))), values


def _refresh_heads(
    parameters: Parameters,
    batches: Sequence[_RegionalBatch],
    training_pairs: Mapping[str, tuple[tuple[str, str], ...]],
    config: TrainingConfig,
    epoch: int,
) -> tuple[dict[str, MLPEHead], float, dict[str, float]]:
    """Refresh fitted regional moments, GLS coefficients and effect posteriors together."""
    encoder, raw_variances = parameters
    assert raw_variances is not None
    refresh = _compiled_refresh if config.jit else _refresh_mlpe
    heads, losses = {}, {}
    for batch in batches:
        result = _checked_call(
            refresh,
            encoder,
            raw_variances[batch.region.name],
            batch.payload,
            config=config,
            context=batch.context,
            batch=batch,
            epoch=epoch,
            training_config=config,
        )
        if not all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(result)):
            raise FloatingPointError(
                f"{batch.region.name}: nonfinite MLPE calibration at epoch {epoch}; "
                "check nonconstant scores, identifiable endpoints and variance conditioning"
            )
        heads[batch.region.name] = MLPEHead.from_fit(
            result,
            region_name=batch.region.name,
            target=batch.observations.target,
            population_ids=batch.region.sampling_unit_ids,
            calibration_pairs=training_pairs[batch.region.name],
            calibration_roles=("training",) * len(batch.payload.targets),
            config=MLPEConfig(variance_floor=config.mlpe_variance_floor, jitter=config.mlpe_jitter),
            optimizer_iterations=epoch,
            optimizer_message="Fixed-budget joint Adam; no variance convergence claim",
            converged=False,
        )
        losses[batch.region.name] = float(result.nll) / len(batch.payload.targets)
    return heads, float(np.mean(list(losses.values()))), losses


def fit(
    region: RegionCollection,
    observations: ObservationCollection,
    *,
    model: AbstractDistanceModel | None = None,
    config: TrainingConfig | None = None,
    validation: tuple[RegionCollection, ObservationCollection] | None = None,
    partition: PartitionCollection = None,
    validation_partition: PartitionCollection = None,
    state: TrainingState | None = None,
    on_epoch: Callable[[TrainingState], None] | None = None,
) -> TrainingState:
    """Fit a shared encoder with direct log1p-MSE or full-ML regional MLPE.

    Each Adam update averages sequential regional gradients, normalized by pair
    count. MLPE profiles signed GLS coefficients and updates the encoder and two
    regional variance parameters together. Updated training scores refresh the
    regional moments, coefficients and posterior before frozen-head validation.
    Validation only selects a model; learning rate and stopping use a fixed
    budget. MLPE explicitly requires enabled JAX float64.

    Fresh fits require an explicit ``model`` encoder and a ``TrainingConfig``
    declaring the objective. Continuation reuses these saved choices.

    ``state`` continues the latest optimizer/RNG trajectory on identical inputs.
    Resume checks structure and declared contracts, not measurement contents;
    keeping landscapes, locations and genetic measurements unchanged is the
    caller's responsibility.
    Only the total epoch budget may increase. The returned complete state exposes
    the selected model through ``model`` and retains both ``latest_model`` and
    ``best_model`` when validation selects an earlier epoch. Validation never
    consumes training random keys.

    ``on_epoch`` receives complete latest/best state after initialization and each
    completed update. It may save a checkpoint; exceptions propagate. A resumed
    fit with no additional updates does not invoke a synthetic epoch callback.
    """
    if state is not None and not isinstance(state, TrainingState):
        raise ValueError("state must be a TrainingState returned by fit")
    if state is not None and model is not None:
        raise ValueError("A continuation state already contains its encoder; omit model")
    if state is None:
        if model is None:
            raise ValueError("Fresh fitting requires an explicit model encoder")
        if config is None:
            raise ValueError("Fresh fitting requires an explicit TrainingConfig with objective")
    else:
        model = state.encoder
        config = state.config if config is None else config
    if on_epoch is not None and not callable(on_epoch):
        raise ValueError("on_epoch must be callable or None")
    if state is not None:
        old, new = asdict(state.config), asdict(config)
        old.pop("epochs")
        new.pop("epochs")
        if old != new or config.epochs < state.config.epochs:
            raise ValueError(
                "Continuation requires identical configuration except an increased epoch budget"
            )
    if config.objective == "mlpe" and not jax.config.read("jax_enable_x64"):
        raise RuntimeError(
            "MLPE fitting requires float64: set JAX_ENABLE_X64=1 or use jax.enable_x64()"
        )
    training_inputs = _normalize_inputs(region, observations, partition)
    if validation is None and validation_partition is not None:
        raise ValueError("validation_partition requires validation observations")
    if validation is not None and (not isinstance(validation, tuple) or len(validation) != 2):
        raise ValueError("validation must be a (regions, observations) tuple")
    validation_inputs = (
        ()
        if validation is None
        else _normalize_inputs(validation[0], validation[1], validation_partition)
    )
    first_region, first_observations, _ = training_inputs[0]
    regional_names = {inputs[0].name for inputs in (*training_inputs, *validation_inputs)}
    if len(regional_names) > 1 and first_region.feature_names is None:
        raise ValueError(
            "Shared multi-region fitting requires explicit feature_names meanings and order"
        )
    for prepared_region, regional_observations, _ in (*training_inputs, *validation_inputs):
        if regional_observations.target != first_observations.target:
            raise ValueError("Regional training and validation target scales must match")
        if prepared_region.feature_array.shape[-1] != first_region.feature_array.shape[-1]:
            raise ValueError("Regional training and validation feature channels must match")
        if prepared_region.feature_names != first_region.feature_names:
            raise ValueError("Regional feature contracts must match meanings and order")
    input_contract = _input_contract(training_inputs, validation_inputs)
    if state is not None:
        if state.input_contract != input_contract:
            raise ValueError(
                "Continuation input shapes, precision or sampling-unit identities changed"
            )
        if (
            state.latest_model.target != first_observations.target
            or state.latest_model.feature_names != first_region.feature_names
        ):
            raise ValueError("Continuation target or feature meanings/order changed")

    random_key = (
        jax.random.split(jax.random.PRNGKey(config.seed))[1] if state is None else state.rng_key
    )
    training_batches, validation_batches = _prepare_batches(
        training_inputs, validation_inputs, model, config
    )

    raw_variances = (
        state.raw_variances
        if state is not None
        else _initial_raw_variances(training_batches, config)
        if config.objective == "mlpe"
        else None
    )
    parameters = (model, raw_variances)
    optimizer = optax.adam(config.learning_rate)
    optimizer_state = (
        state.optimizer_state
        if state is not None
        else optimizer.init(eqx.filter(parameters, eqx.is_inexact_array))
    )

    gradient_function = _compiled_joint_gradient if config.jit else _joint_value_and_grad
    apply_update = eqx.filter_jit(_apply_gradient) if config.jit else _apply_gradient

    training_pairs = {batch.region.name: _labelled_pairs(batch) for batch in training_batches}
    validation_pairs = {batch.region.name: _labelled_pairs(batch) for batch in validation_batches}
    if state is not None and (
        state.latest_model.training_pairs != training_pairs
        or state.latest_model.validation_pairs != validation_pairs
    ):
        raise ValueError("Continuation training/validation pair partitions changed")

    def make_model(heads: dict[str, MLPEHead]) -> CalibratedModel:
        return CalibratedModel(
            parameters[0],
            first_observations.target,
            first_region.feature_array.shape[-1],
            feature_names=first_region.feature_names,
            objective=config.objective,
            calibrations=heads,
            training_pairs=training_pairs,
            validation_pairs=validation_pairs,
        )

    history = list(state.history) if state is not None else []
    selected = state.best_model if state is not None else None
    latest = state.latest_model if state is not None else None
    selected_epoch = state.selected_epoch if state is not None else 0
    best_loss = state.best_loss if state is not None else float("inf")

    def continuation() -> TrainingState:
        assert latest is not None and selected is not None
        return TrainingState(
            raw_variances=parameters[1],
            optimizer_state=optimizer_state,
            rng_key=random_key,
            config=config,
            input_contract=input_contract,
            history=tuple(history),
            latest_model=latest,
            best_model=selected,
            selected_epoch=selected_epoch,
        )

    start_epoch = state.epoch + 1 if state is not None else 0
    for epoch in range(start_epoch, config.epochs + 1):
        if epoch:
            random_key, update_key = jax.random.split(random_key)
            keys = jax.random.split(update_key, len(training_batches))
            accumulated = None
            # Complete each regional backward pass before retaining the next.
            for batch, key in zip(training_batches, keys, strict=True):
                _, gradient = _checked_call(
                    gradient_function,
                    parameters,
                    batch.payload,
                    region_name=batch.region.name,
                    config=config,
                    context=batch.context,
                    key=key,
                    batch=batch,
                    epoch=epoch,
                    training_config=config,
                )
                accumulated = (
                    gradient
                    if accumulated is None
                    else jax.tree.map(
                        lambda total, contribution: total + contribution, accumulated, gradient
                    )
                )
                accumulated = jax.block_until_ready(accumulated)
            averaged = jax.tree.map(lambda value: value / len(training_batches), accumulated)
            parameters, optimizer_state = _checked_call(
                apply_update,
                parameters,
                optimizer_state,
                averaged,
                optimizer=optimizer,
                batch=training_batches[0],
                epoch=epoch,
                training_config=config,
            )
        heads: dict[str, MLPEHead] = {}
        if config.objective == "mlpe":
            heads, training_loss, regional_training = _refresh_heads(
                parameters, training_batches, training_pairs, config, epoch
            )
        else:
            training_loss, regional_training = _evaluate_regions(
                parameters, training_batches, heads, config, epoch
            )
        validation_loss, regional_validation = (
            (None, {})
            if not validation_batches
            else _evaluate_regions(parameters, validation_batches, heads, config, epoch)
        )
        history.append(
            EpochRecord(
                epoch, training_loss, validation_loss, regional_training, regional_validation
            )
        )
        latest = make_model(heads)
        if validation_loss is None or validation_loss < best_loss:
            selected, selected_epoch = latest, epoch
            best_loss = training_loss if validation_loss is None else validation_loss
        if on_epoch is not None:
            on_epoch(continuation())
    return continuation()
