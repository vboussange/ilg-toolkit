"""In-memory fitting with sequential, equally weighted regional contributions."""

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import NamedTuple

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
    decode_mlpe_variances,
    mlpe_effect_posterior,
    mlpe_ml_negative_log_likelihood,
    profiled_mlpe_ml_fit,
    sample_standardize_scores,
)
from .models import ConductanceModel, EmbeddingDistanceModel, UNetEmbeddingDistance
from .model import CalibratedModel
from .resistance import ResistanceSolverContext, build_resistance_context

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


@dataclass(frozen=True)
class EpochRecord:
    """Inference objectives after an update; zero is initialization.

    The aggregate is the arithmetic mean of regional, per-pair means.
    """

    epoch: int
    training_loss: float
    validation_loss: float | None = None
    training_by_region: dict[str, float] = field(default_factory=dict)
    validation_by_region: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class FitResult:
    """Selected shared model and regional optimization history."""

    model: CalibratedModel
    history: tuple[EpochRecord, ...]
    selected_epoch: int
    selection: str
    region_names: tuple[str, ...] = ()
    state: "TrainingState | None" = None

    @property
    def latest_model(self):
        """CalibratedModel at the last update, including when validation chose an earlier one."""
        return self.model if self.state is None else self.state.latest_model

    @property
    def best_model(self):
        """The model selected by the declared selection policy."""
        return self.model


@dataclass(frozen=True)
class TrainingState:
    """Explicit in-memory continuation state; checkpoints add serialization separately.

    The encoder, variance parameters and Adam state describe the latest epoch.
    ``best_model`` retains its own encoder and heads. RNG uses a serializable
    legacy uint32 key. The shipped encoders have no mutable model state.
    """

    encoder: ConductanceModel | EmbeddingDistanceModel
    raw_variances: jax.Array | None
    optimizer_state: object
    rng_key: jax.Array
    epoch: int
    step: int
    config: TrainingConfig
    region_names: tuple[str, ...]
    data_identity: str
    history: tuple[EpochRecord, ...]
    latest_model: CalibratedModel
    best_model: CalibratedModel
    selected_epoch: int
    best_loss: float
    selection: str
    model_state: object = None
    schedule_state: dict = field(default_factory=dict)
    stopping_state: dict = field(default_factory=dict)


class _TrainingPayload(NamedTuple):
    """Named numerical inputs passed intact through JAX objective kernels."""

    features: jax.Array
    nodes: jax.Array
    pairs: tuple[np.ndarray, np.ndarray]
    targets: jax.Array


@dataclass(frozen=True)
class _RegionalBatch:
    """Keep each region's observations and solver state separate from its shared encoder."""

    region: RegionBatch
    observations: PairwiseObservations
    partition: ObservationPartition | None
    payload: _TrainingPayload
    context: ResistanceSolverContext | None = None


def _prepared(region, observations, partition=None, *, role="training", objective="direct_log1p"):
    if partition is not None and partition.role != role:
        raise ValueError(f"Expected a {role} partition, got {partition.role}")
    pairs, values = observations.aligned_pairs(region, partition)
    if objective == "direct_log1p" and (
        observations.target.kind != "dissimilarity" or np.any(values < 0)
    ):
        raise ValueError(
            "direct_log1p requires nonnegative dissimilarities; relatedness is unsupported"
        )
    if objective == "mlpe":
        if observations.target.kind != "dissimilarity":
            raise ValueError("Population MLPE requires declared dissimilarities")
        if any(kind != "population" for kind in region.sampling_unit_kinds):
            raise ValueError("Population MLPE does not support individual sampling units")
        if role == "training" and len(values) < 3:
            raise ValueError("MLPE training requires at least three observed pairs")
        if role == "training":
            degree = np.bincount(np.concatenate(pairs), minlength=len(region.sampling_unit_ids))
            if degree.max() <= 1:
                raise ValueError("MLPE variances are unidentifiable: pairs share no endpoints")
    return _TrainingPayload(
        features=jnp.asarray(region.feature_array),
        nodes=jnp.asarray(region.pixel_nodes),
        pairs=pairs,
        targets=jnp.asarray(
            observations.target.forward(values), dtype=jnp.float64 if objective == "mlpe" else None
        ),
    )


def _normalize_inputs(regions, observations, partitions):
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

    def align(values, expected_type, description, *, optional=False):
        if optional and values is None:
            aligned = [None] * len(ordered)
        elif isinstance(values, expected_type) and len(ordered) == 1:
            aligned = [values]
        elif isinstance(values, Mapping):
            if set(values) != set(names):
                raise ValueError(f"{description} mapping keys must match region names exactly")
            aligned = [values[name] for name in names]
        elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            if len(values) != len(ordered):
                raise ValueError(f"{description} sequence must align with all supplied regions")
            aligned = list(values)
        else:
            raise ValueError(f"{description} must align with the supplied region collection")
        if any(
            not isinstance(value, expected_type) and not (optional and value is None)
            for value in aligned
        ):
            raise ValueError(f"{description} contain unsupported input types")
        return aligned

    observed = align(observations, PairwiseObservations, "Observations")
    selected = align(partitions, ObservationPartition, "Partitions", optional=True)
    return tuple(
        sorted(zip(ordered, observed, selected, strict=True), key=lambda item: item[0].name)
    )


def _regional_loss(encoder, payload, *, context=None, inference=True, key=None):
    predictions = _selected_scores(encoder, payload, context=context, inference=inference, key=key)
    return jnp.mean(jnp.square(jnp.log1p(predictions) - jnp.log1p(payload.targets)))


def _selected_scores(encoder, payload, *, context=None, inference=True, key=None):
    options = dict(inference=inference, key=key)
    if isinstance(encoder, ConductanceModel):
        options["context"] = context
    return encoder.predict_distances(payload.features, payload.nodes, **options)[payload.pairs]


def _joint_loss(parameters, payload, *, objective, region_index, config, context=None, key=None):
    encoder, raw_variances = parameters
    if objective == "direct_log1p":
        return _regional_loss(encoder, payload, context=context, inference=False, key=key)
    scores = _selected_scores(encoder, payload, context=context, inference=False, key=key).astype(
        jnp.float64
    )
    nll, _ = profiled_mlpe_ml_fit(
        scores,
        payload.targets,
        *payload.pairs,
        n_populations=len(payload.nodes),
        raw_variances=raw_variances[region_index],
        variance_floor=config.mlpe_variance_floor,
        jitter=config.mlpe_jitter,
    )
    return nll / len(scores)


def _joint_value_and_grad(parameters, payload, **kwargs):
    value, gradient = eqx.filter_value_and_grad(_joint_loss)(parameters, payload, **kwargs)
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


def _refresh_mlpe(encoder, raw, payload, *, config, context=None):
    scores = _selected_scores(encoder, payload, context=context).astype(jnp.float64)
    _, center, scale = sample_standardize_scores(scores)
    options = dict(
        n_populations=len(payload.nodes),
        raw_variances=raw,
        variance_floor=config.mlpe_variance_floor,
        jitter=config.mlpe_jitter,
        score_center=center,
        score_scale=scale,
    )
    nll, beta = profiled_mlpe_ml_fit(scores, payload.targets, *payload.pairs, **options)
    posterior = mlpe_effect_posterior(
        scores, payload.targets, *payload.pairs, fixed_effects=beta, **options
    )
    variances = jnp.stack(decode_mlpe_variances(raw, variance_floor=config.mlpe_variance_floor))
    return nll, beta, center, scale, variances, posterior


def _frozen_validation_loss(encoder, raw, moments, payload, *, config, context=None):
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


def _labelled_pairs(batch):
    return tuple(
        tuple(sorted((batch.region.sampling_unit_ids[i], batch.region.sampling_unit_ids[j])))
        for i, j in zip(*batch.payload.pairs, strict=True)
    )


def _fit_data_identity(training_inputs, validation_inputs):
    """Hash the normalized landscapes, labelled measurements and selected roles."""
    digest = hashlib.sha256()

    def metadata(value):
        digest.update(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())

    def array(value):
        value = np.asarray(value)
        metadata([str(value.dtype), value.shape])
        digest.update(np.ascontiguousarray(value).tobytes())

    for role, inputs in (("training", training_inputs), ("validation", validation_inputs)):
        for region, observations, partition in inputs:
            pairs, values = observations.aligned_pairs(region, partition)
            metadata(
                [
                    role,
                    region.name,
                    region.sampling_unit_ids,
                    region.sampling_unit_kinds,
                    region.feature_names,
                    asdict(observations.target),
                ]
            )
            array(region.feature_array)
            array(region.grid_positions)
            array(pairs)
            array(values)
    return digest.hexdigest()


def _initial_raw_variances(batches, config):
    values = []
    for batch in batches:
        if config.mlpe_initial_variances is None:
            variance = max(
                float(np.var(np.asarray(batch.payload.targets), ddof=1)),
                config.mlpe_variance_floor * 100,
            )
            pair = (
                max(variance / 4, config.mlpe_variance_floor * 10),
                max(variance / 2, config.mlpe_variance_floor * 10),
            )
        else:
            pair = config.mlpe_initial_variances
        positive = np.asarray(pair) - config.mlpe_variance_floor
        # log(expm1(x)) without overflowing for large user-supplied initial variances.
        raw = positive + np.log(-np.expm1(-positive))
        values.append(raw)
    return jnp.asarray(np.asarray(values), dtype=jnp.float64)


def fit(
    region: RegionCollection,
    observations: ObservationCollection,
    *,
    model: ConductanceModel | EmbeddingDistanceModel | None = None,
    config: TrainingConfig | None = None,
    validation: tuple[RegionCollection, ObservationCollection] | None = None,
    partition: PartitionCollection = None,
    validation_partition: PartitionCollection = None,
    state: TrainingState | None = None,
    on_epoch: Callable[[TrainingState], None] | None = None,
) -> FitResult:
    """Fit a shared encoder with direct log1p-MSE or full-ML regional MLPE.

    Each Adam update averages sequential regional gradients, normalized by pair
    count. MLPE profiles signed GLS coefficients and updates the encoder and two
    regional variance parameters together. Updated training scores refresh the
    regional moments, coefficients and posterior before frozen-head validation.
    Validation only selects a model; learning rate and stopping use a fixed
    budget. MLPE explicitly requires enabled JAX float64.

    ``state`` continues the latest optimizer/RNG trajectory on identical inputs.
    Only the total epoch budget may increase. The returned selected model
    and latest continuation state are separate when validation selects an earlier
    epoch. Validation never consumes training random keys.

    ``on_epoch`` receives complete latest/best state after initialization and each
    completed update. It may save a checkpoint; exceptions propagate. A resumed
    fit with no additional updates does not invoke a synthetic epoch callback.
    """
    if state is not None and not isinstance(state, TrainingState):
        raise ValueError("state must be a TrainingState returned by fit")
    if state is not None and model is not None:
        raise ValueError("A continuation state already contains its encoder; omit model")
    if on_epoch is not None and not callable(on_epoch):
        raise ValueError("on_epoch must be callable or None")
    config = config or (state.config if state is not None else TrainingConfig())
    if state is not None:
        old, new = asdict(state.config), asdict(config)
        old.pop("epochs")
        new.pop("epochs")
        if old != new or config.epochs < state.config.epochs:
            raise ValueError(
                "Continuation requires identical configuration except an increased epoch budget"
            )
    if config.objective == "mlpe" and not jax.config.x64_enabled:
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
    identity = _fit_data_identity(training_inputs, validation_inputs)
    if state is not None and state.data_identity != identity:
        raise ValueError("Continuation inputs, observations, target scale or partitions changed")

    init_key, random_key = jax.random.split(jax.random.PRNGKey(config.seed))
    if state is not None:
        model, random_key = state.encoder, state.rng_key
    elif model is None:
        model = UNetEmbeddingDistance(
            first_region.feature_array.shape[-1],
            patch_size=1,
            base_channels=8,
            embedding_dim=4,
            dropout=0,
            key=init_key,
        )
    contexts = {}

    def make_batch(inputs, role):
        prepared_region, regional_observations, selected_partition = inputs
        payload = _prepared(
            prepared_region,
            regional_observations,
            selected_partition,
            role=role,
            objective=config.objective,
        )
        context = None
        if isinstance(model, ConductanceModel):
            height, width = prepared_region.feature_array.shape[:2]
            if height % model.patch_size or width % model.patch_size:
                raise ValueError("Raster dimensions must be divisible by model patch_size")
            shape = (height // model.patch_size, width // model.patch_size)
            if shape not in contexts:
                contexts[shape] = build_resistance_context(shape, config.solver)
            context = contexts[shape]
        return _RegionalBatch(
            prepared_region, regional_observations, selected_partition, payload, context
        )

    training_batches = tuple(make_batch(inputs, "training") for inputs in training_inputs)
    validation_batches = tuple(make_batch(inputs, "validation") for inputs in validation_inputs)
    training_by_name = {batch.region.name: batch for batch in training_batches}
    region_indices = {batch.region.name: i for i, batch in enumerate(training_batches)}
    for batch in validation_batches:
        training_batch = training_by_name.get(batch.region.name)
        if training_batch is None and config.objective == "mlpe":
            raise ValueError("MLPE validation requires a training calibration for the same region")
        if training_batch is not None:
            if set(_labelled_pairs(training_batch)) & set(_labelled_pairs(batch)):
                raise ValueError(
                    "Training and validation observations overlap within the same region"
                )
            if not np.array_equal(training_batch.region.feature_array, batch.region.feature_array):
                raise ValueError(
                    "Training and validation inputs for the same region must share a landscape"
                )
            locations = dict(
                zip(
                    training_batch.region.sampling_unit_ids,
                    training_batch.region.grid_positions,
                    strict=True,
                )
            )
            if any(
                label in locations and not np.array_equal(locations[label], position)
                for label, position in zip(
                    batch.region.sampling_unit_ids, batch.region.grid_positions, strict=True
                )
            ):
                raise ValueError("Sampling-unit locations disagree within the same region")

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

    def apply_gradient(parameters, optimizer_state, gradient):
        updates, optimizer_state = optimizer.update(gradient, optimizer_state, parameters)
        parameters = eqx.apply_updates(parameters, updates)
        bad = jnp.array(False)
        for leaf in jax.tree.leaves((parameters, optimizer_state)):
            if eqx.is_inexact_array(leaf):
                bad = bad | jnp.any(~jnp.isfinite(leaf))
        return eqx.error_if((parameters, optimizer_state), bad, "Nonfinite optimizer update")

    evaluate = _compiled_loss if config.jit else _regional_loss
    gradient_function = _compiled_joint_gradient if config.jit else _joint_value_and_grad
    refresh = _compiled_refresh if config.jit else _refresh_mlpe
    validation_function = _compiled_validation if config.jit else _frozen_validation_loss
    apply_update = eqx.filter_jit(apply_gradient) if config.jit else apply_gradient

    def checked_call(function, *args, batch, epoch, **kwargs):
        try:
            return jax.block_until_ready(function(*args, **kwargs))
        except RuntimeError as error:
            raise RuntimeError(
                f"{batch.region.name}: {config.objective} calculation failed at epoch {epoch}; "
                f"check finite scores, gradients, score variation, identifiable pair endpoints "
                f"and variance conditioning; solver={config.solver}. {error}"
            ) from error

    def evaluate_regions(batches, epoch, heads):
        values = {}
        for batch in batches:
            if config.objective == "mlpe":
                head = heads[batch.region.name]
                moments = (
                    jnp.asarray(head.score_center),
                    jnp.asarray(head.score_scale),
                    jnp.asarray((head.intercept, head.slope)),
                )
                value = checked_call(
                    validation_function,
                    parameters[0],
                    parameters[1][region_indices[batch.region.name]],
                    moments,
                    batch.payload,
                    config=config,
                    context=batch.context,
                    batch=batch,
                    epoch=epoch,
                )
            else:
                value = checked_call(
                    evaluate,
                    parameters[0],
                    batch.payload,
                    context=batch.context,
                    batch=batch,
                    epoch=epoch,
                )
            if not np.isfinite(float(value)):
                raise FloatingPointError(
                    f"{batch.region.name}: nonfinite {config.objective} objective "
                    f"at epoch {epoch}; "
                    "check scores and variance conditioning"
                )
            values[batch.region.name] = float(value)
        return float(np.mean(list(values.values()))), values

    training_pairs = {batch.region.name: _labelled_pairs(batch) for batch in training_batches}
    validation_pairs = {batch.region.name: _labelled_pairs(batch) for batch in validation_batches}

    def make_model(heads):
        return CalibratedModel(
            parameters[0],
            first_observations.target,
            first_region.feature_array.shape[-1],
            feature_names=first_region.feature_names,
            solver_config=config.solver,
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
    names = tuple(batch.region.name for batch in training_batches)
    selection = "final" if validation is None else "validation"

    def continuation_at(epoch):
        return TrainingState(
            encoder=parameters[0],
            raw_variances=parameters[1],
            optimizer_state=optimizer_state,
            rng_key=random_key,
            epoch=epoch,
            step=epoch,
            config=config,
            region_names=names,
            data_identity=identity,
            history=tuple(history),
            latest_model=latest,
            best_model=selected,
            selected_epoch=selected_epoch,
            best_loss=best_loss,
            selection=selection,
            model_state=None,
            schedule_state={"kind": "fixed", "learning_rate": config.learning_rate},
            stopping_state={"kind": "fixed_budget", "epochs": config.epochs},
        )

    start_epoch = state.epoch + 1 if state is not None else 0
    for epoch in range(start_epoch, config.epochs + 1):
        if epoch:
            random_key, update_key = jax.random.split(random_key)
            keys = jax.random.split(update_key, len(training_batches))
            accumulated = None
            # Complete each regional backward pass before retaining the next.
            for i, (batch, key) in enumerate(zip(training_batches, keys, strict=True)):
                _, gradient = checked_call(
                    gradient_function,
                    parameters,
                    batch.payload,
                    objective=config.objective,
                    region_index=i,
                    config=config,
                    context=batch.context,
                    key=key,
                    batch=batch,
                    epoch=epoch,
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
            parameters, optimizer_state = checked_call(
                apply_update,
                parameters,
                optimizer_state,
                averaged,
                batch=training_batches[0],
                epoch=epoch,
            )
        heads = {}
        if config.objective == "mlpe":
            regional_training = {}
            for i, batch in enumerate(training_batches):
                result = checked_call(
                    refresh,
                    parameters[0],
                    parameters[1][i],
                    batch.payload,
                    config=config,
                    context=batch.context,
                    batch=batch,
                    epoch=epoch,
                )
                if not all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(result)):
                    raise FloatingPointError(
                        f"{batch.region.name}: nonfinite MLPE calibration at epoch {epoch}; "
                        "check nonconstant scores, identifiable endpoints and variance conditioning"
                    )
                nll, beta, center, scale, variances, posterior = result
                mean, covariance, factor = (np.asarray(array) for array in posterior)
                head = MLPEHead(
                    region_name=batch.region.name,
                    target=batch.observations.target,
                    population_ids=batch.region.sampling_unit_ids,
                    score_center=float(center),
                    score_scale=float(scale),
                    intercept=float(beta[0]),
                    slope=float(beta[1]),
                    unit_variance=float(variances[0]),
                    residual_variance=float(variances[1]),
                    effect_mean=tuple(mean.tolist()),
                    effect_covariance=tuple(map(tuple, covariance.tolist())),
                    effect_precision_cholesky=tuple(map(tuple, factor.tolist())),
                    calibration_pairs=training_pairs[batch.region.name],
                    calibration_roles=("training",) * len(batch.payload.targets),
                    ml_log_likelihood=-float(nll),
                    config=MLPEConfig(
                        variance_floor=config.mlpe_variance_floor, jitter=config.mlpe_jitter
                    ),
                    optimizer_iterations=epoch,
                    optimizer_message="Fixed-budget joint Adam; no variance convergence claim",
                    converged=False,
                )
                heads[batch.region.name] = head
                regional_training[batch.region.name] = float(nll) / len(batch.payload.targets)
            training_loss = float(np.mean(list(regional_training.values())))
        else:
            training_loss, regional_training = evaluate_regions(training_batches, epoch, heads)
        validation_loss, regional_validation = (
            (None, {})
            if not validation_batches
            else evaluate_regions(validation_batches, epoch, heads)
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
            on_epoch(continuation_at(epoch))
    continuation = continuation_at(config.epochs)
    return FitResult(selected, tuple(history), selected_epoch, selection, names, continuation)
