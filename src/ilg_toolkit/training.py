"""Joint fitting with equal species influence and equal region influence within species.

Maximum-likelihood population effects (MLPE) calibration is fitted with the distance model."""

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import NamedTuple, cast, final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .calibration import DirectCalibrationModule, GroupKey, MLPECalibrationModule
from .config import TrainingConfig
from .data import ObservationPartition, PairwiseObservations, RegionBatch
from .mlpe import (
    MLPEConfig,
    RegionalMLPECalibration,
    finalize_mlpe_fit,
    mlpe_ml_negative_log_likelihood,
    profiled_mlpe_ml_fit,
)
from .mlpe.likelihood import MLPEFit
from .model import LandscapeGeneticsModel
from .models import AbstractDistanceModel
from .models.base import distances_from_latent, prepare_distance_context, region_for_species
from .models.common import with_dropout_mode
from .resistance import ResistanceSolverContext

RegionCollection = RegionBatch | Sequence[RegionBatch] | Mapping[str, RegionBatch]
RegionObservations = PairwiseObservations | Sequence[PairwiseObservations]
ObservationCollection = (
    RegionObservations | Sequence[RegionObservations] | Mapping[str, RegionObservations]
)
RegionPartitions = ObservationPartition | Sequence[ObservationPartition | None] | None
PartitionCollection = RegionPartitions | Sequence[RegionPartitions] | Mapping[str, RegionPartitions]
Parameters = tuple[AbstractDistanceModel, dict[GroupKey, jax.Array] | None]


@final
class StepRecord(eqx.Module):
    """Inference objectives after an update; zero is initialization.

    The aggregate averages per-pair objectives across regions within each species,
    then across species.
    """

    step: int
    training_loss: float
    validation_loss: float | None = None
    training_by_region: dict[str, float] = eqx.field(default_factory=dict)
    validation_by_region: dict[str, float] = eqx.field(default_factory=dict)
    training_by_species: dict[str, float] | None = None
    validation_by_species: dict[str, float] | None = None


@final
class TrainingState(eqx.Module):
    """Training progress, evaluated models, and state for continuing a fit.

    ``selected_model`` retains the distance model and calibration selected among
    evaluated steps; ``model`` returns it. ``latest_model`` holds the latest
    distance parameters and calibration from the last evaluation, alongside
    the optimizer state and random key needed to resume at ``step``.
    """

    raw_variances: dict[GroupKey, jax.Array] | None
    optimizer_state: optax.OptState
    rng_key: jax.Array
    config: TrainingConfig
    history: tuple[StepRecord, ...]
    latest_model: LandscapeGeneticsModel
    selected_model: LandscapeGeneticsModel
    selected_step: int
    step: int

    @property
    def model(self) -> LandscapeGeneticsModel:
        """The selected model retained from evaluated training steps."""
        return self.selected_model

    @property
    def distance_model(self) -> AbstractDistanceModel:
        return self.latest_model.distance_model

    @property
    def region_names(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    key if isinstance(key, str) else key[0]
                    for key in self.latest_model.training_pairs
                }
            )
        )

    @property
    def selection(self) -> str:
        return "validation" if self.latest_model.validation_pairs else "final"

    @property
    def selected_loss(self) -> float:
        record = next(record for record in self.history if record.step == self.selected_step)
        return record.training_loss if record.validation_loss is None else record.validation_loss


def _validate_state(state: TrainingState) -> None:
    if not 0 <= state.step <= state.config.steps:
        raise ValueError("Training step must be within the configured budget")
    if not 0 <= state.selected_step <= state.step:
        raise ValueError("Selected step must be within the completed training steps")
    if not 1 <= len(state.history) <= state.step + 1:
        raise ValueError("History length must be within the completed training steps plus one")


class _TrainingPayload(NamedTuple):
    """Regional features, sampling-unit nodes, and aligned training observations."""

    features: jax.Array
    nodes: jax.Array
    pairs: tuple[jax.Array, jax.Array]
    targets: jax.Array


class _RegionalInput(NamedTuple):
    region: RegionBatch
    observations: PairwiseObservations
    partition: ObservationPartition | None


class _GroupInput(NamedTuple):
    region: RegionBatch
    observations: PairwiseObservations
    partition: ObservationPartition | None
    species_index: int
    identity: GroupKey
    weight: float


@dataclass(frozen=True)
class _RegionalBatch:
    """One region–species group's observations, numerical settings, and objective weight."""

    region: RegionBatch
    observations: PairwiseObservations
    payload: _TrainingPayload
    species_index: int
    identity: GroupKey
    weight: float
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
            raise ValueError("Population MLPE requires dissimilarities")
        kinds = region.sampling_unit_kinds
        assert kinds is not None
        if any(kind != "population" for kind in kinds):
            raise ValueError("Population MLPE does not support individual sampling units")
        if role == "training" and len(values) < 3:
            raise ValueError("MLPE training requires at least three observed pairs")
        if role == "training":
            degree = jnp.bincount(jnp.concatenate(pairs), length=len(region.sampling_unit_ids))
            if degree.max() <= 1:
                raise ValueError("MLPE variances are unidentifiable: pairs share no sampling units")
    targets = observations.target.forward(values)
    return _TrainingPayload(
        features=region.feature_array,
        nodes=region.pixel_nodes,
        pairs=pairs,
        targets=targets.astype(jnp.float64) if objective == "mlpe" else targets,
    )


def _align_collection(values: object, names: Sequence[str], description: str) -> tuple[object, ...]:
    """A labeled collection aligned with its region batches."""
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
    """Aligned collections with region identities sorted for reproducible random sampling."""
    if isinstance(regions, RegionBatch):
        ordered = [regions]
    elif isinstance(regions, Mapping):
        ordered = list(regions.values())
        if any(
            not isinstance(value, RegionBatch) or key != value.name
            for key, value in regions.items()
        ):
            raise ValueError("Region mapping keys must match each region batch's name")
    elif isinstance(regions, Sequence) and not isinstance(regions, (str, bytes)):
        ordered = list(regions)
    else:
        raise ValueError("Provide a region batch or a nonempty region sequence/mapping")
    if not ordered or any(not isinstance(region, RegionBatch) for region in ordered):
        raise ValueError("Regions must be a nonempty collection of RegionBatch inputs")
    names = [region.name for region in ordered]
    if len(set(names)) != len(names):
        raise ValueError("Region names must be unique; duplicate names cannot share a fit")

    observed = (
        (observations,)
        if isinstance(regions, RegionBatch) and not isinstance(observations, Mapping)
        else _align_collection(observations, names, "Observations")
    )
    selected = (
        (None,) * len(ordered)
        if partitions is None
        else (partitions,)
        if isinstance(regions, RegionBatch) and not isinstance(partitions, Mapping)
        else _align_collection(partitions, names, "Partitions")
    )
    inputs = []
    for region, observation, partition in zip(ordered, observed, selected, strict=True):
        group_observations = (
            (observation,) if isinstance(observation, PairwiseObservations) else observation
        )
        if not isinstance(group_observations, Sequence) or not group_observations:
            raise ValueError("Each region requires observations for at least one species")
        group_partitions = (
            tuple(partition)
            if isinstance(partition, Sequence)
            else (partition,) * len(group_observations)
        )
        if len(group_partitions) != len(group_observations):
            raise ValueError("Partitions must align with each region's observations")
        for observed_group, selected_group in zip(
            group_observations, group_partitions, strict=True
        ):
            if not isinstance(observed_group, PairwiseObservations):
                raise ValueError("Observations contain unsupported input types")
            if selected_group is not None and not isinstance(selected_group, ObservationPartition):
                raise ValueError("Partitions contain unsupported input types")
            inputs.append(_RegionalInput(region, observed_group, selected_group))
    return tuple(sorted(inputs, key=lambda item: item.region.name))


def _index_inputs(
    inputs: Sequence[_RegionalInput], order: tuple[str, ...] | None, num_species: int
) -> tuple[_GroupInput, ...]:
    groups = []
    identities = set()
    for region, observations, partition in inputs:
        name = observations.species
        if name is None and order is not None and len(order) == 1:
            name = order[0]
        if order is None:
            if name is not None or region.species is not None:
                raise ValueError("Named observations require a species order")
            index = 0
        else:
            if name not in order or (num_species > 1 and region.species is None):
                raise ValueError("Observations and sampling units must identify a modeled species")
            index = order.index(name)
            if region.species is not None and not set(region.species).issubset(order):
                raise ValueError("Sampling-unit species must belong to the model species order")
        identity = region.name if num_species == 1 else (region.name, index)
        if identity in identities:
            raise ValueError("Provide one observations object per region–species group")
        identities.add(identity)
        selected = region_for_species(region, name)
        groups.append(_GroupInput(selected, observations, partition, index, identity, 0.0))
    coverage = Counter(group.species_index for group in groups)
    return tuple(
        group._replace(weight=1 / (len(coverage) * coverage[group.species_index]))
        for group in sorted(groups, key=lambda group: (group.region.name, group.species_index))
    )


def _group_objective(
    parameters: tuple[AbstractDistanceModel, jax.Array, jax.Array | None],
    payload: _TrainingPayload,
    *,
    species_index: int,
    objective: str,
    mlpe: MLPEConfig,
    context: ResistanceSolverContext | None = None,
    moments: tuple[jax.Array, jax.Array, jax.Array] | None = None,
    finalize: bool = False,
) -> tuple[jax.Array, MLPEFit | None]:
    distance_model, latent, raw_variances = parameters
    distances = distances_from_latent(
        distance_model,
        latent,
        payload.nodes,
        payload.pairs,
        raster_shape=(payload.features.shape[0], payload.features.shape[1]),
        species_index=species_index,
        context=context,
    )
    if objective == "direct_log1p":
        return jnp.mean(jnp.square(jnp.log1p(distances) - jnp.log1p(payload.targets))), None
    assert raw_variances is not None
    distances = distances.astype(jnp.float64)
    options = dict(
        variance_floor=mlpe.variance_floor,
        jitter=mlpe.jitter,
        min_distance_scale=mlpe.min_distance_scale,
    )
    result = None
    if moments is not None:
        center, scale, beta = moments
        nll = mlpe_ml_negative_log_likelihood(
            distances,
            payload.targets,
            *payload.pairs,
            n_populations=len(payload.nodes),
            raw_variances=raw_variances,
            fixed_effects=beta,
            distance_center=center,
            distance_scale=scale,
            **options,
        )
    elif finalize:
        result = finalize_mlpe_fit(
            distances,
            payload.targets,
            *payload.pairs,
            n_populations=len(payload.nodes),
            raw_variances=raw_variances,
            **options,
        )
        nll = result.nll
    else:
        nll, _ = profiled_mlpe_ml_fit(
            distances,
            payload.targets,
            *payload.pairs,
            n_populations=len(payload.nodes),
            raw_variances=raw_variances,
            **options,
        )
    return nll / len(distances), result


_compiled_objective = eqx.filter_jit(_group_objective)


class _Contribution(NamedTuple):
    payload: _TrainingPayload
    identity: GroupKey
    species_index: int
    weight: float


def _microbatch_objective(parameters, contributions, objective, mlpe, context, key):
    model, latent, raw = parameters
    configured = with_dropout_mode(model, inference=False, key=key)
    return sum(
        item.weight
        * _group_objective(
            (configured, latent, None if raw is None else raw[item.identity]),
            item.payload,
            species_index=item.species_index,
            objective=objective,
            mlpe=mlpe,
            context=context,
        )[0]
        for item in contributions
    )


_microbatch_gradient = eqx.filter_grad(_microbatch_objective)
_compiled_microbatch_gradient = eqx.filter_jit(_microbatch_gradient)


def _backbone(model, features, key):
    return with_dropout_mode(model, inference=key is None, key=key).backbone(features)


_compiled_backbone = eqx.filter_jit(_backbone)


def _regional_gradient(parameters, contributions, config, context, key):
    model, raw = parameters
    features = contributions[0].payload.features
    backbone = _compiled_backbone if config.jit else _backbone
    latent, backward = eqx.filter_vjp(lambda m: backbone(m, features, key), model)
    differentiate = _compiled_microbatch_gradient if config.jit else _microbatch_gradient
    size = config.species_microbatch_size or len(contributions)
    accumulated = None
    for start in range(0, len(contributions), size):
        gradient = differentiate(
            (model, latent, raw),
            contributions[start : start + size],
            config.objective,
            config.mlpe,
            context,
            key,
        )
        accumulated = (
            gradient
            if accumulated is None
            else jax.tree.map(lambda a, b: a + b, accumulated, gradient)
        )
    assert accumulated is not None
    model_gradient, latent_gradient, variance_gradient = accumulated
    model_gradient = jax.tree.map(lambda a, b: a + b, model_gradient, backward(latent_gradient)[0])
    return model_gradient, variance_gradient


def _labelled_pairs(batch: _RegionalBatch) -> tuple[tuple[str, str], ...]:
    return tuple(
        (
            min(batch.region.sampling_unit_ids[i], batch.region.sampling_unit_ids[j]),
            max(batch.region.sampling_unit_ids[i], batch.region.sampling_unit_ids[j]),
        )
        for i, j in zip(*batch.payload.pairs, strict=True)
    )


def _initial_raw_variances(
    batches: Sequence[_RegionalBatch], config: TrainingConfig
) -> dict[GroupKey, jax.Array]:
    values = {}
    for batch in batches:
        if config.mlpe.initial_variances is None:
            variance = jnp.maximum(
                jnp.var(batch.payload.targets, ddof=1),
                config.mlpe.variance_floor * 100,
            )
            pair = jnp.maximum(variance * jnp.asarray([0.25, 0.5]), config.mlpe.variance_floor * 10)
        else:
            pair = jnp.asarray(config.mlpe.initial_variances, dtype=jnp.float64)
        positive = pair - config.mlpe.variance_floor
        # log(expm1(x)) without overflowing for large user-supplied initial variances.
        raw = positive + jnp.log(-jnp.expm1(-positive))
        values[batch.identity] = raw
    return values


def _prepare_batches(
    training_inputs: Sequence[_GroupInput],
    validation_inputs: Sequence[_GroupInput],
    distance_model: AbstractDistanceModel,
    config: TrainingConfig,
) -> tuple[tuple[_RegionalBatch, ...], tuple[_RegionalBatch, ...]]:
    """Regional genetic targets with reusable resistance solvers."""
    contexts: dict[tuple[int, int], ResistanceSolverContext | None] = {}

    def make_batch(inputs: _GroupInput, role: str) -> _RegionalBatch:
        region, observations, partition, species_index, identity, weight = inputs
        payload = _prepared(region, observations, partition, role=role, objective=config.objective)
        shape = (region.feature_array.shape[0], region.feature_array.shape[1])
        if shape not in contexts:
            contexts[shape] = prepare_distance_context(distance_model, region.feature_array)
        context = contexts[shape]
        return _RegionalBatch(
            region, observations, payload, species_index, identity, weight, context
        )

    training = tuple(make_batch(inputs, "training") for inputs in training_inputs)
    validation = tuple(make_batch(inputs, "validation") for inputs in validation_inputs)
    training_by_name = {batch.identity: batch for batch in training}
    for batch in validation:
        training_batch = training_by_name.get(batch.identity)
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


def _apply_gradient(
    parameters: Parameters,
    optimizer_state: optax.OptState,
    gradient: Parameters,
    *,
    learning_rate: jax.Array,
) -> tuple[Parameters, optax.OptState]:
    optimizer = optax.adam(learning_rate)
    # Optax accepts registered Equinox pytrees; its array-container alias omits Modules.
    updates, optimizer_state = optimizer.update(
        cast(optax.Updates, gradient), optimizer_state, cast(optax.Params, parameters)
    )
    parameters = eqx.apply_updates(parameters, updates)
    return parameters, optimizer_state


_compiled_update = eqx.filter_jit(_apply_gradient)


def _evaluate_regions(
    parameters: Parameters,
    batches: Sequence[_RegionalBatch],
    config: TrainingConfig,
    step: int,
    calibrations: Mapping[GroupKey, RegionalMLPECalibration] | None = None,
    species: tuple[str, ...] | None = None,
) -> tuple[
    float, dict[str, float], dict[GroupKey, RegionalMLPECalibration], dict[str, float] | None
]:
    """Regional objectives evaluated with calibration fitted on training observations."""
    evaluate = _compiled_objective if config.jit else _group_objective
    values, fitted = {}, {}
    configured = with_dropout_mode(parameters[0], inference=True, key=None)
    previous_region, latent = None, None
    for batch in batches:
        if batch.region.name != previous_region:
            backbone = _compiled_backbone if config.jit else _backbone
            latent = backbone(parameters[0], batch.payload.features, None)
            previous_region = batch.region.name
        assert latent is not None
        moments = None
        if config.objective == "mlpe" and calibrations is not None:
            calibration = calibrations[batch.identity]
            moments = (
                jnp.asarray(calibration.distance_center),
                jnp.asarray(calibration.distance_scale),
                jnp.asarray((calibration.intercept, calibration.slope)),
            )
        value, result = evaluate(
            (configured, latent, None if parameters[1] is None else parameters[1][batch.identity]),
            batch.payload,
            species_index=batch.species_index,
            objective=config.objective,
            mlpe=config.mlpe,
            context=batch.context,
            moments=moments,
            finalize=calibrations is None,
        )
        value = float(value)
        if not np.isfinite(value) or (
            result is not None
            and not all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(result))
        ):
            raise FloatingPointError(
                f"{batch.region.name}, species {batch.species_index}: "
                f"nonfinite {config.objective} objective at step {step}; "
                "check distances and variance conditioning"
            )
        if result is not None:
            fitted[batch.identity] = RegionalMLPECalibration.from_fit(
                result,
                region_name=batch.region.name,
                target=batch.observations.target,
                population_ids=batch.region.sampling_unit_ids,
                calibration_pairs=_labelled_pairs(batch),
                calibration_roles=("training",) * len(batch.payload.targets),
                config=config.mlpe,
                optimizer_iterations=step,
                optimizer_message="Fixed-budget joint Adam; no variance convergence claim",
                converged=False,
            )
        values[batch.identity] = value
    regional = {
        name: float(np.mean([values[b.identity] for b in batches if b.region.name == name]))
        for name in dict.fromkeys(b.region.name for b in batches)
    }
    if parameters[0].num_species > 1:
        assert species is not None
    by_species = (
        None
        if parameters[0].num_species == 1 or species is None
        else {
            species[index]: float(
                np.mean([values[b.identity] for b in batches if b.species_index == index])
            )
            for index in sorted({b.species_index for b in batches})
        }
    )
    return (
        sum(batch.weight * values[batch.identity] for batch in batches),
        regional,
        fitted,
        by_species,
    )


def fit(
    region: RegionCollection,
    observations: ObservationCollection,
    *,
    distance_model: AbstractDistanceModel | None = None,
    config: TrainingConfig | None = None,
    species: Sequence[str] | None = None,
    validation: tuple[RegionCollection, ObservationCollection] | None = None,
    partition: PartitionCollection = None,
    validation_partition: PartitionCollection = None,
    state: TrainingState | None = None,
    on_step: Callable[[TrainingState], None] | None = None,
) -> TrainingState:
    """Joint model fitting with equal species and within-species region influence.

    ``steps=0`` fits genetic calibration on training observations while keeping
    the distance model fixed. Evaluation at step zero, every ``eval_every``
    updates, and the final step selects by validation loss or uses the final
    evaluated model when validation is absent. ``on_step`` receives resumable
    state after initialization and every update; ``state`` continues the saved
    trajectory with the same inputs and an equal or increased step budget.
    """
    if species is not None and (
        isinstance(species, (str, bytes)) or not isinstance(species, Sequence)
    ):
        raise ValueError("Species order must be a sequence of names")
    if state is not None and not isinstance(state, TrainingState):
        raise ValueError("state must be a TrainingState returned by fit")
    if state is not None and distance_model is not None:
        raise ValueError(
            "A continuation state already contains its distance model; omit distance_model"
        )
    if state is None:
        if distance_model is None:
            raise ValueError("Fresh fitting requires a distance_model")
        if config is None:
            raise ValueError("Fresh fitting requires a TrainingConfig with objective")
    else:
        _validate_state(state)
        distance_model = state.distance_model
        config = state.config if config is None else config
        if species is not None and tuple(species) != state.latest_model.species:
            raise ValueError("Continuation must retain the saved species order")
        species = state.latest_model.species
    if on_step is not None and not callable(on_step):
        raise ValueError("on_step must be callable or None")
    if state is not None:
        old, new = asdict(state.config), asdict(config)
        old.pop("steps")
        new.pop("steps")
        if old != new or config.steps < state.config.steps:
            raise ValueError(
                "Continuation requires identical configuration except an increased step budget"
            )
    if config.objective == "mlpe" and not jax.config.read("jax_enable_x64"):
        raise RuntimeError(
            "MLPE fitting requires float64: set JAX_ENABLE_X64=1 or use jax.enable_x64()"
        )
    if distance_model.num_species > 1 and species is None:
        raise ValueError("A multispecies distance model requires species= in parameter order")
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
        raise ValueError("Shared multi-region fitting requires feature_names meanings and order")
    for prepared_region, regional_observations, _ in (*training_inputs, *validation_inputs):
        if regional_observations.target != first_observations.target:
            raise ValueError("Regional training and validation target scales must match")
        if prepared_region.feature_array.shape[-1] != first_region.feature_array.shape[-1]:
            raise ValueError("Regional training and validation feature channels must match")
        if prepared_region.feature_names != first_region.feature_names:
            raise ValueError("Regional feature contracts must match meanings and order")
    if species is None:
        names = {name for item in training_inputs for name in (item.region.species or ())}
        names.update(
            item.observations.species
            for item in training_inputs
            if item.observations.species is not None
        )
        species = tuple(sorted(names)) if names else None
    prototype = LandscapeGeneticsModel(
        distance_model,
        DirectCalibrationModule(first_observations.target),
        first_region.feature_array.shape[-1],
        feature_names=first_region.feature_names,
        training_objective=config.objective,
        species=None if species is None else tuple(species),
    )
    species = prototype.species
    indexed_training = _index_inputs(training_inputs, species, distance_model.num_species)
    indexed_validation = _index_inputs(validation_inputs, species, distance_model.num_species)
    random_key = (
        jax.random.split(jax.random.PRNGKey(config.seed))[1] if state is None else state.rng_key
    )
    training_batches, validation_batches = _prepare_batches(
        indexed_training, indexed_validation, distance_model, config
    )

    raw_variances = (
        state.raw_variances
        if state is not None
        else _initial_raw_variances(training_batches, config)
        if config.objective == "mlpe"
        else None
    )
    parameters = (distance_model, raw_variances)
    optimizer = optax.adam(config.learning_rate)
    optimizer_state = (
        state.optimizer_state
        if state is not None
        else optimizer.init(eqx.filter(parameters, eqx.is_inexact_array))
    )

    apply_update = _compiled_update if config.jit else _apply_gradient
    regional_batches = tuple(
        tuple(b for b in training_batches if b.region.name == name)
        for name in dict.fromkeys(b.region.name for b in training_batches)
    )
    region_count = len(regional_batches)
    selected_count = min(region_count, config.region_batch_size or region_count)

    training_pairs = {batch.identity: _labelled_pairs(batch) for batch in training_batches}
    validation_pairs = {batch.identity: _labelled_pairs(batch) for batch in validation_batches}

    prototype = eqx.tree_at(
        lambda m: (m.training_pairs, m.validation_pairs),
        prototype,
        (training_pairs, validation_pairs),
    )

    def make_model(calibrations: dict[GroupKey, RegionalMLPECalibration]) -> LandscapeGeneticsModel:
        calibration = (
            MLPECalibrationModule(first_observations.target, calibrations)
            if config.objective == "mlpe"
            else DirectCalibrationModule(first_observations.target)
        )
        return eqx.tree_at(
            lambda m: (m.distance_model, m.calibration), prototype, (parameters[0], calibration)
        )

    history = list(state.history) if state is not None else []
    selected = state.selected_model if state is not None else None
    latest = state.latest_model if state is not None else None
    selected_step = state.selected_step if state is not None else 0
    selected_loss = state.selected_loss if state is not None else float("inf")

    def continuation() -> TrainingState:
        assert latest is not None and selected is not None
        return TrainingState(
            raw_variances=parameters[1],
            optimizer_state=optimizer_state,
            rng_key=random_key,
            config=config,
            history=tuple(history),
            latest_model=latest,
            selected_model=selected,
            selected_step=selected_step,
            step=step,
        )

    step = state.step if state is not None else 0
    start_step = state.step + 1 if state is not None else 0
    for step in range(start_step, config.steps + 1):
        if step:
            random_key, update_key = jax.random.split(random_key)
            keys = jax.random.split(update_key, region_count)
            indices = (
                range(region_count)
                if selected_count == region_count
                else sorted(
                    np.asarray(
                        jax.random.choice(
                            jax.random.fold_in(update_key, region_count),
                            region_count,
                            (selected_count,),
                            replace=False,
                        )
                    ).tolist()
                )
            )
            accumulated = None
            for index in indices:
                groups = regional_batches[index]
                species_count = len(groups)
                sample_count = min(species_count, config.species_batch_size or species_count)
                chosen = (
                    range(species_count)
                    if sample_count == species_count
                    else sorted(
                        np.asarray(
                            jax.random.choice(
                                jax.random.fold_in(keys[index], species_count),
                                species_count,
                                (sample_count,),
                                replace=False,
                            )
                        ).tolist()
                    )
                )
                correction = region_count / selected_count * species_count / sample_count
                contributions = tuple(
                    _Contribution(
                        groups[i].payload,
                        groups[i].identity,
                        groups[i].species_index,
                        groups[i].weight * correction,
                    )
                    for i in chosen
                )
                gradient = _regional_gradient(
                    parameters, contributions, config, groups[0].context, keys[index]
                )
                accumulated = (
                    gradient
                    if accumulated is None
                    else jax.tree.map(
                        lambda total, contribution: total + contribution, accumulated, gradient
                    )
                )
            assert accumulated is not None
            parameters, optimizer_state = apply_update(
                parameters,
                optimizer_state,
                accumulated,
                learning_rate=jnp.asarray(config.learning_rate),
            )
        if step == 0 or step % config.eval_every == 0 or step == config.steps:
            training_loss, regional_training, calibrations, species_training = _evaluate_regions(
                parameters, training_batches, config, step, species=species
            )
            validation_loss, regional_validation, _, species_validation = (
                (None, {}, {}, None if distance_model.num_species == 1 else {})
                if not validation_batches
                else _evaluate_regions(
                    parameters, validation_batches, config, step, calibrations, species
                )
            )
            history.append(
                StepRecord(
                    step,
                    training_loss,
                    validation_loss,
                    regional_training,
                    regional_validation,
                    species_training,
                    species_validation,
                )
            )
            latest = make_model(calibrations)
            if validation_loss is None or validation_loss < selected_loss:
                selected, selected_step = latest, step
                selected_loss = training_loss if validation_loss is None else validation_loss
        else:
            assert latest is not None
            latest = eqx.tree_at(lambda model: model.distance_model, latest, parameters[0])
        if on_step is not None:
            on_step(continuation())
    return continuation()
