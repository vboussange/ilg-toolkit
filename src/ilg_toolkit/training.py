"""Species-balanced full-group fitting through native numerical pair kernels."""

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import NamedTuple, cast, final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from ._pairs import PairSelection
from .calibration import DirectCalibrationModule
from .config import TrainingConfig
from .data import ObservationPartition, PairwiseObservations, RegionBatch
from .model import LandscapeGeneticsModel
from .models import AbstractDistanceModel
from .resistance import ResistanceSolverContext

RegionCollection = Sequence[RegionBatch] | Mapping[str, RegionBatch]
ObservationCollection = Mapping[str, Mapping[str, PairwiseObservations]]
PartitionCollection = Mapping[str, Mapping[str, ObservationPartition | None]] | None
GroupPairs = dict[str, dict[str, tuple[tuple[str, str], ...]]]


@final
class StepRecord(eqx.Module):
    """One fully evaluated snapshot, with available species/group coverage."""

    step: int
    training_loss: float
    validation_loss: float | None = None
    training_by_species: dict[str, float] = eqx.field(default_factory=dict)
    training_by_group: dict[str, dict[str, float]] = eqx.field(default_factory=dict)
    validation_by_species: dict[str, float] = eqx.field(default_factory=dict)
    validation_by_group: dict[str, dict[str, float]] = eqx.field(default_factory=dict)
    validation_missing_groups: tuple[tuple[str, str], ...] = ()
    validation_missing_species: tuple[str, ...] = ()


@final
class TrainingState(eqx.Module):
    """Latest trainable parameters and an independently selected evaluated model.

    Optimizer progress is explicit; history records evaluated snapshots, not a
    progress counter. Names/provenance belong to host state, never numeric kernels.
    """

    step: int
    selected_step: int
    latest_distance_model: AbstractDistanceModel
    raw_variances: jax.Array | None
    selected_raw_variances: jax.Array | None
    optimizer_state: optax.OptState
    rng_key: jax.Array
    config: TrainingConfig
    input_contract: dict
    history: tuple[StepRecord, ...]
    best_model: LandscapeGeneticsModel

    @property
    def model(self) -> LandscapeGeneticsModel:
        return self.best_model

    @property
    def distance_model(self) -> AbstractDistanceModel:
        return self.latest_distance_model

    @property
    def region_names(self) -> tuple[str, ...]:
        return tuple(self.best_model.training_pairs)

    @property
    def selection(self) -> str:
        return "validation" if self.best_model.validation_pairs else "final"

    @property
    def best_loss(self) -> float:
        record = next(record for record in self.history if record.step == self.selected_step)
        return record.training_loss if record.validation_loss is None else record.validation_loss


class _GroupInput(NamedTuple):
    region: RegionBatch
    species: str
    observations: PairwiseObservations
    partition: ObservationPartition | None


class _TrainingPayload(NamedTuple):
    """Numerical data only; array IDs select fixed parameter rows dynamically."""

    features: jax.Array
    nodes: jax.Array
    pairs: tuple[jax.Array, jax.Array]
    targets: jax.Array
    distance_pairs: PairSelection
    species_index: jax.Array
    group_index: jax.Array
    weight: jax.Array


@dataclass(frozen=True)
class _GroupBatch:
    source: _GroupInput
    payload: _TrainingPayload
    context: ResistanceSolverContext | None
    region_index: int


def _normalize_inputs(
    regions: RegionCollection,
    observations: ObservationCollection,
    partitions: PartitionCollection,
    *,
    species=None,
) -> tuple[_GroupInput, ...]:
    """Validate nested host collections and sort presentation-independent groups."""
    if isinstance(regions, Mapping):
        ordered = list(regions.values())
        if any(
            not isinstance(value, RegionBatch) or key != value.name
            for key, value in regions.items()
        ):
            raise ValueError("Region mapping keys must match RegionBatch names")
    elif isinstance(regions, Sequence) and not isinstance(regions, (str, bytes)):
        ordered = list(regions)
    else:
        raise ValueError("regions must be a nonempty collection, including for one region")
    if not ordered or any(not isinstance(region, RegionBatch) for region in ordered):
        raise ValueError("regions must contain RegionBatch values")
    names = [region.name for region in ordered]
    if len(set(names)) != len(names):
        raise ValueError("Region names must be unique")
    if not isinstance(observations, Mapping) or set(observations) != set(names):
        raise ValueError("Observations must be region-then-species mappings matching region names")
    if partitions is not None and (
        not isinstance(partitions, Mapping) or set(partitions) != set(names)
    ):
        raise ValueError("Partitions must be region-then-species mappings matching region names")
    registry = (
        tuple(species)
        if species is not None
        else tuple(sorted({name for region in ordered for name in region.sampling_units}))
    )
    output = []
    for region in sorted(ordered, key=lambda value: value.name):
        if set(region.sampling_units) - set(registry):
            raise ValueError("Region sampling units include unknown registered species")
        groups = observations[region.name]
        if not isinstance(groups, Mapping) or set(groups) - set(region.sampling_units):
            raise ValueError("Observed species must have registered sampling units in their region")
        if not groups:
            raise ValueError("Each supplied region requires at least one observed species group")
        selections = None if partitions is None else partitions[region.name]
        if selections is not None and (
            not isinstance(selections, Mapping) or set(selections) != set(groups)
        ):
            raise ValueError("Partition species keys must match observed groups exactly")
        for name in registry:
            if name not in groups:
                continue
            observed = groups[name]
            selected = None if selections is None else selections[name]
            if not isinstance(observed, PairwiseObservations):
                raise ValueError("Observation groups require PairwiseObservations")
            if selected is not None and not isinstance(selected, ObservationPartition):
                raise ValueError("Partition groups require ObservationPartition or None")
            output.append(_GroupInput(region, name, observed, selected))
    if not output:
        raise ValueError("At least one eligible observed group is required")
    return tuple(output)


def _validate_contract(training, validation, registry):
    first = training[0]
    all_inputs = (*training, *validation)
    if set(registry) - {item.species for item in training}:
        raise ValueError("Every registered species requires observed training coverage")
    if len({item.region.name for item in all_inputs}) > 1 and first.region.feature_names is None:
        raise ValueError("Multiple regions require explicit feature_names")
    assigned, regions = {}, {}
    for item in all_inputs:
        region = item.region
        if item.observations.target != first.observations.target:
            raise ValueError("All region-species target contracts must match")
        if (
            region.feature_array.shape[-1] != first.region.feature_array.shape[-1]
            or region.feature_names != first.region.feature_names
        ):
            raise ValueError("Regional feature contracts must match meanings and order")
        if region.name in regions and not np.array_equal(
            regions[region.name].feature_array, region.feature_array
        ):
            raise ValueError("A named region must have the same raster across training/validation")
        regions[region.name] = region
        for name, units in region.sampling_units.items():
            for label, position, kind in zip(
                units.ids, np.asarray(units.grid_positions), units.kinds, strict=True
            ):
                identity = (name, label)
                definition = (region.name, tuple(position), kind)
                if identity in assigned and assigned[identity] != definition:
                    raise ValueError(
                        "A sampling unit must have one consistent region assignment "
                        "within its species"
                    )
                assigned[identity] = definition


def _prepared(item: _GroupInput, model, *, role, group_index, weight):
    if item.partition is not None and item.partition.role != role:
        raise ValueError(f"Expected a {role} partition, got {item.partition.role}")
    pairs, values = item.observations.aligned_pairs(item.region, item.species, item.partition)
    if item.observations.target.kind != "dissimilarity" or bool((values < 0).any()):
        raise ValueError("direct_log1p requires nonnegative dissimilarities")
    nodes = item.region._pixel_nodes(item.species)
    return _TrainingPayload(
        item.region.feature_array,
        nodes,
        pairs,
        item.observations.target.forward(values),
        model._prepare_pairs(item.region.feature_array, nodes, pairs),
        jnp.asarray(model.species.index(item.species), dtype=jnp.int32),
        jnp.asarray(group_index, dtype=jnp.int32),
        jnp.asarray(weight),
    )


def _prepare_batches(inputs, model, *, role):
    coverage = {name: sum(item.species == name for item in inputs) for name in model.species}
    present_species = sum(count > 0 for count in coverage.values())
    region_names = tuple(sorted({item.region.name for item in inputs}))
    contexts = {}
    batches = []
    for index, item in enumerate(inputs):
        shape = item.region.feature_array.shape
        if shape not in contexts:
            contexts[shape] = model.prepare(item.region.feature_array)
        payload = _prepared(
            item,
            model,
            role=role,
            group_index=index,
            weight=1 / (present_species * coverage[item.species]),
        )
        batches.append(
            _GroupBatch(item, payload, contexts[shape], region_names.index(item.region.name))
        )
    return tuple(batches)


def _labelled_pairs(batch):
    ids = batch.source.region.sampling_units[batch.source.species].ids
    return tuple(
        (ids[int(left)], ids[int(right)]) for left, right in zip(*batch.payload.pairs, strict=True)
    )


def _access(batches) -> GroupPairs:
    result = {}
    for batch in batches:
        result.setdefault(batch.source.region.name, {})[batch.source.species] = _labelled_pairs(
            batch
        )
    return result


def _input_contract(training, validation, registry):
    def fingerprint(array):
        value = np.asarray(array)
        return hashlib.sha256(value.tobytes()).hexdigest()

    def encode(inputs):
        return tuple(
            dict(
                region=item.region.name,
                species=item.species,
                shape=tuple(item.region.feature_array.shape),
                feature_dtype=str(item.region.feature_array.dtype),
                feature_sha256=fingerprint(item.region.feature_array),
                observation_sha256=fingerprint(item.observations.values),
                target_dtype=str(item.observations.values.dtype),
                units=item.region.sampling_units[item.species].ids,
                positions=tuple(
                    map(
                        tuple,
                        np.asarray(
                            item.region.sampling_units[item.species].grid_positions
                        ).tolist(),
                    )
                ),
                kinds=item.region.sampling_units[item.species].kinds,
                pairs=item.observations.observed_pairs,
                selected=None if item.partition is None else item.partition.pairs,
                target=asdict(item.observations.target),
                feature_names=item.region.feature_names,
            )
            for item in inputs
        )

    return dict(species=registry, training=encode(training), validation=encode(validation))


def _group_loss(model, payload, *, context=None, inference=True, key=None):
    distances = model._pair_distances(
        payload.features,
        payload.distance_pairs,
        payload.species_index,
        context=context,
        inference=inference,
        key=key,
    )
    if distances.shape != payload.targets.shape:
        raise ValueError("Distance model must return one value per prepared pair")
    distances = eqx.error_if(
        distances,
        jnp.any(~jnp.isfinite(distances) | (distances < 0)),
        "Distance model produced nonfinite or negative distances",
    )
    return jnp.mean((jnp.log1p(distances) - jnp.log1p(payload.targets)) ** 2)


def _weighted_loss(model, payload, *, context=None, key=None):
    return payload.weight * _group_loss(model, payload, context=context, inference=False, key=key)


_gradient = eqx.filter_value_and_grad(_weighted_loss)
_compiled_gradient = eqx.filter_jit(_gradient)
_compiled_loss = eqx.filter_jit(_group_loss)


def _checked_call[T](
    function: Callable[..., T],
    *args,
    batch: _GroupBatch | None,
    step: int,
    training_config: TrainingConfig,
    **kwargs,
) -> T:
    """Synchronize numerical work and attach identities without changing failures."""
    try:
        return jax.block_until_ready(function(*args, **kwargs))
    except RuntimeError as error:
        identity = (
            "joint groups"
            if batch is None
            else f"region {batch.source.region.name!r}, species {batch.source.species!r}"
        )
        solver = None if batch is None or batch.context is None else batch.context.config
        raise RuntimeError(
            f"{identity}: {training_config.objective} calculation failed at step {step}; "
            f"check finite distances/gradients and numerical conditioning; solver={solver}. "
            f"{error}"
        ) from error


def _apply_gradient(model, optimizer_state, gradient, *, learning_rate):
    optimizer = optax.adam(learning_rate)
    updates, optimizer_state = optimizer.update(
        cast(optax.Updates, gradient), optimizer_state, cast(optax.Params, model)
    )
    model = eqx.apply_updates(model, updates)
    bad = jnp.asarray(False)
    for leaf in jax.tree.leaves((model, optimizer_state)):
        if eqx.is_inexact_array(leaf):
            bad = bad | jnp.any(~jnp.isfinite(leaf))
    return eqx.error_if((model, optimizer_state), bad, "Nonfinite optimizer update")


_compiled_update = eqx.filter_jit(_apply_gradient)


def _evaluate(model, batches, *, config, step):
    group_values = {}
    species_values = {}
    evaluate = _compiled_loss if config.jit else _group_loss
    for batch in batches:
        value = float(
            _checked_call(
                evaluate,
                model,
                batch.payload,
                context=batch.context,
                batch=batch,
                step=step,
                training_config=config,
            )
        )
        if not np.isfinite(value):
            raise FloatingPointError(
                f"region {batch.source.region.name!r}, species {batch.source.species!r}: "
                f"nonfinite objective at step {step}"
            )
        group_values.setdefault(batch.source.region.name, {})[batch.source.species] = value
        species_values.setdefault(batch.source.species, []).append(value)
    by_species = {name: float(np.mean(values)) for name, values in species_values.items()}
    return float(np.mean(list(by_species.values()))), by_species, group_values


def fit(
    regions: RegionCollection,
    observations: ObservationCollection,
    *,
    distance_model: AbstractDistanceModel | None = None,
    config: TrainingConfig | None = None,
    validation: tuple[RegionCollection, ObservationCollection] | None = None,
    partition: PartitionCollection = None,
    validation_partition: PartitionCollection = None,
    state: TrainingState | None = None,
    on_step: Callable[[TrainingState], None] | None = None,
) -> TrainingState:
    """Fit native species groups with equal species and within-species region influence.

    All selected observations contribute their complete per-group mean objective.
    Validation selects evaluated snapshots and consumes no training random keys.
    Continuation permits only an increased total step budget on identical contracts.
    ``on_step`` observes completed optimizer updates, not synthetic initialization.
    """
    if state is not None and not isinstance(state, TrainingState):
        raise ValueError("state must be a TrainingState returned by fit")
    if state is not None:
        if distance_model is not None:
            raise ValueError("Continuation already contains a distance_model; omit distance_model")
        distance_model = state.latest_distance_model
        config = state.config if config is None else config
        old, new = asdict(state.config), asdict(config)
        old.pop("steps")
        new.pop("steps")
        if old != new or config.steps < state.config.steps:
            raise ValueError("Continuation requires unchanged policy and an increased step budget")
    if not isinstance(distance_model, AbstractDistanceModel) or not isinstance(
        config, TrainingConfig
    ):
        raise ValueError("Fresh fit requires an explicit native distance_model and TrainingConfig")
    if on_step is not None and not callable(on_step):
        raise ValueError("on_step must be callable or None")
    if config.objective != "direct_log1p":
        raise NotImplementedError(
            "Native group MLPE training is implemented in the dependent MLPE slice"
        )
    training = _normalize_inputs(regions, observations, partition, species=distance_model.species)
    if validation is None and validation_partition is not None:
        raise ValueError("validation_partition requires validation observations")
    if validation is not None and (not isinstance(validation, tuple) or len(validation) != 2):
        raise ValueError("validation must contain regions and nested observations")
    validating = (
        ()
        if validation is None
        else _normalize_inputs(
            validation[0], validation[1], validation_partition, species=distance_model.species
        )
    )
    _validate_contract(training, validating, distance_model.species)
    contract = _input_contract(training, validating, distance_model.species)
    if state is not None and state.input_contract != contract:
        raise ValueError("Continuation species, input, identity or target contracts changed")
    batches = _prepare_batches(training, distance_model, role="training")
    validation_batches = _prepare_batches(validating, distance_model, role="validation")
    training_pairs, validation_pairs = _access(batches), _access(validation_batches)
    for region, groups in validation_pairs.items():
        for species, pairs in groups.items():
            if set(map(lambda pair: tuple(sorted(pair)), pairs)) & set(
                tuple(sorted(pair)) for pair in training_pairs.get(region, {}).get(species, ())
            ):
                raise ValueError("Training and validation pairs overlap in a region-species group")
    missing_groups = tuple(
        (region, species)
        for region, groups in training_pairs.items()
        for species in groups
        if species not in validation_pairs.get(region, {})
    )
    validated_species = {name for groups in validation_pairs.values() for name in groups}
    missing_species = tuple(
        name for name in distance_model.species if name not in validated_species
    )
    optimizer_state = (
        optax.adam(config.learning_rate).init(eqx.filter(distance_model, eqx.is_inexact_array))
        if state is None
        else state.optimizer_state
    )
    random_key = jax.random.PRNGKey(config.seed) if state is None else state.rng_key
    model = distance_model
    history = [] if state is None else list(state.history)
    selected = None if state is None else state.best_model
    selected_step = 0 if state is None else state.selected_step
    best_loss = float("inf") if state is None else state.best_loss
    progress = 0 if state is None else state.step
    gradient = _compiled_gradient if config.jit else _gradient
    update = _compiled_update if config.jit else _apply_gradient

    def snapshot():
        assert selected is not None
        return TrainingState(
            progress,
            selected_step,
            model,
            None,
            None,
            optimizer_state,
            random_key,
            config,
            contract,
            tuple(history),
            selected,
        )

    start = 0 if state is None else state.step + 1
    for step in range(start, config.steps + 1):
        if step:
            random_key, update_key = jax.random.split(random_key)
            accumulated = None
            for batch in batches:
                key = jax.random.fold_in(update_key, batch.region_index)
                _, contribution = _checked_call(
                    gradient,
                    model,
                    batch.payload,
                    context=batch.context,
                    key=key,
                    batch=batch,
                    step=step,
                    training_config=config,
                )
                accumulated = (
                    contribution
                    if accumulated is None
                    else jax.tree.map(lambda total, value: total + value, accumulated, contribution)
                )
            model, optimizer_state = _checked_call(
                update,
                model,
                optimizer_state,
                accumulated,
                learning_rate=config.learning_rate,
                batch=None,
                step=step,
                training_config=config,
            )
        progress = step
        loss, by_species, by_group = _evaluate(model, batches, config=config, step=step)
        val_loss, val_species, val_group = (
            (None, {}, {})
            if not validation_batches
            else _evaluate(model, validation_batches, config=config, step=step)
        )
        history.append(
            StepRecord(
                step,
                loss,
                val_loss,
                by_species,
                by_group,
                val_species,
                val_group,
                missing_groups,
                missing_species,
            )
        )
        if val_loss is None or val_loss < best_loss:
            selected_step = step
            best_loss = loss if val_loss is None else val_loss
            selected = LandscapeGeneticsModel(
                model,
                DirectCalibrationModule(training[0].observations.target),
                training[0].region.feature_array.shape[-1],
                training[0].region.feature_names,
                training_objective=config.objective,
                training_pairs=training_pairs,
                validation_pairs=validation_pairs,
            )
        if step and on_step is not None:
            on_step(snapshot())
    return snapshot()
