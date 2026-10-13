"""Species-balanced full-group fitting through native numerical pair kernels."""

import hashlib
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import NamedTuple, cast, final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import _training_execution as execution
from ._pairs import PairSelection
from .calibration import DirectCalibrationModule, UnpooledMLPECalibrationModule
from .config import TrainingConfig
from .data import ObservationPartition, PairwiseObservations, RegionBatch
from .mlpe import (
    MLPECalibration,
    MLPEConfig,
    finalize_mlpe_fit,
    mlpe_ml_negative_log_likelihood,
    profiled_mlpe_ml_fit,
)
from .mlpe.likelihood import MLPEFit
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
    progress counter. MLPE nuisance arrays have one two-variance row per entry in
    input_contract["training"], ordered by region name then registry order. The
    selected array matches best_model; the latest array follows optimizer progress.
    Names/provenance belong to host state, never numeric kernels.
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
    assigned, regions, checked = {}, {}, set()
    for item in all_inputs:
        region = item.region
        if item.observations.target != first.observations.target:
            raise ValueError("All region-species target contracts must match")
        # Flattened observed groups can share one RegionBatch. Validate all its
        # declarations once, including unobserved species. Separate validation
        # RegionBatch objects still require a complete consistency check.
        if id(region) in checked:
            continue
        checked.add(id(region))
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


def _prepared(item: _GroupInput, model, *, role, group_index, weight, objective):
    if item.partition is not None and item.partition.role != role:
        raise ValueError(f"Expected a {role} partition, got {item.partition.role}")
    pairs, values = item.observations.aligned_pairs(item.region, item.species, item.partition)
    if objective == "direct_log1p" and (
        item.observations.target.kind != "dissimilarity" or bool((values < 0).any())
    ):
        raise ValueError("direct_log1p requires nonnegative dissimilarities")
    if objective == "mlpe":
        if item.observations.target.kind != "dissimilarity":
            raise ValueError("Population MLPE requires declared dissimilarities")
        units = item.region.sampling_units[item.species]
        assert units.kinds is not None
        if any(kind != "population" for kind in units.kinds):
            raise ValueError("Population MLPE does not support individual sampling units")
        if role == "training":
            if len(values) < 3:
                raise ValueError("MLPE training requires at least three observed pairs")
            degree = jnp.bincount(jnp.concatenate(pairs), length=len(units.ids))
            if degree.max() <= 1:
                raise ValueError("MLPE variances are unidentifiable: pairs share no endpoints")
    targets = item.observations.target.forward(values)
    if objective == "mlpe":
        targets = targets.astype(jnp.float64)
    nodes = item.region._pixel_nodes(item.species)
    return _TrainingPayload(
        item.region.feature_array,
        nodes,
        pairs,
        targets,
        model._prepare_pairs(item.region.feature_array, nodes, pairs),
        jnp.asarray(model.species.index(item.species), dtype=jnp.int32),
        jnp.asarray(group_index, dtype=jnp.int32),
        jnp.asarray(weight),
    )


def _prepare_batches(inputs, model, *, role, objective):
    coverage = Counter(item.species for item in inputs)
    present_species = len(coverage)
    region_indices = {
        name: index for index, name in enumerate(sorted({item.region.name for item in inputs}))
    }
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
            objective=objective,
            group_index=index,
            weight=1 / (present_species * coverage[item.species]),
        )
        batches.append(
            _GroupBatch(item, payload, contexts[shape], region_indices[item.region.name])
        )
    return tuple(batches)


def _by_region(batches):
    regions = {}
    for batch in batches:
        regions.setdefault(batch.region_index, []).append(batch)
    return tuple(tuple(groups) for groups in regions.values())


def _sample_regions(regions, key, config):
    """Horvitz–Thompson weights for uniform region-then-species sampling."""
    count = len(regions)
    selected_count = min(count, config.region_batch_size or count)
    sampling_key = jax.random.fold_in(key, 0x53414D50)
    indices = (
        range(count)
        if selected_count == count
        else sorted(
            np.asarray(
                jax.random.choice(sampling_key, count, (selected_count,), replace=False)
            ).tolist()
        )
    )
    for index in indices:
        groups = regions[index]
        chosen_count = min(len(groups), config.species_batch_size or len(groups))
        chosen = (
            range(len(groups))
            if chosen_count == len(groups)
            else sorted(
                np.asarray(
                    jax.random.choice(
                        jax.random.fold_in(sampling_key, index + 1),
                        len(groups),
                        (chosen_count,),
                        replace=False,
                    )
                ).tolist()
            )
        )
        probability = (selected_count / count) * (chosen_count / len(groups))
        yield tuple(
            replace(
                groups[j],
                payload=groups[j].payload._replace(weight=groups[j].payload.weight / probability),
            )
            for j in chosen
        )


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

    region_records = {}
    for item in (*training, *validation):
        region = item.region
        if id(region) not in region_records:
            region_records[id(region)] = dict(
                region=region.name,
                shape=tuple(region.feature_array.shape),
                feature_dtype=str(region.feature_array.dtype),
                feature_sha256=fingerprint(region.feature_array),
                feature_names=region.feature_names,
                sampling_units=tuple(
                    dict(
                        species=name,
                        units=region.sampling_units[name].ids,
                        positions=tuple(
                            map(
                                tuple,
                                np.asarray(region.sampling_units[name].grid_positions).tolist(),
                            )
                        ),
                        kinds=region.sampling_units[name].kinds,
                    )
                    for name in registry
                    if name in region.sampling_units
                ),
            )

    def declared(inputs):
        return tuple(
            region_records[id(region)]
            for region in {item.region.name: item.region for item in inputs}.values()
        )

    def encode(inputs):
        return tuple(
            dict(
                region=item.region.name,
                species=item.species,
                shape=tuple(item.region.feature_array.shape),
                feature_dtype=str(item.region.feature_array.dtype),
                feature_sha256=region_records[id(item.region)]["feature_sha256"],
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

    return dict(
        species=registry,
        training=encode(training),
        validation=encode(validation),
        training_regions=declared(training),
        validation_regions=declared(validation),
    )


def _selected_distances(model, payload, *, context=None, inference=True, key=None):
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
    return distances


def _group_loss(model, payload, *, context=None, inference=True, key=None):
    distances = _selected_distances(model, payload, context=context, inference=inference, key=key)
    return jnp.mean((jnp.log1p(distances) - jnp.log1p(payload.targets)) ** 2)


def _weighted_loss(model, payload, *, context=None, key=None):
    return payload.weight * _group_loss(model, payload, context=context, inference=False, key=key)


_gradient = eqx.filter_value_and_grad(_weighted_loss)
_compiled_gradient = eqx.filter_jit(_gradient)
_compiled_loss = eqx.filter_jit(_group_loss)


def _mlpe_loss(parameters, payload, *, config, context=None, key=None):
    model, raw = parameters
    distances = _selected_distances(
        model, payload, context=context, inference=False, key=key
    ).astype(jnp.float64)
    nll, _ = profiled_mlpe_ml_fit(
        distances,
        payload.targets,
        *payload.pairs,
        n_populations=len(payload.nodes),
        raw_variances=raw[payload.group_index],
        variance_floor=config.mlpe_variance_floor,
        jitter=config.mlpe_jitter,
    )
    return payload.weight * nll / len(distances)


def _mlpe_gradient(parameters, payload, *, config, context=None, key=None):
    value, gradient = eqx.filter_value_and_grad(_mlpe_loss)(
        parameters, payload, config=config, context=context, key=key
    )
    bad = ~jnp.isfinite(value)
    for leaf in jax.tree.leaves(gradient):
        if eqx.is_inexact_array(leaf):
            bad |= jnp.any(~jnp.isfinite(leaf))
    return eqx.error_if(
        (value, gradient),
        bad,
        "Nonfinite objective or gradient; check nonconstant distances, "
        "identifiable endpoints and variance conditioning",
    )


def _refresh_mlpe(model, raw, payload, *, config, context=None):
    distances = _selected_distances(model, payload, context=context).astype(jnp.float64)
    return finalize_mlpe_fit(
        distances,
        payload.targets,
        *payload.pairs,
        n_populations=len(payload.nodes),
        raw_variances=raw,
        variance_floor=config.mlpe_variance_floor,
        jitter=config.mlpe_jitter,
    )


def _frozen_validation_loss(model, raw, moments, payload, *, config, context=None):
    distances = _selected_distances(model, payload, context=context).astype(jnp.float64)
    center, scale, beta = moments
    return mlpe_ml_negative_log_likelihood(
        distances,
        payload.targets,
        *payload.pairs,
        n_populations=len(payload.nodes),
        fixed_effects=beta,
        raw_variances=raw,
        distance_center=center,
        distance_scale=scale,
        variance_floor=config.mlpe_variance_floor,
        jitter=config.mlpe_jitter,
    ) / len(distances)


_compiled_mlpe_gradient = eqx.filter_jit(_mlpe_gradient)
_compiled_refresh = eqx.filter_jit(_refresh_mlpe)
_compiled_validation = eqx.filter_jit(_frozen_validation_loss)


def _initial_raw_variances(batches, config):
    rows = []
    for batch in batches:
        if config.mlpe_initial_variances is None:
            variance = jnp.maximum(
                jnp.var(batch.payload.targets, ddof=1), config.mlpe_variance_floor * 100
            )
            pair = jnp.maximum(variance * jnp.asarray([0.25, 0.5]), config.mlpe_variance_floor * 10)
        else:
            pair = jnp.asarray(config.mlpe_initial_variances, dtype=jnp.float64)
        positive = pair - config.mlpe_variance_floor
        rows.append(positive + jnp.log(-jnp.expm1(-positive)))
    return jnp.stack(rows)


def _aggregate(group_values):
    species_values = {}
    for groups in group_values.values():
        for species, value in groups.items():
            species_values.setdefault(species, []).append(value)
    by_species = {name: float(np.mean(values)) for name, values in species_values.items()}
    return float(np.mean(list(by_species.values()))), by_species, group_values


def _checked_call[T](
    function: Callable[..., T],
    *args,
    batch: _GroupBatch | Sequence[_GroupBatch] | None,
    step: int,
    training_config: TrainingConfig,
    **kwargs,
) -> T:
    """Synchronize numerical work and attach identities without changing failures."""
    try:
        return jax.block_until_ready(function(*args, **kwargs))
    except RuntimeError as error:
        groups = () if batch is None else ((batch,) if isinstance(batch, _GroupBatch) else batch)
        identity = (
            "joint groups"
            if not groups
            else "; ".join(
                f"region {group.source.region.name!r}, species {group.source.species!r}"
                for group in groups
            )
        )
        solver = None if not groups or groups[0].context is None else groups[0].context.config
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


def _region_gradient(model, raw, groups, *, config, step, key):
    """Finish each tail backward before releasing its bounded intermediates."""
    batch = groups[0]
    if not execution.supports_pooled(model):
        parameters = model if raw is None else (model, raw)
        gradient = (
            (_compiled_gradient if config.jit else _gradient)
            if raw is None
            else (_compiled_mlpe_gradient if config.jit else _mlpe_gradient)
        )
        options = {} if raw is None else {"config": execution.settings(config)}
        accumulated = None
        for group in groups:
            _, contribution = _checked_call(
                gradient,
                parameters,
                group.payload,
                context=group.context,
                key=key,
                **options,
                batch=group,
                step=step,
                training_config=config,
            )
            accumulated = execution.add_gradients(accumulated, contribution)
        return accumulated
    tail, pooled, pullback = _checked_call(
        execution.pooled_region,
        model,
        batch.payload.features,
        jit=config.jit,
        inference=False,
        key=key,
        batch=batch,
        step=step,
        training_config=config,
    )
    gradient = execution.compiled_chunk_gradient if config.jit else execution.chunk_gradient
    accumulated = None
    for chunk in execution.shape_chunks(groups, config.species_chunk_size):
        _, contribution = _checked_call(
            gradient,
            (tail, pooled, raw),
            execution.stack_payload(chunk),
            context=batch.context,
            config=execution.settings(config),
            batch=chunk,
            step=step,
            training_config=config,
        )
        accumulated = execution.add_gradients(accumulated, contribution)
    assert accumulated is not None
    tail_gradient, feature_gradient, raw_gradient = accumulated
    if pullback is not None:
        (backbone_gradient,) = _checked_call(
            pullback, feature_gradient, batch=batch, step=step, training_config=config
        )
        model_gradient = execution.join_backbone(tail_gradient, backbone_gradient)
    else:
        model_gradient = tail_gradient
    return model_gradient if raw is None else (model_gradient, raw_gradient)


def _group_results(model, raw, groups, *, tail, pooled, heads, group_indices, mode, config, step):
    """Yield synchronized bounded outputs; only final posterior output persists."""
    options = execution.settings(config)
    if pooled is not None:
        evaluate = execution.compiled_chunk_evaluate if config.jit else execution.chunk_evaluate
        for chunk in execution.shape_chunks(groups, config.species_chunk_size):
            payload = execution.stack_payload(chunk)
            moments = jnp.zeros((len(chunk), 4), dtype=payload.targets.dtype)
            if mode == "validation":
                indices, rows = [], []
                for batch in chunk:
                    source = batch.source
                    head = heads[source.region.name][source.species]
                    indices.append(group_indices[(source.region.name, source.species)])
                    rows.append(
                        (head.distance_center, head.distance_scale, head.intercept, head.slope)
                    )
                payload = payload._replace(group_index=jnp.asarray(indices, dtype=jnp.int32))
                moments = jnp.asarray(rows)
            result = _checked_call(
                evaluate,
                tail,
                pooled,
                raw,
                payload,
                moments,
                config=options,
                context=chunk[0].context,
                mode=mode,
                batch=chunk,
                step=step,
                training_config=config,
            )
            for index, batch in enumerate(chunk):
                yield batch, jax.tree.map(lambda leaf, index=index: leaf[index], result)
        return
    for batch in groups:
        source = batch.source
        if mode == "direct":
            evaluate = _compiled_loss if config.jit else _group_loss
            args, kwargs = (model, batch.payload), {}
        elif mode == "refresh":
            evaluate = _compiled_refresh if config.jit else _refresh_mlpe
            args, kwargs = (
                (model, raw[batch.payload.group_index], batch.payload),
                {"config": options},
            )
        else:
            evaluate = _compiled_validation if config.jit else _frozen_validation_loss
            head = heads[source.region.name][source.species]
            moments = (
                jnp.asarray(head.distance_center),
                jnp.asarray(head.distance_scale),
                jnp.asarray((head.intercept, head.slope)),
            )
            args = (
                model,
                raw[group_indices[(source.region.name, source.species)]],
                moments,
                batch.payload,
            )
            kwargs = {"config": options}
        yield (
            batch,
            _checked_call(
                evaluate,
                *args,
                **kwargs,
                context=batch.context,
                batch=batch,
                step=step,
                training_config=config,
            ),
        )


def _evaluate_snapshot(model, raw, batches, validation, group_indices, *, config, step):
    # Process the union by region so train/validation share the same inference
    # encoding without retaining pooled features from every region.
    regions = {}
    for batches_for_role, role in ((batches, 0), (validation, 1)):
        for batch in batches_for_role:
            regions.setdefault(batch.source.region.name, ([], []))[role].append(batch)
    heads, training_values, validation_values = {}, {}, {}
    for training, validating in regions.values():
        first = (training or validating)[0]
        tail = pooled = None
        if execution.supports_pooled(model):
            tail, pooled, _ = _checked_call(
                execution.pooled_region,
                model,
                first.payload.features,
                jit=config.jit,
                batch=first,
                step=step,
                training_config=config,
            )
        for groups, mode, values in (
            (training, "direct" if raw is None else "refresh", training_values),
            (validating, "direct" if raw is None else "validation", validation_values),
        ):
            for batch, result in _group_results(
                model,
                raw,
                groups,
                tail=tail,
                pooled=pooled,
                heads=heads,
                group_indices=group_indices,
                mode=mode,
                config=config,
                step=step,
            ):
                source = batch.source
                if not all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(result)):
                    calculation = {
                        "refresh": "MLPE calibration",
                        "validation": "MLPE validation",
                        "direct": "objective",
                    }[mode]
                    raise FloatingPointError(
                        f"region {source.region.name!r}, species {source.species!r}: "
                        f"nonfinite {calculation} at step {step}; check nonconstant distances, "
                        "identifiable endpoints and variance conditioning"
                    )
                if mode == "refresh":
                    assert isinstance(result, MLPEFit)
                    head = MLPECalibration.from_fit(
                        result,
                        region_name=source.region.name,
                        species_name=source.species,
                        target=source.observations.target,
                        population_ids=source.region.sampling_units[source.species].ids,
                        calibration_pairs=_labelled_pairs(batch),
                        calibration_roles=("training",) * len(batch.payload.targets),
                        config=MLPEConfig(
                            variance_floor=config.mlpe_variance_floor, jitter=config.mlpe_jitter
                        ),
                        optimizer_iterations=step,
                        optimizer_message="Fixed-budget joint Adam; no variance convergence claim",
                        converged=False,
                    )
                    heads.setdefault(source.region.name, {})[source.species] = head
                    value = float(result.nll) / len(batch.payload.targets)
                else:
                    assert not isinstance(result, MLPEFit)
                    value = float(result)
                values.setdefault(source.region.name, {})[source.species] = value
    return (
        heads,
        _aggregate(training_values),
        ((None, {}, {}) if not validation else _aggregate(validation_values)),
    )


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
    if config.objective == "mlpe" and not jax.config.read("jax_enable_x64"):
        raise RuntimeError("MLPE training requires float64; set JAX_ENABLE_X64=true")
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
    batches = _prepare_batches(
        training, distance_model, role="training", objective=config.objective
    )
    validation_batches = _prepare_batches(
        validating, distance_model, role="validation", objective=config.objective
    )
    group_indices = {
        (b.source.region.name, b.source.species): index for index, b in enumerate(batches)
    }
    if config.objective == "mlpe" and any(
        (b.source.region.name, b.source.species) not in group_indices for b in validation_batches
    ):
        raise ValueError(
            "MLPE validation requires a training calibration for the same region-species group"
        )
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
    raw = (
        (None if config.objective == "direct_log1p" else _initial_raw_variances(batches, config))
        if state is None
        else state.raw_variances
    )
    parameters = distance_model if raw is None else (distance_model, raw)
    optimizer_state = (
        optax.adam(config.learning_rate).init(eqx.filter(parameters, eqx.is_inexact_array))
        if state is None
        else state.optimizer_state
    )
    random_key = jax.random.PRNGKey(config.seed) if state is None else state.rng_key
    model = distance_model
    history = [] if state is None else list(state.history)
    selected = None if state is None else state.best_model
    selected_raw = None if state is None else state.selected_raw_variances
    selected_step = 0 if state is None else state.selected_step
    best_loss = float("inf") if state is None else state.best_loss
    progress = 0 if state is None else state.step
    update = _compiled_update if config.jit else _apply_gradient
    region_batches = _by_region(batches)

    def snapshot():
        assert selected is not None
        return TrainingState(
            progress,
            selected_step,
            model,
            raw,
            selected_raw,
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
            for groups in _sample_regions(region_batches, update_key, config):
                key = jax.random.fold_in(update_key, groups[0].region_index)
                contribution = _region_gradient(
                    model, raw, groups, config=config, step=step, key=key
                )
                accumulated = execution.add_gradients(accumulated, contribution)
            parameters, optimizer_state = _checked_call(
                update,
                parameters,
                optimizer_state,
                accumulated,
                learning_rate=config.learning_rate,
                batch=None,
                step=step,
                training_config=config,
            )
        if isinstance(parameters, tuple):
            model, raw = parameters
        else:
            model, raw = parameters, None
        progress = step
        if step == 0 or step % config.evaluation_interval == 0 or step == config.steps:
            heads, (loss, by_species, by_group), (val_loss, val_species, val_group) = (
                _evaluate_snapshot(
                    model, raw, batches, validation_batches, group_indices, config=config, step=step
                )
            )
            calibration = (
                DirectCalibrationModule(training[0].observations.target)
                if raw is None
                else UnpooledMLPECalibrationModule(training[0].observations.target, heads)
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
                selected_raw = raw
                best_loss = loss if val_loss is None else val_loss
                selected = LandscapeGeneticsModel(
                    model,
                    calibration,
                    training[0].region.feature_array.shape[-1],
                    training[0].region.feature_names,
                    training_objective=config.objective,
                    training_pairs=training_pairs,
                    validation_pairs=validation_pairs,
                )
        if step and on_step is not None:
            on_step(snapshot())
    return snapshot()
