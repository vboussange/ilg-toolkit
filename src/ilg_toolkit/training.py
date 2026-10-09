"""In-memory fitting with sequential, equally weighted regional contributions."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .config import FitConfig
from .data import ObservationPartition, PairwiseObservations, PreparedRegion
from .models import ConductanceModel, EmbeddingDistanceModel, UNetEmbeddingDistance
from .predictor import Predictor
from .solver import SolverContext, build_solver_context

RegionCollection = PreparedRegion | Sequence[PreparedRegion] | Mapping[str, PreparedRegion]
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
    """Selected shared predictor and regional optimization history."""

    predictor: Predictor
    history: tuple[EpochRecord, ...]
    selected_epoch: int
    selection: str
    region_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class _RegionalBatch:
    """Keep each region's observations and solver state separate from its shared encoder."""

    region: PreparedRegion
    observations: PairwiseObservations
    partition: ObservationPartition | None
    data: tuple
    context: SolverContext | None = None


def _prepared(region, observations, partition=None, *, role="training"):
    if partition is not None and partition.role != role:
        raise ValueError(f"Expected a {role} partition, got {partition.role}")
    pairs, values = observations.aligned_pairs(region, partition)
    if observations.target.kind != "dissimilarity" or np.any(values < 0):
        raise ValueError(
            "direct_log1p requires nonnegative dissimilarities; relatedness is unsupported"
        )
    return (
        jnp.asarray(region.features),
        jnp.asarray(region.pixel_nodes),
        pairs,
        jnp.asarray(observations.target.forward(values)),
    )


def _normalize_inputs(regions, observations, partitions):
    """Align explicit collections, then sort stable region identities for RNG order."""
    if isinstance(regions, PreparedRegion):
        ordered = [regions]
    elif isinstance(regions, Mapping):
        ordered = list(regions.values())
        if any(
            not isinstance(value, PreparedRegion) or key != value.name
            for key, value in regions.items()
        ):
            raise ValueError("Region mapping keys must match each prepared region's name")
    elif isinstance(regions, Sequence) and not isinstance(regions, (str, bytes)):
        ordered = list(regions)
    else:
        raise ValueError("Provide a prepared region or a nonempty region sequence/mapping")
    if not ordered or any(not isinstance(region, PreparedRegion) for region in ordered):
        raise ValueError("Regions must be a nonempty collection of PreparedRegion inputs")
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


def _regional_loss(encoder, data, *, context=None, inference=True, key=None):
    features, nodes, pairs, target = data
    options = dict(inference=inference, key=key)
    if isinstance(encoder, ConductanceModel):
        options["context"] = context
    predictions = encoder.predict_distances(features, nodes, **options)[pairs]
    return jnp.mean(jnp.square(jnp.log1p(predictions) - jnp.log1p(target)))


def _regional_value_and_grad(encoder, data, *, context=None, key=None):
    value, gradient = eqx.filter_value_and_grad(_regional_loss)(
        encoder, data, context=context, inference=False, key=key
    )
    bad_gradient = jnp.array(False)
    for leaf in jax.tree.leaves(gradient):
        if eqx.is_inexact_array(leaf):
            bad_gradient = bad_gradient | jnp.any(~jnp.isfinite(leaf))
    gradient = eqx.error_if(gradient, bad_gradient, "Nonfinite encoder gradient")
    return value, gradient


_compiled_loss = eqx.filter_jit(_regional_loss)
_compiled_value_and_grad = eqx.filter_jit(_regional_value_and_grad)


def fit(
    region: RegionCollection,
    observations: ObservationCollection,
    *,
    model: ConductanceModel | EmbeddingDistanceModel | None = None,
    config: FitConfig | None = None,
    validation: tuple[RegionCollection, ObservationCollection] | None = None,
    partition: PartitionCollection = None,
    validation_partition: PartitionCollection = None,
) -> FitResult:
    """Fit one shared encoder with direct log1p-MSE on one or more regions.

    Collections can be aligned sequences or mappings keyed by declared region
    names. Different regions require identical explicit feature_names.
    Each update averages regional gradients after per-pair normalization, and
    evaluates one region at a time outside an enclosing differentiation graph.
    Region names determine deterministic key order; validation never consumes
    training random keys or changes the fixed budget or learning rate.

    Without validation the final encoder is returned; otherwise validation's
    equal-region objective selects among initialization and updated encoders.
    Genetic prediction returns original declared units without an MLPE head.
    """
    config = config or FitConfig()
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
        if prepared_region.features.shape[-1] != first_region.features.shape[-1]:
            raise ValueError("Regional training and validation feature channels must match")
        if prepared_region.feature_names != first_region.feature_names:
            raise ValueError("Regional feature contracts must match meanings and order")

    init_key, random_key = jax.random.split(jax.random.key(config.seed))
    model = (
        model
        if model is not None
        else UNetEmbeddingDistance(
            first_region.features.shape[-1],
            patch_size=1,
            base_channels=8,
            embedding_dim=4,
            dropout=0,
            key=init_key,
        )
    )
    contexts = {}

    def make_batch(inputs, role):
        prepared_region, regional_observations, selected_partition = inputs
        data = _prepared(prepared_region, regional_observations, selected_partition, role=role)
        context = None
        if isinstance(model, ConductanceModel):
            height, width = prepared_region.features.shape[:2]
            if height % model.patch_size or width % model.patch_size:
                raise ValueError("Raster dimensions must be divisible by model patch_size")
            shape = (height // model.patch_size, width // model.patch_size)
            if shape not in contexts:
                contexts[shape] = build_solver_context(shape, config.solver)
            context = contexts[shape]
        return _RegionalBatch(
            prepared_region, regional_observations, selected_partition, data, context
        )

    training_batches = tuple(make_batch(inputs, "training") for inputs in training_inputs)
    validation_batches = tuple(make_batch(inputs, "validation") for inputs in validation_inputs)
    training_by_name = {batch.region.name: batch for batch in training_batches}

    def labelled_pairs(batch):
        return {
            frozenset((batch.region.sampling_unit_ids[i], batch.region.sampling_unit_ids[j]))
            for i, j in zip(*batch.data[2], strict=True)
        }

    for batch in validation_batches:
        training_batch = training_by_name.get(batch.region.name)
        if training_batch is not None:
            if labelled_pairs(training_batch) & labelled_pairs(batch):
                raise ValueError(
                    "Training and validation observations overlap within the same region"
                )
            if not np.array_equal(training_batch.region.features, batch.region.features):
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

    optimizer = optax.adam(config.learning_rate)
    optimizer_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    def apply_gradient(encoder, state, gradient):
        updates, state = optimizer.update(gradient, state, encoder)
        encoder = eqx.apply_updates(encoder, updates)
        bad_update = jnp.array(False)
        for leaf in jax.tree.leaves((encoder, state)):
            if eqx.is_inexact_array(leaf):
                bad_update = bad_update | jnp.any(~jnp.isfinite(leaf))
        return eqx.error_if((encoder, state), bad_update, "Nonfinite optimizer update")

    evaluate = _compiled_loss if config.jit else _regional_loss
    gradient_function = _compiled_value_and_grad if config.jit else _regional_value_and_grad
    apply_update = eqx.filter_jit(apply_gradient) if config.jit else apply_gradient

    def checked_call(function, *args, batch, epoch, **kwargs):
        try:
            return jax.block_until_ready(function(*args, **kwargs))
        except RuntimeError as error:
            operation = "resistance" if isinstance(model, ConductanceModel) else "encoder"
            raise RuntimeError(
                f"{batch.region.name}: {operation} calculation failed at epoch {epoch}; "
                f"solver={config.solver}"
            ) from error

    def evaluate_regions(batches, epoch):
        regional_values = {}
        for batch in batches:
            value = float(
                checked_call(
                    evaluate, model, batch.data, context=batch.context, batch=batch, epoch=epoch
                )
            )
            if not np.isfinite(value):
                raise FloatingPointError(
                    f"{batch.region.name}: nonfinite objective at epoch {epoch}"
                )
            regional_values[batch.region.name] = value
        return float(np.mean(list(regional_values.values()))), regional_values

    history = []
    selected = model
    selected_epoch = 0
    best_loss = float("inf")
    for epoch in range(config.epochs + 1):
        if epoch:
            random_key, update_key = jax.random.split(random_key)
            keys = jax.random.split(update_key, len(training_batches))
            accumulated = None
            # Keep this host loop outside JIT/grad. Each regional backward pass
            # completes before the next starts; only parameter gradients remain.
            for batch, key in zip(training_batches, keys, strict=True):
                objective, gradient = checked_call(
                    gradient_function,
                    model,
                    batch.data,
                    context=batch.context,
                    key=key,
                    batch=batch,
                    epoch=epoch,
                )
                if not np.isfinite(float(objective)):
                    raise FloatingPointError(
                        f"{batch.region.name}: nonfinite training objective at epoch {epoch}"
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
            model, optimizer_state = checked_call(
                apply_update,
                model,
                optimizer_state,
                averaged,
                batch=training_batches[0],
                epoch=epoch,
            )
        training_loss, regional_training = evaluate_regions(training_batches, epoch)
        validation_loss, regional_validation = (
            (None, {}) if not validation_batches else evaluate_regions(validation_batches, epoch)
        )
        history.append(
            EpochRecord(
                epoch, training_loss, validation_loss, regional_training, regional_validation
            )
        )
        if validation_loss is None or validation_loss < best_loss:
            selected, selected_epoch = model, epoch
            best_loss = training_loss if validation_loss is None else validation_loss
    return FitResult(
        Predictor(
            selected,
            first_observations.target,
            first_region.features.shape[-1],
            feature_names=first_region.feature_names,
            solver_config=config.solver,
        ),
        tuple(history),
        selected_epoch,
        "final" if validation is None else "validation",
        tuple(batch.region.name for batch in training_batches),
    )
