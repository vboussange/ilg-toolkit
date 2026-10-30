"""Training of distance models and statistical parameters across regions and species."""

import contextlib
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, cast, final, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxscape.solvers import SparseDirectState

from ._checks import _require_float64
from ._training_utils import (
    _prepare_training_inputs,
    _RegionSpeciesKey,
    _RegionSpeciesTrainingInput,
)
from .config import TrainingConfig
from .data import PairwiseObservations, RegionBatch
from .losses import _criterion, _LossContext, AbstractLoss
from .model import LandscapeGeneticsModel
from .models import AbstractDistanceModel
from .models.base import (
    _check_regions,
    _distances_from_latent,
    _model_species,
    _prepare_distance_context,
)
from .models.common import _backbone, _inference_model, _reject_stateful_layers, _training_model
from .resistance import _carried_state, _ResistanceSolverContext, _with_carried_state


_Parameters = tuple[AbstractDistanceModel, Any]
_RasterShape = tuple[int, int]
_Contexts = dict[_RasterShape, _ResistanceSolverContext | None]
_SolverStates = dict[_RasterShape, SparseDirectState | None]
# What `eqx.error_if` raises from a compiled call and from a call under `jax.disable_jit()`, and
# what a failed native call raises, such as one of the NVIDIA CUDA Direct Sparse Solver (cuDSS).
_COMPILED_CALL_ERRORS = (
    eqx.EquinoxRuntimeError,
    eqx.EquinoxTracetimeError,
    jax.errors.JaxRuntimeError,
)
# What JAX appends to the message of a compiled call failing on a GPU.
_EXECUTABLE_NAME = re.compile(r"\s*\[executable_name='[^']*'\]\s*$")


@final
class StepRecord(eqx.Module):
    """Evaluation record of one training step; step zero is initialization.

    Aggregate losses average the losses of region–species groups over regions within
    each species, then over species.

    Attributes:
        step: Number of updates before the evaluation.
        training_loss: Aggregate loss of the training groups, without the penalty.
        training_objective: Training objective at the evaluated parameters:
            ``training_loss`` plus the penalty divided by the number of training
            pairs, and ``training_loss`` without a penalty.
        validation_loss: Aggregate loss of the validation groups, or ``None``
            without validation.
        training_by_region: Mean training loss of each region over its species.
        validation_by_region: Mean validation loss of each region over its species.
        training_by_species: Mean training loss of each species over its regions,
            or ``None`` for a single-species distance model.
        validation_by_species: Mean validation loss of each species over its
            regions, or ``None`` for a single-species distance model.
    """

    step: int
    training_loss: float
    training_objective: float
    validation_loss: float | None = None
    training_by_region: dict[str, float] = eqx.field(default_factory=dict)
    validation_by_region: dict[str, float] = eqx.field(default_factory=dict)
    training_by_species: dict[str, float] | None = None
    validation_by_species: dict[str, float] | None = None


@final
class TrainingState[DistanceModel: AbstractDistanceModel](eqx.Module):
    """Training state: progress, optimizer and random state, and the latest and selected models.

    [`train`][ilg_toolkit.train] returns a training state and continues from one.
    Continuation takes unchanged inputs and settings, apart from a larger step
    budget and an optimizer whose state keeps the saved structure. The class is
    generic in its distance model type, like
    [`LandscapeGeneticsModel`][ilg_toolkit.LandscapeGeneticsModel].

    Attributes:
        loss_state: Statistical parameter state after the latest update.
        optimizer_state: Optimizer state after the latest update.
        rng_key: Random key for the next update.
        config: Training configuration.
        history: Evaluation records, in step order.
        latest_model: Landscape genetics model of the latest step.
        selected_model: Selected model: the evaluated model with the lowest
            validation loss, or the final evaluated model without validation.
        selected_step: Step of the selected model.
        step: Number of completed updates.
        converged: Whether an update reached the configuration's gradient tolerance.
        training_groups: Region name and species index of each training
            region–species group.
        validation_groups: Region name and species index of each validation
            region–species group.
    """

    loss_state: Any
    optimizer_state: optax.OptState
    rng_key: jax.Array
    config: TrainingConfig
    history: tuple[StepRecord, ...]
    latest_model: LandscapeGeneticsModel[DistanceModel]
    selected_model: LandscapeGeneticsModel[DistanceModel]
    selected_step: int
    step: int
    converged: bool
    training_groups: tuple[tuple[str, int], ...]
    validation_groups: tuple[tuple[str, int], ...]

    @property
    def selection(self) -> str:
        """Selection rule: ``"validation"`` with validation groups, otherwise ``"final"``."""
        return "validation" if self.validation_groups else "final"

    @property
    def selected_loss(self) -> float:
        """Validation loss, or training loss without validation, of the selected model."""
        record = next(record for record in self.history if record.step == self.selected_step)
        return record.training_loss if record.validation_loss is None else record.validation_loss


def _validate_state(state: TrainingState) -> None:
    """Raises a ``ValueError`` if the state's steps, history or convergence are inconsistent."""
    if not 0 <= state.step <= state.config.steps:
        raise ValueError("Training step must be within the configured budget")
    if not 0 <= state.selected_step <= state.step:
        raise ValueError("Selected step must be within the completed training steps")
    if not 1 <= len(state.history) <= state.step + 1:
        raise ValueError("History length must be within the completed training steps plus one")
    if not isinstance(state.converged, bool):
        raise ValueError("Training convergence must be recorded as a boolean")


def _node_types(structure: Any) -> str:
    """Returns the node type names of a tree structure, with ``*`` for each leaf."""
    node = structure.node_data()
    if node is None:
        return "*"
    return f"{node[0].__name__}({','.join(map(_node_types, structure.children()))})"


def _optimizer_state_signature(optimizer_state: optax.OptState) -> dict[str, Any]:
    """Returns the node types of an optimizer state and the path, shape and dtype of its leaves."""
    return {
        "nodes": _node_types(jax.tree.structure(optimizer_state)),
        "leaves": [
            [jax.tree_util.keystr(path), list(leaf.shape), str(leaf.dtype)]
            if hasattr(leaf, "dtype")
            else [jax.tree_util.keystr(path), type(leaf).__name__]
            for path, leaf in jax.tree_util.tree_leaves_with_path(optimizer_state)
        ],
    }


def _same_optimizer_state_structure(first: optax.OptState, second: optax.OptState) -> bool:
    """Returns whether two optimizer states share their tree structure, leaf shapes and dtypes."""
    if jax.tree.structure(first) != jax.tree.structure(second):
        return False
    return _optimizer_state_signature(first) == _optimizer_state_signature(second)


def _match_update_weak_types(
    optimizer: optax.GradientTransformation,
    optimizer_state: optax.OptState,
    trainable: _Parameters,
) -> optax.OptState:
    """Returns the optimizer state with the weak types that one update gives its leaves.

    Optimizers such as ``optax.lbfgs()`` initialize weakly typed scalars that their
    update returns strongly typed; matching them lets every update share one
    compiled program.
    """

    def update(state: optax.OptState) -> optax.OptState:
        value = jnp.zeros((), jnp.float64)
        return optax.with_extra_args_support(optimizer).update(
            cast(optax.Updates, trainable),
            state,
            cast(optax.Params, trainable),
            value=value,
            grad=trainable,
            value_fn=lambda _: value,
        )[1]

    updated = eqx.filter_eval_shape(update, optimizer_state)

    def match(leaf: Any, shape: Any) -> Any:
        if (
            isinstance(leaf, jax.Array)
            and isinstance(shape, jax.ShapeDtypeStruct)
            and leaf.weak_type
            and not shape.weak_type
            and leaf.dtype == shape.dtype
        ):
            return jax.lax.convert_element_type(leaf, leaf.dtype)
        return leaf

    return jax.tree.map(match, optimizer_state, updated)


class _TrainingPayload(NamedTuple):
    """Landscape features, sampling-unit pixels and loss context of one region–species group."""

    features: jax.Array
    nodes: jax.Array
    context: _LossContext


@final
@dataclass(frozen=True)
class _RegionSpeciesTrainingBatch:
    """One region–species group's inputs, loss weight and raster shape."""

    region: RegionBatch
    observations: PairwiseObservations
    payload: _TrainingPayload
    species_index: int
    key: _RegionSpeciesKey
    loss_weight: float
    raster_shape: _RasterShape


def _prepared(region, observations, loss, *, training):
    """Returns the training payload of one region–species group."""
    return _TrainingPayload(
        region.feature_array,
        region.pixel_nodes,
        loss._prepare(region, observations, training=training),
    )


@eqx.filter_jit
def _group_distances(model, latent, payload, species_index, context):
    """Returns a region–species group's ecological distances and the next solve's context."""
    return _distances_from_latent(
        model,
        latent,
        payload.nodes,
        payload.context.pairs,
        raster_shape=(payload.features.shape[0], payload.features.shape[1]),
        species_index=species_index,
        context=context,
    )


def _group_name(species: tuple[str, ...] | None, group: _RegionSpeciesKey) -> str:
    """Returns the name of a region–species group in error messages."""
    region, index = group
    return f"region {region!r}, species {index if species is None else repr(species[index])}"


def _failure_reason(error: Exception) -> str:
    """Returns the line of a compiled call's error that states why the call failed.

    That line follows Equinox's marker for ``eqx.error_if``, ends the traceback of a
    failed callback, and opens the message of a failed native call. It excludes the
    name of the compiled program.
    """
    message = _EXECUTABLE_NAME.sub("", str(error))
    lines = message.rpartition("equinox.EquinoxRuntimeError: ")[2].strip().split("\n")
    return lines[-1] if lines[0].endswith("Traceback (most recent call last):") else lines[0]


def _training_failure(
    species: tuple[str, ...] | None,
    groups: Sequence[_RegionSpeciesKey],
    step: int,
    error: Exception,
    *,
    in_line_search: bool = False,
) -> RuntimeError:
    """Returns the error naming the region–species groups and step of a failed compiled call.

    The message ends with the reason of ``error``, which raises from the returned
    error.
    """
    names = "; ".join(_group_name(species, group) for group in groups)
    phase = "line search: " if in_line_search else ""
    return RuntimeError(f"{names}: step {step}: {phase}{_failure_reason(error)}")


@contextlib.contextmanager
def _naming_failures(
    species: tuple[str, ...] | None,
    groups: Sequence[_RegionSpeciesKey],
    step: int,
    *,
    in_line_search: bool = False,
) -> Iterator[None]:
    """Raises the error naming the region–species groups and step of a call failing in the block."""
    try:
        yield
    except _COMPILED_CALL_ERRORS as error:
        raise _training_failure(
            species, groups, step, error, in_line_search=in_line_search
        ) from error


class _Contribution(NamedTuple):
    """One region–species group's arrays entering a compiled gradient.

    The group's region name stays outside these arrays; groups whose arrays share
    shapes share compiled programs.
    """

    payload: _TrainingPayload
    species_index: jax.Array
    loss_weight: jax.Array


def _microbatch_loss(parameters, contributions, loss, context, key):
    """Returns the weighted loss of a species microbatch within one region and the next context."""
    model, latent, states = parameters
    configured = _training_model(model, key)
    values = []
    for item, state in zip(contributions, states, strict=True):
        distances, context = _group_distances(
            configured, latent, item.payload, item.species_index, context
        )
        values.append(item.loss_weight * loss._value(distances, item.payload.context, state))
    return sum(values), context


_microbatch_gradient = eqx.filter_jit(eqx.filter_grad(_microbatch_loss, has_aux=True))


def _failing_groups(model, latent, contributions, groups, context, key):
    """Returns the region–species groups to name for a failed microbatch.

    That is the first region–species group whose distances fail again, or every group.
    """
    configured = _training_model(model, key)
    for item, group in zip(contributions, groups, strict=True):
        try:
            _, context = _group_distances(
                configured, latent, item.payload, item.species_index, context
            )
        except _COMPILED_CALL_ERRORS:
            return (group,)
    return groups


def _regional_gradient(parameters, contributions, groups, config, context, key, step):
    """Returns one region's weighted loss gradient, over species microbatches, and the next context.

    A failed microbatch raises the error naming the region–species group that failed.
    """
    model, loss_state = parameters
    features = contributions[0].payload.features
    latent, backward = eqx.filter_vjp(lambda m: _backbone(m, features, key), model)
    size = config.species_microbatch_size or len(contributions)
    accumulated = None
    for start in range(0, len(contributions), size):
        selected = groups[start : start + size]
        microbatch = contributions[start : start + size]
        selected_states, state_backward = eqx.filter_vjp(
            lambda current, selected=selected: tuple(
                config.loss._group_state(current, group) for group in selected
            ),
            loss_state,
        )
        try:
            (model_gradient, latent_gradient, selected_gradient), context = _microbatch_gradient(
                (model, latent, selected_states), microbatch, config.loss, context, key
            )
        except _COMPILED_CALL_ERRORS as error:
            failing = _failing_groups(model, latent, microbatch, selected, context, key)
            raise _training_failure(model.species, failing, step, error) from error
        gradient = (model_gradient, latent_gradient, state_backward(selected_gradient)[0])
        accumulated = (
            gradient
            if accumulated is None
            else jax.tree.map(lambda a, b: a + b, accumulated, gradient)
        )
    assert accumulated is not None
    model_gradient, latent_gradient, state_gradient = accumulated
    model_gradient = jax.tree.map(lambda a, b: a + b, model_gradient, backward(latent_gradient)[0])
    return (model_gradient, state_gradient), context


def _solver_states(contexts: _Contexts) -> _SolverStates:
    """Returns the carried solver state of each raster shape, as arrays and static fields only."""
    return {shape: _carried_state(context) for shape, context in contexts.items()}


@eqx.filter_jit
def _penalty_term(distance_model, penalty, pair_count) -> jax.Array:
    """Returns the penalty of a distance model divided by the number of training pairs."""
    return jnp.asarray(penalty(distance_model), jnp.float64) / pair_count


_penalty_gradient = eqx.filter_jit(eqx.filter_grad(_penalty_term))


def _when(
    condition: jax.Array,
    compute: Callable[[_SolverStates], tuple[jax.Array, _SolverStates]],
    solver_states: _SolverStates,
) -> tuple[jax.Array, _SolverStates]:
    """Returns ``compute(solver_states)`` under ``condition``, otherwise zero and the same states.

    It runs one branch.
    """
    result, _ = jax.eval_shape(compute, solver_states)
    return jax.lax.cond(
        condition,
        compute,
        lambda solver_states: (jnp.zeros(result.shape, result.dtype), solver_states),
        solver_states,
    )


@final
class _TrainingObjective(eqx.Module):
    """One update's training objective as a function of the parameters.

    It evaluates the same regions, species and random keys as the update's gradient:
    one deterministic training objective within an update. It holds every training
    group and skips the groups not sampled for the update, which carry zero weight; its
    structure and the compiled update stay fixed across updates. The penalty enters
    once, whichever groups the update samples.

    A call takes the carried solver state of each raster shape, ``None`` for a shape
    without one, and returns them refreshed. Each region–species group's solve takes
    the state the solve of the previous one returned.
    """

    regions: tuple[tuple[_Contribution, ...], ...]
    contexts: _Contexts
    keys: jax.Array
    penalty: Callable[[Any], Any] | None
    pair_count: jax.Array
    shapes: tuple[_RasterShape, ...] = eqx.field(static=True)
    groups: tuple[tuple[_RegionSpeciesKey, ...], ...] = eqx.field(static=True)
    loss: AbstractLoss = eqx.field(static=True)

    def __call__(
        self, parameters: _Parameters, solver_states: _SolverStates
    ) -> tuple[jax.Array, _SolverStates]:
        model, loss_state = parameters

        def group_value(index: int, item: _Contribution, group, latent, solver_states):
            shape = self.shapes[index]
            value, context = _microbatch_loss(
                (model, latent, (self.loss._group_state(loss_state, group),)),
                (item,),
                self.loss,
                _with_carried_state(self.contexts[shape], solver_states[shape]),
                self.keys[index],
            )
            return cast(jax.Array, value), {**solver_states, shape: _carried_state(context)}

        def region_value(index: int, solver_states):
            latent = _backbone(model, self.regions[index][0].payload.features, self.keys[index])
            values = []
            for item, group in zip(self.regions[index], self.groups[index], strict=True):
                value, solver_states = _when(
                    item.loss_weight > 0,
                    partial(group_value, index, item, group, latent),
                    solver_states,
                )
                values.append(value)
            return cast(jax.Array, sum(values)), solver_states

        values = []
        for index, contributions in enumerate(self.regions):
            value, solver_states = _when(
                jnp.any(jnp.stack([item.loss_weight for item in contributions]) > 0),
                partial(region_value, index),
                solver_states,
            )
            values.append(value)
        value = cast(jax.Array, sum(values))
        if self.penalty is None:
            return value, solver_states
        return value + _penalty_term(model, self.penalty, self.pair_count), solver_states


def _labeled_pairs(batch: _RegionSpeciesTrainingBatch) -> tuple[tuple[str, str], ...]:
    """Returns the observed pairs of a batch as sorted sampling-unit labels."""
    return tuple(
        (
            min(batch.region.sampling_unit_ids[i], batch.region.sampling_unit_ids[j]),
            max(batch.region.sampling_unit_ids[i], batch.region.sampling_unit_ids[j]),
        )
        for i, j in zip(*batch.payload.context.pairs, strict=True)
    )


def _prepare_batches(
    training_inputs: Sequence[_RegionSpeciesTrainingInput],
    validation_inputs: Sequence[_RegionSpeciesTrainingInput],
    distance_model: AbstractDistanceModel,
    config: TrainingConfig,
) -> tuple[
    tuple[_RegionSpeciesTrainingBatch, ...], tuple[_RegionSpeciesTrainingBatch, ...], _Contexts
]:
    """Returns training and validation batches, and the distance context of each raster shape."""
    contexts: _Contexts = {}

    def make_batch(
        inputs: _RegionSpeciesTrainingInput, *, training: bool
    ) -> _RegionSpeciesTrainingBatch:
        region, observations, species_index, key, loss_weight = inputs
        payload = _prepared(region, observations, config.loss, training=training)
        raster_shape = (region.feature_array.shape[0], region.feature_array.shape[1])
        if raster_shape not in contexts:
            contexts[raster_shape] = _prepare_distance_context(distance_model, region.feature_array)
        return _RegionSpeciesTrainingBatch(
            region, observations, payload, species_index, key, loss_weight, raster_shape
        )

    training = tuple(make_batch(inputs, training=True) for inputs in training_inputs)
    validation = tuple(make_batch(inputs, training=False) for inputs in validation_inputs)
    training_by_name = {batch.key: batch for batch in training}
    for batch in validation:
        training_batch = training_by_name.get(batch.key)
        if training_batch is None:
            continue
        if set(_labeled_pairs(training_batch)) & set(_labeled_pairs(batch)):
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
    return training, validation, contexts


@eqx.filter_jit
def _apply_update(
    parameters: _Parameters,
    optimizer: optax.GradientTransformation,
    optimizer_state: optax.OptState,
    gradient: _Parameters,
    training_objective: _TrainingObjective,
    solver_states: _SolverStates,
) -> tuple[_Parameters, optax.OptState, _SolverStates]:
    """Applies one optimizer update with the training objective's value, gradient and function.

    The gradient comes from the regional and microbatch programs, and the value from
    the training objective that line searches evaluate. The update returns the carried
    solver states after its last evaluation of the training objective.

    Optax calls the function inside a closed ``while_loop`` that takes a scalar back,
    so each evaluation reads the carried solver states from references and writes
    them back. Without a carried solver state, XLA removes the value when the
    optimizer ignores it. With one, every evaluation solves, because a solve is a
    side effect.
    """
    trainable, fixed = eqx.partition(parameters, eqx.is_inexact_array)
    references = jax.tree.map(jax.new_ref, solver_states)

    def value_fn(candidate):
        value, fresh = training_objective(
            eqx.combine(candidate, fixed), jax.tree.map(lambda ref: ref[...], references)
        )
        jax.tree.map(lambda ref, leaf: ref.__setitem__(..., leaf), references, fresh)
        return value

    value = value_fn(trainable)
    # Optax accepts registered Equinox pytrees; its array-container alias omits Modules.
    updates, optimizer_state = optax.with_extra_args_support(optimizer).update(
        cast(optax.Updates, gradient),
        optimizer_state,
        cast(optax.Params, trainable),
        value=value,
        grad=gradient,
        value_fn=value_fn,
    )
    parameters = eqx.apply_updates(parameters, updates)
    return parameters, optimizer_state, jax.tree.map(lambda ref: ref[...], references)


@eqx.filter_jit
def _gradient_norm(gradient: _Parameters) -> jax.Array:
    """Returns the global L2 norm of a gradient."""
    return optax.tree.norm(gradient)


class _Evaluation(NamedTuple):
    """The losses of an evaluation of batches, with their calibrations and the next contexts.

    Attributes:
        loss: Aggregate loss.
        by_region: Mean loss of each region over its species.
        calibrations: Calibrations of training batches; empty for an evaluation given
            calibrations.
        by_species: Mean loss of each species over its regions, or ``None`` for a
            single-species distance model.
        contexts: The contexts for the next solves.
    """

    loss: float
    by_region: dict[str, float]
    calibrations: dict
    by_species: dict[str, float] | None
    contexts: _Contexts


def _evaluate_regions(
    parameters: _Parameters,
    batches: Sequence[_RegionSpeciesTrainingBatch],
    config: TrainingConfig,
    step: int,
    contexts: _Contexts,
    calibrations: Mapping | None = None,
) -> _Evaluation:
    """Evaluates the loss of every batch, and calibrates training batches."""
    contexts = dict(contexts)
    values, fitted = {}, {}
    species = parameters[0].species
    configured = _inference_model(parameters[0])
    previous_region, latent = None, None
    for batch in batches:
        group_state = config.loss._group_state(parameters[1], batch.key)
        calibration = None if calibrations is None else calibrations.get(batch.key)
        if calibrations is not None and group_state is not None and calibration is None:
            raise ValueError(
                "Validation requires a training calibration for the same region–species group"
            )
        fixed = None if calibration is None else config.loss._fixed_parameters(calibration)
        with _naming_failures(species, (batch.key,), step):
            if batch.region.name != previous_region:
                latent = _backbone(parameters[0], batch.payload.features, None)
                previous_region = batch.region.name
            distances, contexts[batch.raster_shape] = _group_distances(
                configured,
                latent,
                batch.payload,
                jnp.asarray(batch.species_index),
                contexts[batch.raster_shape],
            )
            value = float(
                _criterion(config.loss, distances, batch.payload.context, group_state, fixed)
            )
        if not np.isfinite(value):
            raise FloatingPointError(
                f"{_group_name(species, batch.key)}: step {step}: nonfinite loss"
            )
        if calibrations is None:
            fitted[batch.key] = config.loss._finalize(
                distances,
                batch.payload.context,
                group_state,
                region=batch.region,
                observations=batch.observations,
                step=step,
            )
        values[batch.key] = value
    regional = {
        name: float(np.mean([values[b.key] for b in batches if b.region.name == name]))
        for name in dict.fromkeys(b.region.name for b in batches)
    }
    by_species = (
        None
        if species is None or len(species) == 1
        else {
            species[index]: float(
                np.mean([values[b.key] for b in batches if b.species_index == index])
            )
            for index in sorted({b.species_index for b in batches})
        }
    )
    return _Evaluation(
        sum(batch.loss_weight * values[batch.key] for batch in batches),
        regional,
        fitted,
        by_species,
        contexts,
    )


def train[DistanceModel: AbstractDistanceModel](
    region: RegionBatch | Sequence[RegionBatch] | Mapping[str, RegionBatch],
    observations: PairwiseObservations
    | Sequence[PairwiseObservations | Sequence[PairwiseObservations]]
    | Mapping[str, PairwiseObservations | Sequence[PairwiseObservations]],
    *,
    distance_model: DistanceModel | None = None,
    config: TrainingConfig | None = None,
    validation: tuple[
        RegionBatch | Sequence[RegionBatch] | Mapping[str, RegionBatch],
        PairwiseObservations
        | Sequence[PairwiseObservations | Sequence[PairwiseObservations]]
        | Mapping[str, PairwiseObservations | Sequence[PairwiseObservations]],
    ]
    | None = None,
    state: TrainingState[DistanceModel] | None = None,
    on_step: Callable[[TrainingState[DistanceModel]], None] | None = None,
) -> TrainingState[DistanceModel]:
    """Trains a distance model and its statistical parameters jointly across regions.

    Training averages the losses of region–species groups over regions within each
    species, then over species. It evaluates at step zero, every
    ``config.eval_every`` updates and at the final step, and selects the model with
    the lowest validation loss, or the final model without validation. ``state``
    continues training from its latest parameters and random state.

    ``config.optimizer`` applies each update and receives the update's training
    objective value, gradient and a function evaluating the same training objective
    (same regions, species and random key), so ``optax.lbfgs()`` line-searches
    deterministically within an update. With ``config.gradient_tolerance``, training
    stops after the first update whose gradient norm is at or below it, evaluates
    that step and sets ``converged=True``; continuing a converged state makes no
    updates.

    The training objective is the aggregate loss plus ``config.penalty`` divided by
    the number of training pairs (see [`TrainingConfig`][ilg_toolkit.TrainingConfig]).
    Training losses, validation losses and model selection exclude the penalty; each
    record's ``training_objective`` includes it.

    Training a conductance model prepares the resistance solver once for each raster
    shape. With ``solver="cudss"``, every update, line-search evaluation, evaluation
    and ``on_step`` evaluation on a raster shape carries that shape's solver state, so
    the resistance solver repeats no analysis. A distance model that composes a
    conductance model solves through ``ConductanceModel.ecological_distances``, which
    prepares its solver state itself and repeats the analysis on every solve. A failed
    compiled step raises at once, so the last checkpoint ``on_step`` wrote stays as it
    is.

    Training compiles its update and evaluation steps; ``with jax.disable_jit():``
    runs them eagerly for debugging with conjugate gradient and algebraic multigrid
    (AMG).

    Args:
        region: One region batch, a sequence of region batches, or a mapping keyed
            by region name.
        observations: Genetic observations aligned with the regions; each region
            may supply observations for several species.
        distance_model: Initial distance model, which carries the species names;
            ``None`` when continuing from ``state``.
        config: Training configuration; ``None`` when continuing with the saved
            configuration. When continuing, only the step budget may increase, and
            the optimizer may change if its state keeps the saved structure.
        validation: Regions and observations disjoint from the training
            observations, selecting the model by validation loss with the
            calibration estimated on the training observations.
        state: Training state to continue, with unchanged inputs and settings.
        on_step: Function called with the training state after initialization and
            after each update.

    Returns:
        The training state, with the history, the latest and selected models, and
        the optimizer and random state for continuation.

    Raises:
        TypeError: If the distance model has stateful layers.
        ValueError: If the inputs, settings or continuation state are invalid, or a
            raster is not a multiple of the distance model's patch size.
        ImportError: If the distance model's ``solver_config`` selects ``"cudss"``
            without spineax.
        FloatingPointError: If an evaluation produces a nonfinite loss or training
            objective. The message names the step, and the region and species of a
            nonfinite loss.
        RuntimeError: If float64 is disabled, the distance model's ``solver_config``
            selects ``"cudss"`` without a CUDA-enabled GPU backend, or a compiled step
            fails, such as an effective-resistance solve that reaches ``max_steps`` or a
            cuDSS factorization that fails. The message of a failed step names the
            region, species and step, with every region–species group sampled for the
            update when a line-search candidate fails.

    Examples:
        >>> loss = MLPELoss()
        >>> state = train(region, observations.select_pairs(training_pairs),
        ...               distance_model=distance_model,
        ...               config=TrainingConfig(loss=loss, steps=500))
        >>> model = state.selected_model
        >>> continued = train(region, observations.select_pairs(training_pairs), state=state,
        ...                   config=dataclasses.replace(state.config, steps=1000))

        A distance model with few parameters trains with L-BFGS to a gradient
        tolerance:

        >>> config = TrainingConfig(loss=Log1pLoss(), optimizer=optax.lbfgs(),
        ...                         steps=200, gradient_tolerance=1e-8)
        >>> state = train(region, observations, distance_model=distance_model, config=config)
        >>> state.converged
        True

        Zero steps initializes the statistical parameters and fits the coefficients
        at those parameters; ``fit_calibration_head`` optimizes a calibration head
        for a frozen distance model. Validation uses the coefficients and distance
        moments from training.
    """
    _require_float64()
    if state is not None and not isinstance(state, TrainingState):
        raise ValueError("state must be a TrainingState returned by train")
    if state is not None and distance_model is not None:
        raise ValueError(
            "A continuation state already contains its distance model; omit distance_model"
        )
    if state is None:
        if distance_model is None:
            raise ValueError("Fresh training requires a distance_model")
        if config is None:
            raise ValueError("Fresh training requires a TrainingConfig with loss")
    else:
        _validate_state(state)
        distance_model = state.latest_model.distance_model
        config = state.config if config is None else config
    assert distance_model is not None and config is not None
    _reject_stateful_layers(distance_model)
    if on_step is not None and not callable(on_step):
        raise ValueError("on_step must be callable or None")
    if state is not None:
        if (
            replace(config, steps=state.config.steps) != state.config
            or config.steps < state.config.steps
        ):
            raise ValueError(
                "Continuation requires identical configuration except an increased step budget"
            )
    if validation is not None and (not isinstance(validation, tuple) or len(validation) != 2):
        raise ValueError("validation must be a (regions, observations) tuple")
    prepared = _prepare_training_inputs(
        region, observations, distance_model.species, validation=validation
    )
    _check_regions(distance_model, prepared.regions)
    first_region, first_observations = prepared.inputs[0]
    prototype = LandscapeGeneticsModel(
        distance_model,
        config.loss._calibration_head(first_observations.target, {}, distance_model.species),
        first_region.feature_names,
    )
    random_key = (
        jax.random.split(jax.random.key(config.seed))[1] if state is None else state.rng_key
    )
    training_batches, validation_batches, contexts = _prepare_batches(
        prepared.training, prepared.validation, distance_model, config
    )
    pair_count = jnp.asarray(
        sum(len(batch.payload.context.targets) for batch in training_batches), jnp.float64
    )

    loss_state = (
        state.loss_state
        if state is not None
        else config.loss._initialize(
            {batch.key: batch.payload.context for batch in training_batches}
        )
    )
    parameters = (distance_model, loss_state)
    trainable = eqx.filter(parameters, eqx.is_inexact_array)
    if state is None:
        optimizer_state = config.optimizer.init(cast(optax.Params, trainable))
    else:
        expected = eqx.filter_eval_shape(config.optimizer.init, trainable)
        if not _same_optimizer_state_structure(expected, state.optimizer_state):
            raise ValueError(
                "Continuation requires an optimizer whose state has the saved structure"
            )
        optimizer_state = state.optimizer_state
    optimizer_state = _match_update_weak_types(config.optimizer, optimizer_state, trainable)

    regional_batches = tuple(
        tuple(b for b in training_batches if b.region.name == name)
        for name in dict.fromkeys(b.region.name for b in training_batches)
    )
    region_count = len(regional_batches)
    selected_count = min(region_count, config.region_batch_size or region_count)
    regional_shapes = tuple(groups[0].raster_shape for groups in regional_batches)
    regional_groups = tuple(tuple(batch.key for batch in groups) for groups in regional_batches)

    training_groups = tuple(batch.key for batch in training_batches)
    validation_groups = tuple(batch.key for batch in validation_batches)
    if state is not None and (
        training_groups != state.training_groups or validation_groups != state.validation_groups
    ):
        raise ValueError("Continuation requires unchanged training and validation groups")

    def make_model(calibrations) -> LandscapeGeneticsModel[DistanceModel]:
        head = config.loss._calibration_head(
            first_observations.target, calibrations, parameters[0].species
        )
        return eqx.tree_at(
            lambda m: (m.distance_model, m.calibration_head), prototype, (parameters[0], head)
        )

    history = list(state.history) if state is not None else []
    selected = state.selected_model if state is not None else None
    latest = state.latest_model if state is not None else None
    selected_step = state.selected_step if state is not None else 0
    selected_loss = state.selected_loss if state is not None else float("inf")

    def continuation() -> TrainingState:
        assert latest is not None and selected is not None
        return TrainingState(
            loss_state=parameters[1],
            optimizer_state=optimizer_state,
            rng_key=random_key,
            config=config,
            history=tuple(history),
            latest_model=latest,
            selected_model=selected,
            selected_step=selected_step,
            step=step,
            converged=converged,
            training_groups=training_groups,
            validation_groups=validation_groups,
        )

    step = state.step if state is not None else 0
    converged = state.converged if state is not None else False
    start_step = state.step + 1 if state is not None else 0
    for step in range(start_step, 0 if converged else config.steps + 1):
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
            sampled: list[_RegionSpeciesKey] = []
            weights = [[0.0] * len(groups) for groups in regional_batches]
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
                for i in chosen:
                    weights[index][i] = groups[i].loss_weight * correction
                contributions = tuple(
                    _Contribution(
                        groups[i].payload,
                        jnp.asarray(groups[i].species_index),
                        jnp.asarray(groups[i].loss_weight * correction),
                    )
                    for i in chosen
                )
                chosen_groups = tuple(groups[i].key for i in chosen)
                sampled.extend(chosen_groups)
                with _naming_failures(distance_model.species, chosen_groups, step):
                    gradient, contexts[regional_shapes[index]] = _regional_gradient(
                        parameters,
                        contributions,
                        chosen_groups,
                        config,
                        contexts[regional_shapes[index]],
                        keys[index],
                        step,
                    )
                accumulated = (
                    gradient
                    if accumulated is None
                    else jax.tree.map(
                        lambda total, contribution: total + contribution, accumulated, gradient
                    )
                )
            assert accumulated is not None
            if config.penalty is not None:
                model_gradient, state_gradient = accumulated
                penalty_gradient = _penalty_gradient(parameters[0], config.penalty, pair_count)
                accumulated = (
                    jax.tree.map(lambda a, b: a + b, model_gradient, penalty_gradient),
                    state_gradient,
                )
            training_objective = _TrainingObjective(
                tuple(
                    tuple(
                        _Contribution(
                            batch.payload,
                            jnp.asarray(batch.species_index),
                            jnp.asarray(weight),
                        )
                        for batch, weight in zip(groups, region_weights, strict=True)
                    )
                    for groups, region_weights in zip(regional_batches, weights, strict=True)
                ),
                contexts,
                keys,
                config.penalty,
                pair_count,
                regional_shapes,
                regional_groups,
                config.loss,
            )
            if config.gradient_tolerance is not None:
                converged = float(_gradient_norm(accumulated)) <= config.gradient_tolerance
            # The training objective just passed at the current parameters, so a candidate failed.
            with _naming_failures(distance_model.species, sampled, step, in_line_search=True):
                parameters, optimizer_state, solver_states = _apply_update(
                    parameters,
                    config.optimizer,
                    optimizer_state,
                    accumulated,
                    training_objective,
                    _solver_states(contexts),
                )
            contexts = {
                shape: _with_carried_state(context, solver_states[shape])
                for shape, context in contexts.items()
            }
        if step == 0 or step % config.eval_every == 0 or step == config.steps or converged:
            evaluation = _evaluate_regions(parameters, training_batches, config, step, contexts)
            contexts = evaluation.contexts
            training_objective = evaluation.loss
            if config.penalty is not None:
                training_objective += float(
                    _penalty_term(parameters[0], config.penalty, pair_count)
                )
            if not np.isfinite(training_objective):
                raise FloatingPointError(f"Nonfinite training objective at step {step}")
            validation_loss, regional_validation, species_validation = (
                None,
                {},
                None if len(_model_species(distance_model)) == 1 else {},
            )
            if validation_batches:
                validated = _evaluate_regions(
                    parameters, validation_batches, config, step, contexts, evaluation.calibrations
                )
                contexts = validated.contexts
                validation_loss, regional_validation, species_validation = (
                    validated.loss,
                    validated.by_region,
                    validated.by_species,
                )
            history.append(
                StepRecord(
                    step,
                    evaluation.loss,
                    training_objective,
                    validation_loss,
                    evaluation.by_region,
                    regional_validation,
                    evaluation.by_species,
                    species_validation,
                )
            )
            latest = make_model(evaluation.calibrations)
            if validation_loss is None or validation_loss < selected_loss:
                selected, selected_step = latest, step
                selected_loss = evaluation.loss if validation_loss is None else validation_loss
        elif on_step is not None:
            evaluation = _evaluate_regions(parameters, training_batches, config, step, contexts)
            contexts = evaluation.contexts
            latest = make_model(evaluation.calibrations)
        if on_step is not None:
            on_step(continuation())
        if converged:
            break
    return continuation()
