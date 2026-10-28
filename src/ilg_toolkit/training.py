"""Joint training with equal species influence and equal region influence within species.

Maximum-likelihood population effects (MLPE) calibration is fitted with the distance model.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast, final, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from ._training_utils import (
    _prepare_training_inputs,
    _RegionSpeciesTrainingInput,
    RegionSpeciesKey,
)
from .config import TrainingConfig
from .data import ObservationCollection, PairwiseObservations, RegionBatch, RegionCollection
from .losses import _LossContext
from .model import LandscapeGeneticsModel
from .models import AbstractDistanceModel
from .models.base import distances_from_latent, prepare_distance_context
from .models.common import with_dropout_mode
from .resistance import ResistanceSolverContext


Parameters = tuple[AbstractDistanceModel, Any]


@final
class StepRecord(eqx.Module):
    """Evaluated group criteria after an update; zero is initialization.

    The aggregate averages group criteria across regions within each species,
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
    """Training progress and matching fitted models for continuing optimization.

    ``latest_model`` and ``loss_state`` belong to the latest update, together with
    optimizer and random state. ``selected_model`` retains its own distance model
    and fitted calibration from the selected evaluation. Continuation requires
    unchanged inputs and settings, with only the total step budget allowed to increase.
    """

    loss_state: Any
    optimizer_state: optax.OptState
    rng_key: jax.Array
    config: TrainingConfig
    history: tuple[StepRecord, ...]
    latest_model: LandscapeGeneticsModel
    selected_model: LandscapeGeneticsModel
    selected_step: int
    step: int
    training_groups: tuple[RegionSpeciesKey, ...]
    validation_groups: tuple[RegionSpeciesKey, ...]

    @property
    def selection(self) -> str:
        """Selection rule: minimum validation loss or the final training evaluation."""
        return "validation" if self.validation_groups else "final"

    @property
    def selected_loss(self) -> float:
        """Validation or training loss at the selected model's evaluation."""
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
    """Regional features, sampling-unit vertices, and aligned training observations."""

    features: jax.Array
    nodes: jax.Array
    context: _LossContext


@dataclass(frozen=True)
class _RegionSpeciesTrainingBatch:
    """One region–species group's observations, numerical settings, and loss weight."""

    region: RegionBatch
    observations: PairwiseObservations
    payload: _TrainingPayload
    species_index: int
    key: RegionSpeciesKey
    loss_weight: float
    context: ResistanceSolverContext | None = None


def _prepared(region, observations, loss, *, training):
    return _TrainingPayload(
        region.feature_array,
        region.pixel_nodes,
        loss._prepare(region, observations, training=training),
    )


@eqx.filter_jit
def _group_distances(model, latent, payload, species_index, context):
    return distances_from_latent(
        model,
        latent,
        payload.nodes,
        payload.context.pairs,
        raster_shape=(payload.features.shape[0], payload.features.shape[1]),
        species_index=species_index,
        context=context,
    )


class _Contribution(NamedTuple):
    payload: _TrainingPayload
    key: RegionSpeciesKey
    species_index: jax.Array
    loss_weight: jax.Array


class _NumericContribution(NamedTuple):
    payload: _TrainingPayload
    species_index: jax.Array
    loss_weight: jax.Array


def _microbatch_objective(parameters, contributions, loss, context, key):
    model, latent, states = parameters
    configured = with_dropout_mode(model, inference=False, key=key)
    return sum(
        item.loss_weight
        * loss._value(
            _group_distances(configured, latent, item.payload, item.species_index, context),
            item.payload.context,
            state,
        )
        for item, state in zip(contributions, states, strict=True)
    )


_microbatch_gradient = eqx.filter_jit(eqx.filter_grad(_microbatch_objective))


@eqx.filter_jit
def _backbone(model, features, key):
    return with_dropout_mode(model, inference=key is None, key=key).backbone(features)


def _regional_gradient(parameters, contributions, config, context, key):
    model, loss_state = parameters
    features = contributions[0].payload.features
    latent, backward = eqx.filter_vjp(lambda m: _backbone(m, features, key), model)
    size = config.species_microbatch_size or len(contributions)
    accumulated = None
    for start in range(0, len(contributions), size):
        selected = contributions[start : start + size]
        selected_states, state_backward = eqx.filter_vjp(
            lambda current, selected=selected: tuple(
                config.loss._group_state(current, item.key) for item in selected
            ),
            loss_state,
        )
        numeric = tuple(
            _NumericContribution(item.payload, item.species_index, item.loss_weight)
            for item in selected
        )
        model_gradient, latent_gradient, selected_gradient = _microbatch_gradient(
            (model, latent, selected_states), numeric, config.loss, context, key
        )
        gradient = (model_gradient, latent_gradient, state_backward(selected_gradient)[0])
        accumulated = (
            gradient
            if accumulated is None
            else jax.tree.map(lambda a, b: a + b, accumulated, gradient)
        )
    assert accumulated is not None
    model_gradient, latent_gradient, state_gradient = accumulated
    model_gradient = jax.tree.map(lambda a, b: a + b, model_gradient, backward(latent_gradient)[0])
    return model_gradient, state_gradient


def _labelled_pairs(batch: _RegionSpeciesTrainingBatch) -> tuple[tuple[str, str], ...]:
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
) -> tuple[tuple[_RegionSpeciesTrainingBatch, ...], tuple[_RegionSpeciesTrainingBatch, ...]]:
    """Regional genetic targets with reusable resistance solvers."""
    contexts: dict[tuple[int, int], ResistanceSolverContext | None] = {}

    def make_batch(
        inputs: _RegionSpeciesTrainingInput, *, training: bool
    ) -> _RegionSpeciesTrainingBatch:
        region, observations, species_index, key, loss_weight = inputs
        payload = _prepared(region, observations, config.loss, training=training)
        shape = (region.feature_array.shape[0], region.feature_array.shape[1])
        if shape not in contexts:
            contexts[shape] = prepare_distance_context(distance_model, region.feature_array)
        context = contexts[shape]
        return _RegionSpeciesTrainingBatch(
            region, observations, payload, species_index, key, loss_weight, context
        )

    training = tuple(make_batch(inputs, training=True) for inputs in training_inputs)
    validation = tuple(make_batch(inputs, training=False) for inputs in validation_inputs)
    training_by_name = {batch.key: batch for batch in training}
    for batch in validation:
        training_batch = training_by_name.get(batch.key)
        if training_batch is None:
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


@eqx.filter_jit
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


@eqx.filter_jit
def _criterion(loss, distances, context, state, fixed):
    return loss._value(distances, context, state, fixed)


def _evaluate_regions(
    parameters: Parameters,
    batches: Sequence[_RegionSpeciesTrainingBatch],
    config: TrainingConfig,
    step: int,
    calibrations: Mapping | None = None,
    species: tuple[str, ...] | None = None,
) -> tuple[float, dict[str, float], dict, dict[str, float] | None]:
    """Regional criteria and calibration from matching distance and statistical parameters."""
    values, fitted = {}, {}
    configured = with_dropout_mode(parameters[0], inference=True, key=None)
    previous_region, latent = None, None
    for batch in batches:
        if batch.region.name != previous_region:
            latent = _backbone(parameters[0], batch.payload.features, None)
            previous_region = batch.region.name
        distances = _group_distances(
            configured, latent, batch.payload, jnp.asarray(batch.species_index), batch.context
        )
        group_state = config.loss._group_state(parameters[1], batch.key)
        calibration = None if calibrations is None else calibrations.get(batch.key)
        if calibrations is not None and group_state is not None and calibration is None:
            raise ValueError(
                "Validation requires a training calibration for the same region–species group"
            )
        fixed = None if calibration is None else config.loss._fixed_parameters(calibration)
        value = float(_criterion(config.loss, distances, batch.payload.context, group_state, fixed))
        if not np.isfinite(value):
            raise FloatingPointError(
                f"{batch.region.name}, species {batch.species_index}: nonfinite loss at step {step}"
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
        if parameters[0].num_species == 1 or species is None
        else {
            species[index]: float(
                np.mean([values[b.key] for b in batches if b.species_index == index])
            )
            for index in sorted({b.species_index for b in batches})
        }
    )
    return (
        sum(batch.loss_weight * values[batch.key] for batch in batches),
        regional,
        fitted,
        by_species,
    )


def train(
    region: RegionCollection,
    observations: ObservationCollection,
    *,
    distance_model: AbstractDistanceModel | None = None,
    config: TrainingConfig | None = None,
    species: Sequence[str] | None = None,
    validation: tuple[RegionCollection, ObservationCollection] | None = None,
    state: TrainingState | None = None,
    on_step: Callable[[TrainingState], None] | None = None,
) -> TrainingState:
    """Joint training with equal species and within-species region influence.

    ``config.steps`` is the total target update count. Evaluation at zero, every
    ``eval_every`` updates, and the final step selects by validation loss or keeps
    the final model without validation. ``on_step`` receives resumable state after
    every update; supplied ``state`` continues the latest parameters and random state.

    Training always compiles its update and evaluation steps. To debug them
    eagerly, call ``train`` inside ``with jax.disable_jit():``.

    Args:
        region: One region batch, a sequence, or a mapping keyed by region name.
        observations: Genetic observations aligned with the supplied regions; each
            region may supply observations for several species.
        distance_model: Initial distance model for a new training run; omit for continuation.
        config: Loss and training settings. For continuation, only the total step
            budget may increase; omitted settings use the saved configuration.
        species: Species names in distance-model parameter order. Required for
            multispecies models; a single species can be inferred from the inputs.
        validation: Regions and disjoint observations used to select the model by
            validation loss, using calibration estimated from training observations.
        state: Saved training state to continue with unchanged inputs and settings.
        on_step: Callback receiving resumable state after initialization and each update.

    Returns:
        Training progress, optimizer and random state, and the latest and selected models.

    Examples:
        >>> loss = MLPELoss()
        >>> state = train(region, observations.select_pairs(training_pairs),
        ...               distance_model=distance_model,
        ...               config=TrainingConfig(loss=loss, steps=500))
        >>> fitted = state.selected_model
        >>> continued = train(region, observations.select_pairs(training_pairs), state=state,
        ...                   config=dataclasses.replace(state.config, steps=1000))

        Zero steps initializes statistical parameters and fits the coefficients at those
        parameters; use ``calibrate`` to optimize calibration on a frozen distance
        model. Built-in validation uses training-fitted coefficients and distance moments.
    """
    if species is not None and (
        isinstance(species, (str, bytes)) or not isinstance(species, Sequence)
    ):
        raise ValueError("Species order must be a sequence of names")
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
        if species is not None and tuple(species) != state.latest_model.species:
            raise ValueError("Continuation must retain the saved species order")
        species = state.latest_model.species
    assert distance_model is not None and config is not None
    if on_step is not None and not callable(on_step):
        raise ValueError("on_step must be callable or None")
    if state is not None:
        from dataclasses import replace

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
        region, observations, species, distance_model.num_species, validation=validation
    )
    first_region, first_observations = prepared.inputs[0]
    species = prepared.species
    prototype = LandscapeGeneticsModel(
        distance_model,
        config.loss._calibration(first_observations.target, {}),
        first_region.feature_array.shape[-1],
        feature_names=first_region.feature_names,
        species=species,
    )
    random_key = (
        jax.random.split(jax.random.PRNGKey(config.seed))[1] if state is None else state.rng_key
    )
    training_batches, validation_batches = _prepare_batches(
        prepared.training, prepared.validation, distance_model, config
    )

    loss_state = (
        state.loss_state
        if state is not None
        else config.loss._initialize(
            {batch.key: batch.payload.context for batch in training_batches}
        )
    )
    parameters = (distance_model, loss_state)
    optimizer = optax.adam(config.learning_rate)
    optimizer_state = (
        state.optimizer_state
        if state is not None
        else optimizer.init(eqx.filter(parameters, eqx.is_inexact_array))
    )

    regional_batches = tuple(
        tuple(b for b in training_batches if b.region.name == name)
        for name in dict.fromkeys(b.region.name for b in training_batches)
    )
    region_count = len(regional_batches)
    selected_count = min(region_count, config.region_batch_size or region_count)

    training_groups = tuple(batch.key for batch in training_batches)
    validation_groups = tuple(batch.key for batch in validation_batches)
    if state is not None and (
        training_groups != state.training_groups or validation_groups != state.validation_groups
    ):
        raise ValueError("Continuation requires unchanged training and validation groups")

    def make_model(calibrations) -> LandscapeGeneticsModel:
        calibration = config.loss._calibration(first_observations.target, calibrations)
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
            loss_state=parameters[1],
            optimizer_state=optimizer_state,
            rng_key=random_key,
            config=config,
            history=tuple(history),
            latest_model=latest,
            selected_model=selected,
            selected_step=selected_step,
            step=step,
            training_groups=training_groups,
            validation_groups=validation_groups,
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
                        groups[i].key,
                        jnp.asarray(groups[i].species_index),
                        jnp.asarray(groups[i].loss_weight * correction),
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
            parameters, optimizer_state = _apply_gradient(
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
        elif on_step is not None:
            _, _, current_calibrations, _ = _evaluate_regions(
                parameters, training_batches, config, step, species=species
            )
            latest = make_model(current_calibrations)
        if on_step is not None:
            on_step(continuation())
    return continuation()
