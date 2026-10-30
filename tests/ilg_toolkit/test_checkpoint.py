"""Tests of checkpoints: continuation, penalties, random keys, optimizer state and integrity."""

import functools
import json
from dataclasses import asdict, dataclass, replace
from types import SimpleNamespace
from typing import Any, cast

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from ilg_toolkit import (
    ArtifactError,
    load_checkpoint,
    Log1pLoss,
    MLPELoss,
    PairwiseObservations,
    RegionBatch,
    RidgePenalty,
    save_checkpoint,
    train,
    TrainingConfig,
)
from ilg_toolkit.models import AbstractDistanceModel


@dataclass(frozen=True)
class _ResumeScenario:
    """A training run whose continuation from a saved state equals the uninterrupted run."""

    regions: tuple[RegionBatch, ...]
    targets: tuple[Any, ...]
    distance_model: AbstractDistanceModel
    config: TrainingConfig
    validation: tuple[tuple[RegionBatch, ...], tuple[Any, ...]] | None = None


def _two_regions(
    training_problem, family: str, species: tuple[str, ...] | None = None
) -> tuple[tuple[RegionBatch, ...], tuple[Any, ...], AbstractDistanceModel]:
    """Returns the problem's region and a renamed copy with targets scaled by 1.2."""
    region, observations, model = training_problem(family, species=species)
    other = replace(region, name="valley")
    if isinstance(observations, PairwiseObservations):
        other_observations = replace(observations, values=observations.values * 1.2)
    else:
        other_observations = tuple(
            replace(observed, values=observed.values * 1.2) for observed in observations
        )
    return (region, other), (observations, other_observations), model


_CONFIG = TrainingConfig(
    loss=MLPELoss(), steps=2, eval_every=2, seed=21, optimizer=optax.adam(1e-4)
)


def _conductance_region_batches(training_problem):
    """Returns random region minibatches of a conductance model trained with MLPE."""
    regions, targets, model = _two_regions(training_problem, "conductance")
    return _ResumeScenario(regions, targets, model, replace(_CONFIG, region_batch_size=1))


def _dropout_validation(training_problem):
    """Returns dropout keys and a completed budget, with the model selected at step 0."""
    regions, targets, model = _two_regions(training_problem, "dropout")
    # Initial predictions as validation targets give step 0 the lowest validation loss.
    validation_targets = tuple(
        PairwiseObservations.from_pairs(prediction.pairs, prediction.values, target=o.target)
        for prediction, o in zip((model.predict(r) for r in regions), targets, strict=True)
    )
    validation_regions = tuple(replace(r, name=r.name + "-validation") for r in regions)
    return _ResumeScenario(
        regions,
        targets,
        model,
        replace(_CONFIG, loss=Log1pLoss(), optimizer=optax.adam(0.002)),
        validation=(validation_regions, validation_targets),
    )


def _species_microbatch_validation(training_problem):
    """Returns species batches and microbatches of an embedding model with MLPE validation."""
    regions, groups, model = _two_regions(training_problem, "embedding", species=("frog", "newt"))
    validation = tuple(group[0].select_pairs(group[0].observed_pairs[-1:]) for group in groups)
    targets = tuple(tuple(o.select_pairs(o.observed_pairs[:-1]) for o in group) for group in groups)
    config = replace(_CONFIG, region_batch_size=1, species_batch_size=1, species_microbatch_size=1)
    return _ResumeScenario(
        regions,
        targets,
        model,
        config,
        validation=(regions, validation),
    )


def _penalized_conductance_region_batches(training_problem):
    """Returns random region minibatches, in which the penalty enters each update once."""
    scenario = _conductance_region_batches(training_problem)
    penalty = RidgePenalty(prior_sd=0.5, where=lambda model: model.log_scale)
    return replace(scenario, config=replace(scenario.config, penalty=penalty))


def _penalized_dropout_validation(training_problem):
    """Returns a selected model at step 0, with the penalty on a nested field."""
    scenario = _dropout_validation(training_problem)
    penalty = RidgePenalty(prior_sd=0.5, where=lambda model: model.distance_model.weight)
    return replace(scenario, config=replace(scenario.config, penalty=penalty))


@pytest.mark.parametrize(
    "build",
    [
        _conductance_region_batches,
        _dropout_validation,
        _species_microbatch_validation,
        _penalized_conductance_region_batches,
        _penalized_dropout_validation,
    ],
    ids=[
        "conductance-regions",
        "dropout-validation",
        "species-microbatches",
        "penalized-conductance-regions",
        "penalized-dropout-validation",
    ],
)
def test_resume_equals_uninterrupted(tmp_path, training_problem, build):
    scenario = build(training_problem)
    regions, targets, validation = scenario.regions, scenario.targets, scenario.validation
    states = []
    uninterrupted = train(
        regions,
        targets,
        distance_model=scenario.distance_model,
        config=scenario.config,
        validation=validation,
        on_step=states.append,
    )
    partial = states[1]
    if validation is not None:
        assert partial.selected_step == 0
        assert not eqx.tree_equal(partial.latest_model, partial.selected_model)
    path = tmp_path / "checkpoint.eqx"
    save_checkpoint(path, partial)
    restored = load_checkpoint(
        path,
        distance_model=scenario.distance_model,
        optimizer=scenario.config.optimizer,
        penalty=scenario.config.penalty,
    )
    assert eqx.tree_equal(restored, partial)
    if scenario.config.penalty is not None:
        assert restored.config.penalty is scenario.config.penalty
        record = uninterrupted.history[-1]
        assert record.training_objective > record.training_loss
    # Reversing region order also checks restoration of the random trajectory.
    resumed = train(regions[::-1], targets[::-1], state=restored, validation=validation)
    assert eqx.tree_equal(resumed, uninterrupted)
    for prepared in regions:
        species = scenario.distance_model.species
        query = None if species is None else dict.fromkeys(species)
        assert eqx.tree_equal(
            resumed.selected_model.predict(prepared, query),
            uninterrupted.selected_model.predict(prepared, query),
        )
    if validation is not None and scenario.distance_model.species is None:
        # A completed budget returns the saved state without further updates.
        save_checkpoint(path, resumed)
        completed = load_checkpoint(
            path,
            distance_model=scenario.distance_model,
            optimizer=scenario.config.optimizer,
            penalty=scenario.config.penalty,
        )
        callbacks = []
        unchanged = train(
            regions, targets, state=completed, validation=validation, on_step=callbacks.append
        )
        assert callbacks == []
        assert eqx.tree_equal(unchanged, resumed)
        assert resumed.selected_step == 0
        validation_loss = resumed.history[-1].validation_loss
        assert validation_loss is not None and validation_loss > resumed.selected_loss
    if validation is not None and scenario.distance_model.species is not None:
        assert resumed.history[-1].validation_by_species is not None
        assert set(resumed.history[-1].validation_by_species) == {"frog"}


@pytest.fixture(scope="module")
def initial_state(training_problem):
    """Returns the embedding problem and its training state at step zero."""
    region, observations, distance_model = training_problem("embedding")
    state = train(
        region,
        observations,
        distance_model=distance_model,
        config=TrainingConfig(loss=Log1pLoss(), steps=0),
    )
    return region, observations, distance_model, state


def test_checkpoint_restores_the_typed_random_key_and_its_stream(initial_state, tmp_path):
    _, _, distance_model, state = initial_state
    path = tmp_path / "key.eqx"
    save_checkpoint(path, state)
    restored = load_checkpoint(path, distance_model=distance_model, optimizer=optax.adam(1e-3))
    assert jnp.issubdtype(restored.rng_key.dtype, jax.dtypes.prng_key)
    np.testing.assert_array_equal(
        jax.random.uniform(restored.rng_key, (4,)), jax.random.uniform(state.rng_key, (4,))
    )


def test_checkpoint_rejects_optimizer_precision_that_differs_from_model(training_problem, tmp_path):
    with jax.enable_x64():
        region, observations, model = training_problem("embedding")
        model = jax.tree.map(
            lambda value: value.astype(np.float32) if eqx.is_inexact_array(value) else value, model
        )
        result = train(
            region,
            observations,
            distance_model=model,
            config=TrainingConfig(loss=Log1pLoss(), steps=0),
        )

        changed_optimizer = jax.tree.map(
            lambda value: value.astype(np.float64) if value.dtype == np.float32 else value,
            result.optimizer_state,
        )
        path = tmp_path / "mixed-precision.eqx"
        save_checkpoint(path, replace(result, optimizer_state=changed_optimizer))
        with pytest.raises(ValueError):
            load_checkpoint(path, distance_model=model, optimizer=result.config.optimizer)


def test_interrupted_replacement_retains_previous_complete_generation(
    training_problem, tmp_path, monkeypatch
):
    import os
    from pathlib import Path

    region, observations, model = training_problem("embedding")
    completed = []
    config = TrainingConfig(loss=Log1pLoss(), steps=1, seed=21, optimizer=optax.adam(0.002))
    result = train(
        region,
        observations,
        distance_model=model,
        config=config,
        on_step=completed.append,
    )

    assert eqx.tree_equal(completed[-1], result)
    path = tmp_path / "atomic.eqx"
    save_checkpoint(path, completed[0])
    original = path.read_bytes()

    def interrupt_replace(source, destination):
        assert Path(source).is_file()
        assert Path(source).stat().st_size > 0
        raise InterruptedError("simulated filesystem interruption")

    with monkeypatch.context() as fault:
        fault.setattr(os, "replace", interrupt_replace)
        with pytest.raises(InterruptedError):
            save_checkpoint(path, result)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
    restored = load_checkpoint(path, distance_model=model, optimizer=config.optimizer)
    assert eqx.tree_equal(restored, completed[0])


def test_disk_continuation_uses_caller_supplied_observations(initial_state, tmp_path):
    region, observations, distance_model, original = initial_state
    config = TrainingConfig(loss=Log1pLoss(), steps=0)
    path = tmp_path / "continuation.eqx"
    save_checkpoint(path, original)
    restored = load_checkpoint(path, distance_model=distance_model, optimizer=config.optimizer)
    changed_region = replace(
        region,
        features=cx.field(
            np.concatenate([region.feature_array, region.feature_array + 0.1], axis=0),
            "row",
            "column",
            region.features.axes["feature"],
        ),
        sampling_unit_ids=region.sampling_unit_ids[::-1],
    )
    changed_observations = replace(observations, values=observations.values * 1.2)
    resumed = train(
        changed_region,
        changed_observations.select_pairs(observations.observed_pairs[:-1]),
        state=restored,
        config=replace(config, steps=1),
    )
    assert resumed.step == 1
    assert not eqx.tree_equal(
        resumed.latest_model.distance_model, restored.latest_model.distance_model
    )
    assert np.isfinite(resumed.selected_model.predict(changed_region).values).all()


def test_checkpoint_rejects_truncated_leaves_and_wrong_distance_model_structure(
    initial_state, WeightedEmbedding, tmp_path
):
    _, _, model, result = initial_state
    path = tmp_path / "incompatible.eqx"
    save_checkpoint(path, result)
    with pytest.raises(ArtifactError):
        load_checkpoint(
            path,
            distance_model=WeightedEmbedding(jnp.zeros(3)),
            optimizer=result.config.optimizer,
        )
    path.write_bytes(path.read_bytes()[:-20])
    with pytest.raises(ArtifactError):
        load_checkpoint(path, distance_model=model, optimizer=result.config.optimizer)


@pytest.mark.parametrize("boundary", ["save", "load", "resume"])
def test_checkpoint_enforces_progress_bounds(initial_state, tmp_path, boundary):
    region, observations, distance_model, state = initial_state
    path = tmp_path / "bounds.eqx"
    save_checkpoint(path, state)
    with path.open("rb") as stream:
        header = json.loads(stream.readline())
    changes = (
        {"step": -1},
        {"step": 1},
        {"selected_step": -1},
        {"selected_step": 1},
        {"history": ()},
        {"history": state.history * 2},
    )
    for change in changes:
        invalid = replace(state, **change)
        if boundary == "load":
            header["metadata"]["history"] = [asdict(record) for record in invalid.history]
            with path.open("wb") as stream:
                stream.write(json.dumps(header).encode() + b"\n")
                # Checkpoints store a typed random key as its raw key data.
                stored = replace(invalid, rng_key=jax.random.key_data(invalid.rng_key))
                eqx.tree_serialise_leaves(stream, stored)
        with pytest.raises(ValueError):
            if boundary == "save":
                save_checkpoint(path, invalid)
            elif boundary == "load":
                load_checkpoint(
                    path, distance_model=distance_model, optimizer=state.config.optimizer
                )
            else:
                train(region, observations, state=invalid)


def test_loading_requires_float64(initial_state, tmp_path, float64_required):
    _, _, distance_model, state = initial_state
    path = tmp_path / "checkpoint.eqx"
    save_checkpoint(path, state)
    with float64_required():
        load_checkpoint(path, distance_model=distance_model, optimizer=state.config.optimizer)


def test_checkpoints_resume_with_the_supplied_optimizer(
    tmp_path, ScalarEmbedding, regions_and_observations
):
    regions, observations = regions_and_observations()
    model = ScalarEmbedding(jnp.asarray(1.0))
    config = TrainingConfig(
        loss=Log1pLoss(), steps=100, optimizer=optax.lbfgs(), gradient_tolerance=1e-9
    )
    states = []
    uninterrupted = train(
        regions, observations, distance_model=model, config=config, on_step=states.append
    )
    assert uninterrupted.converged and uninterrupted.step > 2
    path = tmp_path / "lbfgs.eqx"
    save_checkpoint(path, states[2])
    with path.open("rb") as stream:
        assert "optimizer" not in json.loads(stream.readline())["metadata"]["config"]
    # A fresh optimizer instance rebuilds the line-search state, as a template does.
    restored = load_checkpoint(path, distance_model=model, optimizer=optax.lbfgs())
    assert eqx.tree_equal(restored, states[2])
    assert not restored.converged
    resumed = train(regions, observations, state=restored)
    assert eqx.tree_equal(resumed, uninterrupted)

    save_checkpoint(path, uninterrupted)
    converged = load_checkpoint(path, distance_model=model, optimizer=optax.lbfgs())
    assert converged.converged
    assert eqx.tree_equal(converged, uninterrupted)


def test_checkpoints_reject_an_optimizer_with_a_different_state(initial_state, tmp_path):
    _, _, distance_model, state = initial_state
    path = tmp_path / "adam.eqx"
    save_checkpoint(path, state)
    # AdaBelief and AdamW keep Adam's leaf paths, shapes and dtypes in other state types.
    for optimizer in (optax.sgd(0.1), optax.lbfgs(), optax.adabelief(0.1), optax.adamw(0.1)):
        with pytest.raises(ArtifactError, match="optimizer"):
            load_checkpoint(path, distance_model=distance_model, optimizer=optimizer)
    restored = load_checkpoint(path, distance_model=distance_model, optimizer=optax.adam(0.1))
    assert eqx.tree_equal(restored, state)


def test_unpenalized_checkpoints_round_trip_the_training_objective(
    tmp_path, ScalarEmbedding, regions_and_observations
):
    regions, observations = regions_and_observations()
    model = ScalarEmbedding(jnp.asarray(1.0))
    state = train(
        regions,
        observations,
        distance_model=model,
        config=TrainingConfig(loss=Log1pLoss(), steps=2, optimizer=optax.adam(0.1)),
    )
    path = tmp_path / "unpenalized.eqx"
    save_checkpoint(path, state)
    restored = load_checkpoint(path, distance_model=model, optimizer=state.config.optimizer)
    assert restored.config.penalty is None
    assert eqx.tree_equal(restored, state)
    assert [record.training_objective for record in restored.history] == [
        record.training_loss for record in state.history
    ]


def _log_weights(model):
    """Returns the leaves that the ridge penalties of these tests select."""
    return model.log_weights


def test_continuing_lbfgs_with_the_same_penalty_reproduces_the_uninterrupted_run(
    tmp_path, training_problem
):
    regions, targets, model = _two_regions(training_problem, "embedding")
    penalty = RidgePenalty(prior_sd=0.5, where=_log_weights)
    optimizer = optax.lbfgs()
    config = TrainingConfig(loss=MLPELoss(), steps=4, optimizer=optimizer, penalty=penalty)
    states = []
    uninterrupted = train(
        regions, targets, distance_model=model, config=config, on_step=states.append
    )
    assert uninterrupted.step == 4
    record = uninterrupted.history[-1]
    assert record.training_objective > record.training_loss
    path = tmp_path / "penalized.eqx"
    save_checkpoint(path, states[2])
    restored = load_checkpoint(path, distance_model=model, optimizer=optimizer, penalty=penalty)
    assert restored.config.penalty is penalty
    assert eqx.tree_equal(restored, states[2])
    assert eqx.tree_equal(train(regions, targets, state=restored), uninterrupted)


@pytest.fixture(scope="module")
def penalized_checkpoint(training_problem, tmp_path_factory):
    """Returns a checkpoint saved after two updates of a ridge-penalized run."""
    regions, targets, model = _two_regions(training_problem, "embedding")
    optimizer = optax.adam(0.1)
    state = train(
        regions,
        targets,
        distance_model=model,
        config=TrainingConfig(
            loss=Log1pLoss(),
            steps=2,
            optimizer=optimizer,
            penalty=RidgePenalty(prior_sd=0.5, where=_log_weights),
        ),
    )
    path = tmp_path_factory.mktemp("penalized") / "checkpoint.eqx"
    save_checkpoint(path, state)
    return SimpleNamespace(path=path, distance_model=model, optimizer=optimizer, state=state)


def _load_penalized(checkpoint, penalty=None):
    """Returns the checkpoint of the ``penalized_checkpoint`` fixture, loaded with a penalty."""
    return load_checkpoint(
        checkpoint.path,
        distance_model=checkpoint.distance_model,
        optimizer=checkpoint.optimizer,
        penalty=penalty,
    )


def test_loading_a_penalized_checkpoint_requires_a_penalty(penalized_checkpoint):
    with pytest.raises(ArtifactError, match="records a penalty"):
        _load_penalized(penalized_checkpoint)


def test_loading_an_unpenalized_checkpoint_rejects_a_penalty(initial_state, tmp_path):
    _, _, distance_model, state = initial_state
    path = tmp_path / "unpenalized.eqx"
    save_checkpoint(path, state)
    with pytest.raises(ArtifactError, match="records no penalty"):
        load_checkpoint(
            path,
            distance_model=distance_model,
            optimizer=state.config.optimizer,
            penalty=RidgePenalty(prior_sd=0.5, where=_log_weights),
        )


def test_a_checkpoint_records_the_penalty_of_the_latest_model_and_not_the_selected_one(
    tmp_path, training_problem
):
    scenario = _penalized_dropout_validation(training_problem)
    state = train(
        scenario.regions,
        scenario.targets,
        distance_model=scenario.distance_model,
        config=scenario.config,
        validation=scenario.validation,
    )
    assert (state.selected_step, state.step) == (0, 2)
    path = tmp_path / "penalized.eqx"
    save_checkpoint(path, state)
    with path.open("rb") as stream:
        recorded = json.loads(stream.readline())["metadata"]["penalty_value"]
    weight = float(cast(Any, state.latest_model.distance_model).weight)
    assert weight != float(cast(Any, state.selected_model.distance_model).weight)
    # A ridge with prior_sd 0.5 is twice the squared weight, and no pair count enters.
    np.testing.assert_allclose(recorded, 2 * weight**2, rtol=1e-12)


@pytest.mark.parametrize(
    "changed",
    [
        RidgePenalty(prior_sd=1.0, where=_log_weights),
        RidgePenalty(prior_sd=0.5, where=lambda model: model.log_weights[0]),
        lambda model: jnp.sum(model.log_weights**2),
        lambda model: jnp.asarray(jnp.nan),
    ],
    ids=["prior_sd", "where", "function", "nonfinite"],
)
def test_loading_rejects_a_penalty_with_a_different_value(penalized_checkpoint, changed):
    assert penalized_checkpoint.state.step > 0
    with pytest.raises(ArtifactError, match="differs"):
        _load_penalized(penalized_checkpoint, changed)


@pytest.mark.parametrize("value_scale", [1 + 1e-11, 1 - 1e-11], ids=["above", "below"])
def test_loading_accepts_a_penalty_value_within_the_relative_tolerance(
    penalized_checkpoint, value_scale
):
    penalty = RidgePenalty(prior_sd=0.5 / value_scale**0.5, where=_log_weights)
    assert _load_penalized(penalized_checkpoint, penalty).config.penalty is penalty


@pytest.mark.parametrize("value_scale", [1 + 1e-8, 1 - 1e-8], ids=["above", "below"])
def test_loading_rejects_a_penalty_value_beyond_the_relative_tolerance(
    penalized_checkpoint, value_scale
):
    penalty = RidgePenalty(prior_sd=0.5 / value_scale**0.5, where=_log_weights)
    with pytest.raises(ArtifactError, match="differs"):
        _load_penalized(penalized_checkpoint, penalty)


def test_a_separately_built_penalty_with_the_same_value_continues_training(
    penalized_checkpoint, training_problem
):
    regions, targets, _ = _two_regions(training_problem, "embedding")
    restored = _load_penalized(
        penalized_checkpoint, RidgePenalty(prior_sd=0.5, where=lambda model: model.log_weights)
    )
    continued = train(regions, targets, state=restored, config=replace(restored.config, steps=3))
    assert continued.step == 3


def _step_zero_checkpoint(path, distance_model, region, observations):
    """Saves the step-zero state of a ridge with prior_sd 0.5 and returns its optimizer."""
    state = train(
        region,
        observations,
        distance_model=distance_model,
        config=TrainingConfig(
            loss=Log1pLoss(), steps=0, penalty=RidgePenalty(prior_sd=0.5, where=_log_weights)
        ),
    )
    save_checkpoint(path, state)
    return state.config.optimizer


def test_a_step_zero_checkpoint_compares_the_penalty_at_the_initial_parameters(
    tmp_path, WeightedEmbedding, embedding_problem
):
    region, observations = embedding_problem("alpine")
    path = tmp_path / "initial.eqx"
    changed = RidgePenalty(prior_sd=1.0, where=_log_weights)

    model = WeightedEmbedding(jnp.ones(2))
    optimizer = _step_zero_checkpoint(path, model, region, observations)
    with pytest.raises(ArtifactError, match="differs"):
        load_checkpoint(path, distance_model=model, optimizer=optimizer, penalty=changed)

    # A ridge on all-zero parameters is 0 at any prior_sd.
    model = WeightedEmbedding(jnp.zeros(2))
    optimizer = _step_zero_checkpoint(path, model, region, observations)
    restored = load_checkpoint(path, distance_model=model, optimizer=optimizer, penalty=changed)
    assert restored.step == 0


@pytest.mark.parametrize(
    "penalty",
    [
        "ridge",
        lambda model: jnp.ones(2),
        RidgePenalty(prior_sd=0.5, where=lambda model: model.coefficients),
        RidgePenalty(prior_sd=0.5, where=lambda model: model.log_weights[:, 0, 0]),
        lambda model: 1 / 0,
    ],
    ids=["not-callable", "not-scalar", "attribute-error", "index-error", "any-exception"],
)
def test_loading_rejects_a_penalty_that_cannot_be_evaluated_at_the_latest_model(
    penalized_checkpoint, penalty
):
    with pytest.raises(ArtifactError):
        _load_penalized(penalized_checkpoint, penalty)


def _squared_weight(distance_model):
    """Returns the sum of the squared weights: a penalty that is a plain function."""
    return jnp.sum(distance_model.weight**2)


def _scaled_squared_weight(distance_model, scale):
    """Returns the sum of the squared weights times ``scale``, a ``functools.partial`` setting."""
    return scale * jnp.sum(distance_model.weight**2)


class _SquaredWeight:
    """A penalty that is a callable object and not an Equinox module."""

    def __call__(self, distance_model):
        return jnp.sum(distance_model.weight**2)


@pytest.mark.parametrize(
    "penalty",
    [
        _squared_weight,
        lambda model: jnp.sum(model.weight**2),
        functools.partial(_scaled_squared_weight, scale=2.0),
        _SquaredWeight(),
    ],
    ids=["function", "lambda", "partial", "object"],
)
def test_on_step_saves_training_with_a_penalty_that_is_not_a_module(
    tmp_path, ScalarEmbedding, regions_and_observations, penalty
):
    regions, observations = regions_and_observations()
    model = ScalarEmbedding(jnp.asarray(1.0))
    path = tmp_path / "checkpoint.eqx"
    optimizer = optax.adam(0.1)
    state = train(
        regions,
        observations,
        distance_model=model,
        config=TrainingConfig(loss=Log1pLoss(), steps=2, optimizer=optimizer, penalty=penalty),
        on_step=lambda state: save_checkpoint(path, state),
    )
    restored = load_checkpoint(path, distance_model=model, optimizer=optimizer, penalty=penalty)
    assert restored.config.penalty is penalty
    assert eqx.tree_equal(restored, state)


def test_an_unhashable_penalty_round_trips_through_a_checkpoint_and_continues(
    tmp_path, ScalarEmbedding, UnhashablePenalty, regions_and_observations
):
    regions, observations = regions_and_observations()
    model = ScalarEmbedding(jnp.asarray(1.0))
    optimizer = optax.adam(0.1)
    config = TrainingConfig(
        loss=Log1pLoss(), steps=2, optimizer=optimizer, penalty=UnhashablePenalty(2.0)
    )
    path = tmp_path / "checkpoint.eqx"
    state = train(
        regions,
        observations,
        distance_model=model,
        config=config,
        on_step=lambda state: save_checkpoint(path, state),
    )
    # A separately built penalty with the same value and equal fields loads and continues.
    restored = load_checkpoint(
        path, distance_model=model, optimizer=optimizer, penalty=UnhashablePenalty(2.0)
    )
    assert eqx.tree_equal(restored, state)
    continued = train(regions, observations, state=restored, config=replace(config, steps=3))
    uninterrupted = train(
        regions, observations, distance_model=model, config=replace(config, steps=3)
    )
    assert eqx.tree_equal(continued, uninterrupted)
    with pytest.raises(ArtifactError, match="differs"):
        load_checkpoint(
            path, distance_model=model, optimizer=optimizer, penalty=UnhashablePenalty(3.0)
        )


def test_a_checkpoint_saved_between_evaluations_continues_the_penalized_run(
    tmp_path, ScalarEmbedding, regions_and_observations
):
    regions, observations = regions_and_observations()
    model = ScalarEmbedding(jnp.asarray(1.0))
    optimizer = optax.adam(0.1)
    config = TrainingConfig(
        loss=Log1pLoss(), steps=3, eval_every=2, optimizer=optimizer, penalty=_squared_weight
    )
    states = []
    uninterrupted = train(
        regions, observations, distance_model=model, config=config, on_step=states.append
    )
    between = states[1]
    assert [record.step for record in between.history] == [0]
    path = tmp_path / "between.eqx"
    save_checkpoint(path, between)
    restored = load_checkpoint(
        path, distance_model=model, optimizer=optimizer, penalty=_squared_weight
    )
    assert eqx.tree_equal(restored, between)
    assert eqx.tree_equal(train(regions, observations, state=restored), uninterrupted)


def test_saving_loading_and_continuing_reuse_the_penalty_programs_that_training_compiled(
    tmp_path, ScalarEmbedding, regions_and_observations
):
    regions, observations = regions_and_observations()
    traces = []

    def penalty(model):
        traces.append(None)
        return jnp.sum(model.weight**2)

    # A strongly typed weight gives every update the same compiled signature.
    model = ScalarEmbedding(jnp.asarray(1.0, jnp.float64))
    optimizer = optax.adam(0.1)
    config = TrainingConfig(loss=Log1pLoss(), steps=2, optimizer=optimizer, penalty=penalty)
    state = train(regions, observations, distance_model=model, config=config)
    traces.clear()
    path = tmp_path / "checkpoint.eqx"
    save_checkpoint(path, state)
    restored = load_checkpoint(path, distance_model=model, optimizer=optimizer, penalty=penalty)
    assert traces == []
    train(regions, observations, state=restored, config=replace(restored.config, steps=3))
    # Only the check of the penalty's output shape runs the penalty again.
    assert len(traces) == 1


def test_a_penalty_module_holding_arrays_is_supplied_on_load_and_compared_by_value(
    tmp_path, ScalarEmbedding, CenteredPenalty, regions_and_observations
):
    regions, observations = regions_and_observations()
    model = ScalarEmbedding(jnp.asarray(1.0))
    optimizer = optax.adam(0.1)
    state = train(
        regions,
        observations,
        distance_model=model,
        config=TrainingConfig(
            loss=Log1pLoss(),
            steps=2,
            optimizer=optimizer,
            penalty=CenteredPenalty(jnp.asarray(0.5)),
        ),
    )
    path = tmp_path / "centered.eqx"
    save_checkpoint(path, state)
    restored = load_checkpoint(
        path, distance_model=model, optimizer=optimizer, penalty=CenteredPenalty(jnp.asarray(0.5))
    )
    assert eqx.tree_equal(restored, state)
    with pytest.raises(ArtifactError, match="differs"):
        load_checkpoint(
            path,
            distance_model=model,
            optimizer=optimizer,
            penalty=CenteredPenalty(jnp.asarray(0.6)),
        )


def test_saving_rejects_a_nonfinite_penalty_and_keeps_the_previous_file(
    penalized_checkpoint, tmp_path
):
    state = penalized_checkpoint.state
    path = tmp_path / "checkpoint.eqx"
    save_checkpoint(path, state)
    original = path.read_bytes()
    nonfinite = replace(state, config=replace(state.config, penalty=lambda model: jnp.inf))
    with pytest.raises(ValueError):
        save_checkpoint(path, nonfinite)
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "drop_training_objective", [False, True], ids=["penalty-record", "training-objective"]
)
@pytest.mark.parametrize(
    "supplied",
    [None, RidgePenalty(prior_sd=0.5, where=_log_weights)],
    ids=["without-penalty", "with-penalty"],
)
def test_a_checkpoint_without_a_penalty_record_raises_an_artifact_error(
    penalized_checkpoint, tmp_path, supplied, drop_training_objective
):
    with penalized_checkpoint.path.open("rb") as stream:
        header = json.loads(stream.readline())
        leaves = stream.read()
    del header["metadata"]["penalty_value"]
    if drop_training_objective:
        for record in header["metadata"]["history"]:
            del record["training_objective"]
    path = tmp_path / "no-record.eqx"
    path.write_bytes(json.dumps(header).encode() + b"\n" + leaves)
    with pytest.raises(ArtifactError, match="no penalty record"):
        load_checkpoint(
            path,
            distance_model=penalized_checkpoint.distance_model,
            optimizer=penalized_checkpoint.optimizer,
            penalty=supplied,
        )
