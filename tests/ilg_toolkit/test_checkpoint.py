"""Tests of checkpoints: continuation, optimizer and penalty supply, and file integrity."""

import os
from dataclasses import replace

import equinox as eqx
import jax.numpy as jnp
import optax
import pytest
from ilg_toolkit import (
    ArtifactError,
    load_checkpoint,
    Log1pLoss,
    MLPELoss,
    PairwiseObservations,
    RidgePenalty,
    save_checkpoint,
    train,
    TrainingConfig,
)


def _two_species_regions(training_problem):
    """Returns two regions of two species, with region and species sampling and MLPE."""
    region, groups, model = training_problem("embedding", species=("frog", "newt"))
    regions = (region, replace(region, name="valley"))
    observations = (groups, tuple(replace(o, values=o.values * 1.2) for o in groups))
    config = TrainingConfig(
        loss=MLPELoss(),
        steps=2,
        eval_every=2,
        seed=21,
        optimizer=optax.adam(1e-4),
        region_batch_size=1,
        species_batch_size=1,
        species_microbatch_size=1,
    )
    return regions, observations, model, config, None


def _dropout_validation(training_problem):
    """Returns dropout keys and validation, with the model selected at step 0."""
    region, observations, model = training_problem("dropout")
    # Initial predictions as validation targets give step 0 the lowest validation loss.
    prediction = model.predict(region)
    validation = (
        replace(region, name="validation"),
        PairwiseObservations.from_pairs(
            prediction.pairs, prediction.values, target=observations.target
        ),
    )
    config = TrainingConfig(
        loss=Log1pLoss(), steps=2, eval_every=2, seed=21, optimizer=optax.adam(0.002)
    )
    return region, observations, model, config, validation


@pytest.mark.parametrize(
    "build", [_two_species_regions, _dropout_validation], ids=["sampled-species", "validation"]
)
def test_resume_equals_uninterrupted(tmp_path, training_problem, build):
    region, observations, model, config, validation = build(training_problem)
    states = []
    uninterrupted = train(
        region,
        observations,
        distance_model=model,
        config=config,
        validation=validation,
        on_step=states.append,
    )
    path = tmp_path / "checkpoint.eqx"
    save_checkpoint(path, states[1])
    restored = load_checkpoint(path, distance_model=model, optimizer=config.optimizer)
    assert eqx.tree_equal(restored, states[1])
    resumed = train(region, observations, state=restored, validation=validation)
    assert eqx.tree_equal(resumed, uninterrupted)


def test_continuing_lbfgs_with_the_penalty_reproduces_the_uninterrupted_run(
    tmp_path, ScalarEmbedding, regions_and_observations
):
    regions, observations = regions_and_observations()
    optimizer = optax.lbfgs()
    model = ScalarEmbedding(jnp.asarray(1.0))
    penalty = RidgePenalty(prior_sd=0.5, where=lambda current: current.weight)
    config = TrainingConfig(loss=Log1pLoss(), steps=3, optimizer=optimizer, penalty=penalty)
    states = []
    uninterrupted = train(
        regions, observations, distance_model=model, config=config, on_step=states.append
    )
    path = tmp_path / "penalized.eqx"
    save_checkpoint(path, states[1])
    restored = load_checkpoint(path, distance_model=model, optimizer=optimizer, penalty=penalty)
    assert restored.config.penalty is penalty
    assert eqx.tree_equal(train(regions, observations, state=restored), uninterrupted)


@pytest.fixture(scope="module")
def initial_state(training_problem):
    """Returns a distance model and its training state at step zero."""
    region, observations, distance_model = training_problem("embedding")
    state = train(
        region,
        observations,
        distance_model=distance_model,
        config=TrainingConfig(loss=Log1pLoss(), steps=0),
    )
    return distance_model, state


def test_loading_rejects_an_incompatible_file_template_or_optimizer(
    initial_state, WeightedEmbedding, tmp_path
):
    model, state = initial_state
    path = tmp_path / "checkpoint.eqx"
    save_checkpoint(path, state)
    optimizer = state.config.optimizer
    with pytest.raises(ArtifactError):
        load_checkpoint(path, distance_model=WeightedEmbedding(jnp.zeros(3)), optimizer=optimizer)
    # AdaBelief keeps Adam's leaf paths, shapes and dtypes in another state type.
    for other in (optax.sgd(0.1), optax.adabelief(0.1)):
        with pytest.raises(ArtifactError, match="optimizer"):
            load_checkpoint(path, distance_model=model, optimizer=other)
    path.write_bytes(path.read_bytes()[:-20])
    with pytest.raises(ArtifactError):
        load_checkpoint(path, distance_model=model, optimizer=optimizer)


def test_interrupted_replacement_keeps_the_previous_checkpoint(
    initial_state, tmp_path, monkeypatch
):
    _, state = initial_state
    path = tmp_path / "atomic.eqx"
    save_checkpoint(path, state)
    original = path.read_bytes()

    def interrupt(source, destination):
        raise InterruptedError("simulated filesystem interruption")

    monkeypatch.setattr(os, "replace", interrupt)
    with pytest.raises(InterruptedError):
        save_checkpoint(path, state)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("change", [{"step": 1}, {"selected_step": 1}, {"history": ()}])
def test_saving_rejects_an_inconsistent_state(initial_state, tmp_path, change):
    _, state = initial_state
    with pytest.raises(ValueError):
        save_checkpoint(tmp_path / "invalid.eqx", replace(state, **change))


def test_loading_requires_float64(initial_state, tmp_path, float64_required):
    distance_model, state = initial_state
    path = tmp_path / "checkpoint.eqx"
    save_checkpoint(path, state)
    with float64_required():
        load_checkpoint(path, distance_model=distance_model, optimizer=state.config.optimizer)
