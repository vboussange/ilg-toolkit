"""Scientific behavior of checkpoint."""

import json
from dataclasses import asdict, dataclass, replace
from typing import Any

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from ilg_toolkit import (
    load_checkpoint,
    Log1pLoss,
    MLPELoss,
    PairwiseObservations,
    RegionBatch,
    save_checkpoint,
    train,
    TrainingConfig,
)
from ilg_toolkit.models import AbstractDistanceModel


@dataclass(frozen=True)
class _ResumeScenario:
    """A training run whose interrupted continuation must equal the uninterrupted run."""

    regions: tuple[RegionBatch, ...]
    targets: tuple[Any, ...]
    distance_model: AbstractDistanceModel
    config: TrainingConfig
    validation: tuple[tuple[RegionBatch, ...], tuple[Any, ...]] | None = None


def _two_regions(
    training_problem, family: str, num_species: int = 1
) -> tuple[tuple[RegionBatch, ...], tuple[Any, ...], AbstractDistanceModel]:
    """The problem's region and a renamed copy with targets scaled by 1.2."""
    region, observations, model = training_problem(family, num_species=num_species)
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
    """Random region minibatches of a conductance model fitted with MLPE."""
    regions, targets, model = _two_regions(training_problem, "conductance")
    return _ResumeScenario(regions, targets, model, replace(_CONFIG, region_batch_size=1))


def _dropout_validation(training_problem):
    """Dropout keys and a completed budget, selected at step 0 by validation."""
    regions, targets, model = _two_regions(training_problem, "dropout")
    # Initial predictions as validation targets make step 0 the best validation step.
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
    """Species batches and microbatches of an embedding model with MLPE validation."""
    regions, groups, model = _two_regions(training_problem, "embedding", num_species=2)
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


@pytest.mark.parametrize(
    "build",
    [_conductance_region_batches, _dropout_validation, _species_microbatch_validation],
    ids=["conductance-regions", "dropout-validation", "species-microbatches"],
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
        path, distance_model=scenario.distance_model, optimizer=scenario.config.optimizer
    )
    assert eqx.tree_equal(restored, partial)
    # Reversing region order also checks restoration of the random trajectory.
    resumed = train(regions[::-1], targets[::-1], state=restored, validation=validation)
    assert eqx.tree_equal(resumed, uninterrupted)
    for prepared in regions:
        for name in scenario.distance_model.species or (None,):
            np.testing.assert_array_equal(
                resumed.selected_model.predict(prepared, species=name).values,
                uninterrupted.selected_model.predict(prepared, species=name).values,
            )
    if validation is not None and scenario.distance_model.species is None:
        # A completed budget returns the saved state without further updates.
        save_checkpoint(path, resumed)
        completed = load_checkpoint(
            path, distance_model=scenario.distance_model, optimizer=scenario.config.optimizer
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
        assert set(resumed.history[-1].validation_by_species) == {"species-0"}


@pytest.fixture(scope="module")
def initial_state(training_problem):
    """Embedding problem and its training state at step zero, shared by read-only checks."""
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
            "feature",
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


def test_checkpoint_rejects_truncated_leaves_and_wrong_encoder_structure(
    initial_state, WeightedEmbedding, tmp_path
):
    from ilg_toolkit import ArtifactError

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
    from ilg_toolkit import ArtifactError

    _, _, distance_model, state = initial_state
    path = tmp_path / "adam.eqx"
    save_checkpoint(path, state)
    for optimizer in (optax.sgd(0.1), optax.lbfgs()):
        with pytest.raises(ArtifactError, match="optimizer"):
            load_checkpoint(path, distance_model=distance_model, optimizer=optimizer)
    restored = load_checkpoint(path, distance_model=distance_model, optimizer=optax.adam(0.1))
    assert eqx.tree_equal(restored, state)
