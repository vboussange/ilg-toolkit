"""Scientific behavior of persistence."""

from dataclasses import replace
from typing import cast

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ilg_toolkit import (
    ArtifactError,
    DirectCalibrationModule,
    LandscapeGeneticsModel,
    Log1pLoss,
    MLPECalibrationModule,
    MLPELoss,
    PairwiseObservations,
    RegionBatch,
    ResistanceSolverConfig,
    TargetSpec,
    TrainingConfig,
    load_checkpoint,
    load_model,
    save_checkpoint,
    save_model,
    train,
)
from ilg_toolkit.models import ConductanceModel


# Direct and MLPE calibrations, without and with species, save different trees.
@pytest.mark.parametrize(("loss", "num_species"), [(Log1pLoss(), 1), (MLPELoss(), 2)])
def test_saved_precision_is_preserved(tmp_path, training_problem, loss, num_species):
    with jax.enable_x64():
        region, observations, distance_model = training_problem(
            "embedding", num_species=num_species
        )
        species = ("species-1", "species-0") if num_species > 1 else None
        state = train(
            region,
            observations,
            distance_model=distance_model,
            config=TrainingConfig(loss=loss, steps=0),
            species=species,
        )
        for kind in ("model", "checkpoint"):
            path = tmp_path / f"{kind}.eqx"
            if kind == "model":
                save_model(path, state.selected_model)
                model = load_model(path, distance_model=distance_model)
                assert eqx.tree_equal(model, state.selected_model)
            else:
                save_checkpoint(path, state)
                restored = load_checkpoint(path, distance_model=distance_model)
                assert eqx.tree_equal(restored, state)
                model = restored.selected_model
            for name in species or (None,):
                predictions = state.selected_model.predict(region, species=name).values
                np.testing.assert_array_equal(
                    model.predict(region, species=name).values, predictions
                )
                assert predictions.dtype == jnp.float64
            with jax.enable_x64(False):
                narrow = jax.tree.map(
                    lambda leaf: leaf.astype(jnp.float32) if eqx.is_inexact_array(leaf) else leaf,
                    distance_model,
                )
                with pytest.raises(ArtifactError):
                    load = load_model if kind == "model" else load_checkpoint
                    load(path, distance_model=narrow)
                assert not jax.config.read("jax_enable_x64")


def test_odds_target_model_reloads_and_predicts_in_original_units(
    training_problem, WeightedEmbedding, tmp_path
):
    region, observations, distance_model = training_problem("embedding")
    target = TargetSpec("Fst", units="proportion", transform="odds")
    observations = replace(observations, values=observations.values / 2, target=target)
    state = train(
        region,
        observations,
        distance_model=distance_model,
        config=TrainingConfig(loss=Log1pLoss(), steps=3, learning_rate=0.05),
    )
    assert state.history[-1].training_loss < state.history[0].training_loss
    model = state.selected_model
    path = tmp_path / "odds.eqx"
    save_model(path, model)
    restored = load_model(path, distance_model=WeightedEmbedding(jnp.ones(2)))
    distances = np.asarray(model.distance_model.predict(region).values)
    prediction = restored.predict(region)
    assert restored.target == target
    np.testing.assert_array_equal(prediction.values, model.predict(region).values)
    np.testing.assert_allclose(prediction.values, distances / (1 + distances))
    assert np.all((np.asarray(prediction.values) >= 0) & (np.asarray(prediction.values) < 1))


def test_direct_model_reload_uses_caller_template_and_preserves_predictions(
    training_problem, WeightedEmbedding, tmp_path
):
    region, observations, distance_model = training_problem("embedding")
    observations = replace(observations, target=replace(observations.target, transform="log1p"))
    model = train(
        region,
        observations,
        distance_model=distance_model,
        config=TrainingConfig(steps=0, loss=Log1pLoss()),
    ).selected_model
    path = tmp_path / "direct.eqx"
    save_model(path, model)
    restored = load_model(path, distance_model=WeightedEmbedding(jnp.asarray([0.5, -0.5])))
    np.testing.assert_array_equal(
        restored.distance_model.predict(region).as_matrix(),
        model.distance_model.predict(region).as_matrix(),
    )
    np.testing.assert_array_equal(restored.predict(region).values, model.predict(region).values)
    assert restored.target == model.target
    assert restored.feature_names == model.feature_names


def test_custom_mlpe_reload_preserves_regional_posteriors_and_conditioning(
    persistence_problem, WeightedEmbedding, tmp_path
):
    from dataclasses import replace

    region, observations = persistence_problem()
    other = replace(region, name="second-calibration")
    other_observations = PairwiseObservations.from_matrix(
        observations.sampling_unit_ids, observations.values * 2.0, target=observations.target
    )
    labels = set(region.sampling_unit_ids[:4])
    pairs = tuple(pair for pair in observations.observed_pairs if set(pair) <= labels)
    model = train(
        (region, other),
        (observations.select_pairs(pairs[:-1]), other_observations.select_pairs(pairs[:-1])),
        distance_model=WeightedEmbedding(jnp.asarray([0.2, -0.7])),
        config=TrainingConfig(steps=0, loss=MLPELoss()),
        validation=(region, observations.select_pairs(pairs[-1:])),
    ).selected_model
    path = tmp_path / "custom-mlpe.eqx"
    save_model(path, model)
    restored = load_model(path, distance_model=WeightedEmbedding(jnp.asarray([4.0, 3.0])))
    assert type(restored.distance_model) is WeightedEmbedding
    for prepared in (region, other):
        np.testing.assert_array_equal(
            restored.predict(prepared).values, model.predict(prepared).values
        )
        assert eqx.tree_equal(
            cast(MLPECalibrationModule, restored.calibration).regions[(prepared.name, 0)],
            cast(MLPECalibrationModule, model.calibration).regions[(prepared.name, 0)],
        )
    query = (("population-4", "population-2"), ("population-5", "population-3"))
    before = model.predict(region, pairs=query, known_effects=True)
    after = restored.predict(region, pairs=query, known_effects=True)
    np.testing.assert_array_equal(after.values, before.values)
    np.testing.assert_array_equal(after.extras["model_variance"], before.extras["model_variance"])
    support = PairwiseObservations.from_pairs(
        [("population-4", "population-0")], [1.5], target=model.target
    )
    before = model.predict(region, pairs=query, support=support)
    after = restored.predict(region, pairs=query, support=support)
    np.testing.assert_array_equal(after.values, before.values)
    np.testing.assert_array_equal(after.extras["model_variance"], before.extras["model_variance"])
    assert (
        cast(MLPECalibrationModule, restored.calibration).regions[(region.name, 0)].converged
        is False
    )


def test_conductance_reload_preserves_surface_distances_and_solver_settings(
    PointConductance, tmp_path
):
    features = np.random.default_rng(1).uniform(0.5, 1.5, size=(8, 8, 2))
    region = RegionBatch(
        "graph-reload",
        cx.Field(jnp.asarray(features), dims=("row", "column", "feature")),
        ("west", "north", "east"),
        np.array([[0, 0], [0, 7], [7, 7]]),
        feature_names=("elevation", "canopy"),
    )
    solver_config = ResistanceSolverConfig(rtol=1e-9, atol=1e-10)
    assert solver_config != ResistanceSolverConfig()
    model = LandscapeGeneticsModel(
        PointConductance(jnp.asarray(0.3), solver_config=solver_config),
        DirectCalibrationModule(TargetSpec("divergence")),
        feature_count=2,
        feature_names=region.feature_names,
    )
    path = tmp_path / "conductance.eqx"
    save_model(path, model)
    restored = load_model(path, distance_model=PointConductance(jnp.asarray(-1.0)))
    restored_model = cast(ConductanceModel, restored.distance_model)
    np.testing.assert_array_equal(
        restored_model.conductance_surface(region),
        cast(ConductanceModel, model.distance_model).conductance_surface(region),
    )
    np.testing.assert_array_equal(
        restored_model.predict(region).as_matrix(),
        model.distance_model.predict(region).as_matrix(),
    )
    np.testing.assert_array_equal(restored.predict(region).values, model.predict(region).values)
    assert restored_model.solver_config == solver_config


def test_model_reload_uses_native_shape_and_dtype_compatibility(WeightedEmbedding, tmp_path):
    with jax.enable_x64():
        distance_model = WeightedEmbedding(jnp.asarray([0.2, -0.4], dtype=jnp.float32))
        model = LandscapeGeneticsModel(
            distance_model, DirectCalibrationModule(TargetSpec("divergence")), 2
        )
        path = tmp_path / "model.eqx"
        save_model(path, model)
        assert eqx.tree_equal(load_model(path, distance_model=distance_model), model)
        with pytest.raises(ArtifactError):
            load_model(path, distance_model=WeightedEmbedding(jnp.zeros(3, dtype=jnp.float32)))
        with pytest.raises(ArtifactError):
            load_model(path, distance_model=WeightedEmbedding(jnp.zeros(2, dtype=jnp.float64)))

        truncated = tmp_path / "truncated.eqx"
        truncated.write_bytes(path.read_bytes()[:-10])
        with pytest.raises(ArtifactError):
            load_model(truncated, distance_model=distance_model)


def test_interrupted_model_replacement_preserves_previous_predictions(
    persistence_problem, WeightedEmbedding, tmp_path, monkeypatch
):
    import os

    region, _ = persistence_problem()

    def model(log_weights):
        return LandscapeGeneticsModel(
            WeightedEmbedding(jnp.asarray(log_weights)),
            DirectCalibrationModule(TargetSpec("divergence")),
            2,
            feature_names=region.feature_names,
        )

    first, replacement = model([0.2, -0.7]), model([1.0, 0.5])
    path = tmp_path / "model.eqx"
    save_model(path, first)

    def interrupt(source, destination):
        raise OSError("interrupted replacement")

    monkeypatch.setattr(os, "replace", interrupt)
    with pytest.raises(OSError):
        save_model(path, replacement)
    restored = load_model(path, distance_model=replacement.distance_model)
    np.testing.assert_array_equal(restored.predict(region).values, first.predict(region).values)
    assert not np.array_equal(replacement.predict(region).values, first.predict(region).values)
    assert set(tmp_path.iterdir()) == {path}
