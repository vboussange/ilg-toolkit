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
    DirectCalibrationModule,
    Log1pLoss,
    MLPECalibrationModule,
    MLPELoss,
    PairwiseObservations,
    RegionBatch,
    TargetSpec,
    TrainingConfig,
    load_checkpoint,
    load_model,
    save_checkpoint,
    save_model,
    train,
)
from ilg_toolkit.models import ConductanceModel, UNetEmbeddingDistance


@pytest.mark.parametrize("loss", [Log1pLoss(), MLPELoss()])
@pytest.mark.parametrize("num_species", [1, 2])
def test_saved_precision_is_preserved(tmp_path, training_problem, loss, num_species):
    from ilg_toolkit import ArtifactError

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


def test_direct_model_reload_uses_caller_encoder_and_preserves_predictions(
    training_problem, checkpoint_model, tmp_path
):
    region, observations, encoder = training_problem("unet")
    observations = replace(observations, target=replace(observations.target, transform="log1p"))

    model = train(
        region,
        observations,
        distance_model=encoder,
        config=TrainingConfig(steps=0, loss=Log1pLoss()),
    ).selected_model
    path = tmp_path / "direct.eqx"
    save_model(path, model)
    restored = load_model(path, distance_model=checkpoint_model(7))
    np.testing.assert_array_equal(
        restored.distance_model.predict(region).as_matrix(),
        model.distance_model.predict(region).as_matrix(),
    )
    np.testing.assert_array_equal(restored.predict(region).values, model.predict(region).values)
    assert restored.target == model.target
    assert restored.feature_names == model.feature_names
    assert cast(UNetEmbeddingDistance, restored.distance_model).encoder1.dropout.p == 0.2


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


def test_conductance_reload_preserves_surface_distances_and_solver_settings(tmp_path):
    from ilg_toolkit import LandscapeGeneticsModel, ResistanceSolverConfig
    from ilg_toolkit.models import ResNet9Conductance

    region = RegionBatch(
        "graph-reload",
        cx.Field(
            jnp.asarray(np.random.default_rng(1).normal(size=(8, 8, 2))),
            dims=("row", "column", "feature"),
        ),
        ("west", "north", "east"),
        np.array([[0, 0], [0, 7], [7, 7]]),
        feature_names=("elevation", "canopy"),
    )

    def encoder(seed):
        return ResNet9Conductance(
            2,
            patch_size=4,
            min_conductance=0.02,
            patch_microbatch_size=3,
            solver_config=ResistanceSolverConfig(rtol=1e-9, atol=1e-10)
            if seed == 3
            else ResistanceSolverConfig(),
            key=jax.random.key(seed),
        )

    model = LandscapeGeneticsModel(
        encoder(3),
        DirectCalibrationModule(TargetSpec("divergence")),
        feature_count=2,
        feature_names=region.feature_names,
    )
    path = tmp_path / "conductance.eqx"
    save_model(path, model)
    restored = load_model(path, distance_model=encoder(9))
    np.testing.assert_array_equal(
        cast(ConductanceModel, restored.distance_model).conductance_surface(region),
        cast(ConductanceModel, model.distance_model).conductance_surface(region),
    )
    np.testing.assert_array_equal(
        restored.distance_model.predict(region).as_matrix(),
        model.distance_model.predict(region).as_matrix(),
    )
    np.testing.assert_array_equal(restored.predict(region).values, model.predict(region).values)
    assert (
        cast(ResNet9Conductance, restored.distance_model).solver_config
        == cast(ResNet9Conductance, model.distance_model).solver_config
    )
    assert cast(ResNet9Conductance, restored.distance_model).patch_microbatch_size == 3
    assert cast(ResNet9Conductance, restored.distance_model).min_conductance == 0.02


def test_model_reload_uses_native_shape_and_dtype_compatibility(tmp_path):
    from ilg_toolkit import ArtifactError, LandscapeGeneticsModel

    encoder = UNetEmbeddingDistance(
        2, patch_size=1, base_channels=2, embedding_dim=2, key=jax.random.key(0)
    )
    model = LandscapeGeneticsModel(
        encoder,
        DirectCalibrationModule(TargetSpec("divergence")),
        2,
    )
    path = tmp_path / "model.eqx"
    save_model(path, model)
    wrong_shape = UNetEmbeddingDistance(
        2, patch_size=1, base_channels=2, embedding_dim=3, key=jax.random.key(0)
    )
    with pytest.raises(ArtifactError):
        load_model(path, distance_model=wrong_shape)
    wrong_dtype = jax.tree.map(
        lambda value: value.astype(jnp.float64) if isinstance(value, jax.Array) else value,
        encoder,
    )
    with pytest.raises(ArtifactError):
        load_model(path, distance_model=wrong_dtype)

    truncated = tmp_path / "truncated.eqx"
    truncated.write_bytes(path.read_bytes()[:-10])
    with pytest.raises(ArtifactError):
        load_model(truncated, distance_model=encoder)


def test_interrupted_model_replacement_preserves_previous_predictions(
    persistence_problem, tmp_path, monkeypatch
):
    import os

    from ilg_toolkit import LandscapeGeneticsModel

    region, _ = persistence_problem()

    def model(seed):
        return LandscapeGeneticsModel(
            UNetEmbeddingDistance(
                2, patch_size=1, base_channels=2, embedding_dim=2, key=jax.random.key(seed)
            ),
            DirectCalibrationModule(TargetSpec("divergence")),
            2,
            feature_names=region.feature_names,
        )

    first, replacement = model(3), model(7)
    path = tmp_path / "model.eqx"
    save_model(path, first)

    def interrupt(source, destination):
        raise OSError("interrupted replacement")

    monkeypatch.setattr(os, "replace", interrupt)
    with pytest.raises(OSError):
        save_model(path, replacement)
    restored = load_model(path, distance_model=replacement.distance_model)
    np.testing.assert_array_equal(restored.predict(region).values, first.predict(region).values)
    assert set(tmp_path.iterdir()) == {path}
