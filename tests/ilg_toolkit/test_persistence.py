"""Scientific behavior of persistence."""

from typing import cast

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ilg_toolkit import (
    DirectCalibrationModule,
    MLPECalibrationModule,
    PairwiseObservations,
    RegionBatch,
    TargetSpec,
    TrainingConfig,
    fit,
    load_checkpoint,
    load_model,
    save_checkpoint,
    save_model,
)
from ilg_toolkit.models import ConductanceModel, UNet


@pytest.mark.parametrize("kind", ["model", "checkpoint"])
@pytest.mark.parametrize("objective", ["direct_log1p", "mlpe"])
def test_saved_precision_is_preserved(tmp_path, training_problem, kind, objective):
    from ilg_toolkit import ArtifactError

    path = tmp_path / "precision.eqx"
    with jax.enable_x64():
        region, observations, encoder = training_problem("embedding")
        state = fit(
            region,
            observations,
            distance_model=encoder,
            config=TrainingConfig(objective=objective, epochs=0),
        )
        if kind == "model":
            save_model(path, state.model)
            model = load_model(path, distance_model=encoder)
            assert eqx.tree_equal(model, state.model)
        else:
            save_checkpoint(path, state)
            restored = load_checkpoint(path, distance_model=encoder)
            assert eqx.tree_equal(restored, state)
            model = restored.model
        expected = state.model.predict(region).values
        np.testing.assert_array_equal(model.predict(region).values, expected)
        assert expected.dtype == jnp.float64
    with jax.enable_x64(False):
        narrow = jax.tree.map(
            lambda leaf: leaf.astype(jnp.float32) if eqx.is_inexact_array(leaf) else leaf,
            encoder,
        )
        with pytest.raises(ArtifactError):
            load = load_model if kind == "model" else load_checkpoint
            load(path, distance_model=narrow)
        assert not jax.config.read("jax_enable_x64")


def test_direct_model_reload_uses_caller_encoder_and_preserves_predictions(
    persistence_problem, tmp_path
):
    region, observations = persistence_problem()

    def encoder(seed):
        return UNet(
            2,
            patch_size=1,
            base_channels=2,
            embedding_dim=3,
            dropout=0.23,
            key=jax.random.key(seed),
        )

    model = fit(
        region,
        observations,
        distance_model=encoder(3),
        config=TrainingConfig(epochs=0, objective="direct_log1p"),
    ).model
    path = tmp_path / "direct.ilg"
    save_model(path, model)
    restored = load_model(path, distance_model=encoder(7))
    np.testing.assert_array_equal(
        restored.distance_model.predict(region).as_matrix(),
        model.distance_model.predict(region).as_matrix(),
    )
    np.testing.assert_array_equal(restored.predict(region).values, model.predict(region).values)
    assert restored.target == model.target
    assert restored.feature_names == model.feature_names
    assert restored.training_pairs == model.training_pairs
    assert restored.training_objective == "direct_log1p"
    assert cast(UNet, restored.distance_model).encoder1.dropout.p == 0.23


def test_custom_mlpe_reload_preserves_regional_posteriors_and_conditioning(
    persistence_problem, WeightedEmbedding, tmp_path
):
    from dataclasses import replace

    from ilg_toolkit import ObservationPartition

    region, observations = persistence_problem()
    other = replace(region, name="second-calibration")
    other_observations = PairwiseObservations.from_matrix(
        observations.sampling_unit_ids, observations.values * 2.0, target=observations.target
    )
    labels = set(region.sampling_unit_ids[:4])
    pairs = tuple(pair for pair in observations.observed_pairs if set(pair) <= labels)
    training = ObservationPartition(region.name, pairs[:-1], role="training")
    validation = ObservationPartition(region.name, pairs[-1:], role="validation")
    model = fit(
        (region, other),
        (observations, other_observations),
        distance_model=WeightedEmbedding(jnp.asarray([0.2, -0.7])),
        config=TrainingConfig(epochs=0, objective="mlpe"),
        partition=(training, replace(training, region_name=other.name)),
        validation=(region, observations),
        validation_partition=validation,
    ).model
    path = tmp_path / "custom-mlpe.ilg"
    save_model(path, model)
    restored = load_model(path, distance_model=WeightedEmbedding(jnp.asarray([4.0, 3.0])))
    assert type(restored.distance_model) is WeightedEmbedding
    for prepared in (region, other):
        np.testing.assert_array_equal(
            restored.predict(prepared).values, model.predict(prepared).values
        )
        assert eqx.tree_equal(
            cast(MLPECalibrationModule, restored.calibration).regions[prepared.name],
            cast(MLPECalibrationModule, model.calibration).regions[prepared.name],
        )
    assert restored.training_pairs == model.training_pairs
    assert restored.validation_pairs == model.validation_pairs
    query = (("population-4", "population-2"), ("population-5", "population-3"))
    before = model.predict_known_effects(region, query)
    after = restored.predict_known_effects(region, query)
    np.testing.assert_array_equal(after.prediction.values, before.prediction.values)
    np.testing.assert_array_equal(after.model_variance, before.model_variance)
    assert after.provenance == before.provenance
    support = PairwiseObservations.from_pairs(
        [("population-4", "population-0")], [1.5], target=model.target
    )
    declared = ObservationPartition(region.name, support.observed_pairs, role="support")
    before = model.predict_with_support(region, query, support, support_partition=declared)
    after = restored.predict_with_support(region, query, support, support_partition=declared)
    np.testing.assert_array_equal(after.prediction.values, before.prediction.values)
    np.testing.assert_array_equal(after.model_variance, before.model_variance)
    assert after.provenance == before.provenance
    assert cast(MLPECalibrationModule, restored.calibration).regions[region.name].converged is False

    from ilg_toolkit import (
        Ensemble,
        EnsembleMember,
        MemberIdentity,
        PopulationFold,
        predict_out_of_fold,
    )

    fold = PopulationFold(
        "held-out",
        {region.name: region.sampling_unit_ids[-2:]},
        {region.name: training},
        validation={region.name: validation},
    )
    evaluation_pairs = (
        validation.pairs[0],
        ("population-2", "population-4"),
        ("population-4", "population-5"),
    )

    def evaluate(calibrated):
        member = EnsembleMember(MemberIdentity(fold.fold_id, 3), fold, "completed", calibrated)
        return predict_out_of_fold(Ensemble((member,)), region, evaluation_pairs)

    before_evaluation, after_evaluation = evaluate(model), evaluate(restored)
    np.testing.assert_array_equal(after_evaluation.eligible_counts, [0, 0, 1])
    np.testing.assert_array_equal(after_evaluation.values, before_evaluation.values)
    assert after_evaluation.access == before_evaluation.access
    assert after_evaluation.exclusion_reasons == before_evaluation.exclusion_reasons


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
        training_objective="direct_log1p",
    )
    path = tmp_path / "conductance.ilg"
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

    encoder = UNet(2, patch_size=1, base_channels=2, embedding_dim=2, key=jax.random.key(0))
    model = LandscapeGeneticsModel(
        encoder,
        DirectCalibrationModule(TargetSpec("divergence")),
        2,
        training_objective="direct_log1p",
    )
    path = tmp_path / "model.ilg"
    save_model(path, model)
    wrong_shape = UNet(2, patch_size=1, base_channels=2, embedding_dim=3, key=jax.random.key(0))
    with pytest.raises(ArtifactError):
        load_model(path, distance_model=wrong_shape)
    wrong_dtype = jax.tree.map(
        lambda value: value.astype(jnp.float64) if isinstance(value, jax.Array) else value,
        encoder,
    )
    with pytest.raises(ArtifactError):
        load_model(path, distance_model=wrong_dtype)


def test_model_reload_rejects_truncated_leaves(tmp_path):
    from ilg_toolkit import ArtifactError, LandscapeGeneticsModel

    encoder = UNet(2, patch_size=1, base_channels=2, embedding_dim=2, key=jax.random.key(0))
    model = LandscapeGeneticsModel(
        encoder,
        DirectCalibrationModule(TargetSpec("divergence")),
        2,
        training_objective="direct_log1p",
    )
    path = tmp_path / "model.ilg"
    save_model(path, model)
    truncated = tmp_path / "truncated.ilg"
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
            UNet(2, patch_size=1, base_channels=2, embedding_dim=2, key=jax.random.key(seed)),
            DirectCalibrationModule(TargetSpec("divergence")),
            2,
            feature_names=region.feature_names,
            training_objective="direct_log1p",
        )

    first, replacement = model(3), model(7)
    path = tmp_path / "model.ilg"
    save_model(path, first)

    def interrupt(source, destination):
        raise OSError("interrupted replacement")

    monkeypatch.setattr(os, "replace", interrupt)
    with pytest.raises(OSError):
        save_model(path, replacement)
    restored = load_model(path, distance_model=replacement.distance_model)
    np.testing.assert_array_equal(restored.predict(region).values, first.predict(region).values)
    assert set(tmp_path.iterdir()) == {path}
