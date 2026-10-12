"""Inference artifacts preserve public predictions using a caller encoder template."""

from typing import final

import coordax as cx
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
    load_model,
    save_model,
)
from ilg_toolkit.models import UNetEmbeddingDistance


def problem(name="inference"):
    coordinates = np.array([0, 0.2, 0.8, 1.1, 1.9, 2.8])
    features = np.stack((coordinates, coordinates * 0.1 + 1), axis=-1).reshape(2, 3, 2)
    labels = tuple(f"population-{index}" for index in range(6))
    region = RegionBatch(
        name,
        cx.Field(jnp.asarray(features), dims=("row", "column", "feature")),
        labels,
        np.array(list(np.ndindex(2, 3))),
        feature_names=("east_covariate", "canopy_covariate"),
    )
    scores = (coordinates[:, None] - coordinates[None, :]) ** 2
    effects = np.array([0.2, -0.15, 0.1, 0.3, -0.2, 0.05])
    observed = 2 + 0.7 * scores + effects[:, None] + effects[None, :]
    noise = np.random.default_rng(17).normal(scale=0.12, size=(6, 6))
    observed += (noise + noise.T) / 2
    np.fill_diagonal(observed, 0)
    observations = PairwiseObservations.from_matrix(
        labels,
        observed,
        target=TargetSpec("synthetic divergence", units="index", transform="log1p"),
    )
    return region, observations


def test_direct_model_reload_uses_caller_encoder_and_preserves_predictions(tmp_path):
    region, observations = problem()

    def encoder(seed):
        return UNetEmbeddingDistance(
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
    np.testing.assert_array_equal(restored.landscape_scores(region), model.landscape_scores(region))
    np.testing.assert_array_equal(restored.predict(region).values, model.predict(region).values)
    assert restored.target == model.target
    assert restored.feature_names == model.feature_names
    assert restored.training_pairs == model.training_pairs
    assert restored.training_objective == "direct_log1p"
    assert isinstance(restored.distance_model, UNetEmbeddingDistance)
    assert restored.distance_model.encoder1.dropout.p == 0.23


def test_custom_mlpe_reload_preserves_regional_posteriors_and_conditioning(tmp_path):
    from dataclasses import replace

    import equinox as eqx

    from ilg_toolkit import ObservationPartition
    from ilg_toolkit.models import EmbeddingDistanceModel

    @final
    class ScaledEmbedding(EmbeddingDistanceModel):
        log_weights: jax.Array
        patch_size: int = eqx.field(static=True, default=1)

        def __init__(self, log_weights: jax.Array):
            self.log_weights = log_weights

        def embedding_grid(self, features, *, inference=True, key=None, patch_batch_size=None):
            return features * jnp.exp(self.log_weights)

    region, observations = problem()
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
        distance_model=ScaledEmbedding(jnp.asarray([0.2, -0.7])),
        config=TrainingConfig(epochs=0, objective="mlpe"),
        partition=(training, replace(training, region_name=other.name)),
        validation=(region, observations),
        validation_partition=validation,
    ).model
    path = tmp_path / "custom-mlpe.ilg"
    save_model(path, model)
    restored = load_model(path, distance_model=ScaledEmbedding(jnp.asarray([4.0, 3.0])))
    assert type(restored.distance_model) is ScaledEmbedding
    for prepared in (region, other):
        np.testing.assert_array_equal(
            restored.predict(prepared).values, model.predict(prepared).values
        )
        assert isinstance(model.calibration, MLPECalibrationModule)
        assert isinstance(restored.calibration, MLPECalibrationModule)
        assert eqx.tree_equal(
            restored.calibration.regions[prepared.name], model.calibration.regions[prepared.name]
        )
    assert restored.training_pairs == model.training_pairs
    assert restored.validation_pairs == model.validation_pairs
    query = (("population-4", "population-2"), ("population-5", "population-3"))
    before = model.predict_known_effects(region, query)
    after = restored.predict_known_effects(region, query)
    np.testing.assert_array_equal(after.values, before.values)
    np.testing.assert_array_equal(after.model_variance, before.model_variance)
    assert after.provenance == before.provenance
    support = PairwiseObservations.from_pairs(
        [("population-4", "population-0")], [1.5], target=model.target
    )
    declared = ObservationPartition(region.name, support.observed_pairs, role="support")
    before = model.predict_with_support(region, query, support, support_partition=declared)
    after = restored.predict_with_support(region, query, support, support_partition=declared)
    np.testing.assert_array_equal(after.values, before.values)
    np.testing.assert_array_equal(after.model_variance, before.model_variance)
    assert after.provenance == before.provenance
    assert isinstance(restored.calibration, MLPECalibrationModule)
    assert restored.calibration.regions[region.name].converged is False

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


@pytest.mark.parametrize(
    "saved_dtype,template_dtype",
    [("float64", "float32"), ("complex128", "complex64"), ("int64", "int32"), ("uint64", "uint32")],
)
def test_model_reload_refuses_jax_precision_truncation(tmp_path, saved_dtype, template_dtype):
    import equinox as eqx
    import pytest

    from ilg_toolkit import ArtifactError, LandscapeGeneticsModel
    from ilg_toolkit.models import EmbeddingDistanceModel

    @final
    class CustomEmbedding(EmbeddingDistanceModel):
        weight: jax.Array
        patch_size: int = eqx.field(static=True, default=1)

        def __init__(self, weight: jax.Array):
            self.weight = weight

        def embedding_grid(self, features, *, inference=True, key=None, patch_batch_size=None):
            return features * self.weight

    path = tmp_path / "precision.ilg"
    with jax.enable_x64():
        encoder = CustomEmbedding(jnp.asarray([1.1234567890123456], dtype=saved_dtype))
        model = LandscapeGeneticsModel(
            encoder,
            DirectCalibrationModule(TargetSpec("divergence")),
            2,
            training_objective="direct_log1p",
        )
        save_model(path, model)
        restored = load_model(path, distance_model=CustomEmbedding(jnp.zeros(1, dtype=saved_dtype)))
        assert isinstance(restored.distance_model, CustomEmbedding)
        np.testing.assert_array_equal(restored.distance_model.weight, encoder.weight)
    with jax.enable_x64(False):
        template = CustomEmbedding(jnp.zeros(1, dtype=template_dtype))
        with pytest.raises(ArtifactError, match="JAX_ENABLE_X64"):
            load_model(path, distance_model=template)


def test_conductance_reload_preserves_surface_scores_and_solver_settings(tmp_path):
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
            patch_batch_size=3,
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
        restored.conductance_surface(region), model.conductance_surface(region)
    )
    np.testing.assert_array_equal(restored.landscape_scores(region), model.landscape_scores(region))
    np.testing.assert_array_equal(restored.predict(region).values, model.predict(region).values)
    assert isinstance(restored.distance_model, ResNet9Conductance)
    assert isinstance(model.distance_model, ResNet9Conductance)
    assert restored.distance_model.solver_config == model.distance_model.solver_config
    assert restored.distance_model.patch_batch_size == 3
    assert restored.distance_model.min_conductance == 0.02


def test_model_reload_uses_native_shape_and_dtype_compatibility(tmp_path):
    from ilg_toolkit import ArtifactError, LandscapeGeneticsModel

    encoder = UNetEmbeddingDistance(
        2, patch_size=1, base_channels=2, embedding_dim=2, key=jax.random.key(0)
    )
    model = LandscapeGeneticsModel(
        encoder,
        DirectCalibrationModule(TargetSpec("divergence")),
        2,
        training_objective="direct_log1p",
    )
    path = tmp_path / "model.ilg"
    save_model(path, model)
    wrong_shape = UNetEmbeddingDistance(
        2, patch_size=1, base_channels=2, embedding_dim=3, key=jax.random.key(0)
    )
    with pytest.raises(ArtifactError, match="changed shape"):
        load_model(path, distance_model=wrong_shape)
    wrong_dtype = jax.tree.map(
        lambda value: value.astype(jnp.float64) if isinstance(value, jax.Array) else value,
        encoder,
    )
    with pytest.raises(ArtifactError, match="changed dtype"):
        load_model(path, distance_model=wrong_dtype)


def test_model_reload_rejects_truncated_leaves_and_legacy_archives(tmp_path):
    import json
    import zipfile

    from ilg_toolkit import ArtifactError, LandscapeGeneticsModel

    encoder = UNetEmbeddingDistance(
        2, patch_size=1, base_channels=2, embedding_dim=2, key=jax.random.key(0)
    )
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
    with pytest.raises(ArtifactError, match="supplied distance_model template"):
        load_model(truncated, distance_model=encoder)
    legacy = tmp_path / "legacy.ilg"
    with zipfile.ZipFile(legacy, "w") as archive:
        archive.writestr("manifest.json", '{"format":"ilg-toolkit","schema":1}')
    with pytest.raises(ArtifactError, match="Unsupported"):
        load_model(legacy, distance_model=encoder)
    header, leaves = path.read_bytes().split(b"\n", 1)
    outdated_header = json.loads(header)
    outdated_header["version"] = 2
    old_native = tmp_path / "old-native.ilg"
    old_native.write_bytes(json.dumps(outdated_header).encode() + b"\n" + leaves)
    with pytest.raises(ArtifactError, match="Unsupported"):
        load_model(old_native, distance_model=encoder)


def test_interrupted_model_replacement_preserves_previous_predictions(tmp_path, monkeypatch):
    import os

    from ilg_toolkit import LandscapeGeneticsModel

    region, _ = problem()

    def model(seed):
        return LandscapeGeneticsModel(
            UNetEmbeddingDistance(
                2, patch_size=1, base_channels=2, embedding_dim=2, key=jax.random.key(seed)
            ),
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
    with pytest.raises(OSError, match="interrupted replacement"):
        save_model(path, replacement)
    restored = load_model(path, distance_model=replacement.distance_model)
    np.testing.assert_array_equal(restored.predict(region).values, first.predict(region).values)
    assert set(tmp_path.iterdir()) == {path}
