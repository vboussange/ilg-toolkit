"""Complete inference reload through the public fitting/prediction boundary."""

from dataclasses import replace

import jax
import numpy as np
import pytest
from test_recalibration import problem

from ilg_toolkit import FitConfig, fit
from ilg_toolkit.models import UNetEmbeddingDistance


def test_direct_predictor_roundtrip_preserves_prepared_contract_and_predictions(tmp_path):
    from ilg_toolkit import PairwiseObservations, load_predictor, save_predictor

    region, observations, _, _ = problem()
    observations = PairwiseObservations.from_matrix(
        observations.sampling_unit_ids,
        observations.values,
        target=replace(observations.target, transform="log1p"),
    )
    model = UNetEmbeddingDistance(
        2,
        patch_size=1,
        base_channels=2,
        embedding_dim=3,
        dropout=0.23,
        key=jax.random.key(3),
    )
    predictor = fit(region, observations, model=model, config=FitConfig(epochs=0)).predictor
    path = tmp_path / "direct.ilg"
    save_predictor(path, predictor)
    restored = load_predictor(path)
    np.testing.assert_array_equal(
        restored.landscape_scores(region), predictor.landscape_scores(region)
    )
    np.testing.assert_array_equal(restored.predict(region).values, predictor.predict(region).values)
    assert restored.target == predictor.target
    assert restored.feature_names == predictor.feature_names
    assert restored.training_pairs == predictor.training_pairs
    assert restored.encoder.encoder1.dropout.p == 0.23


def test_mlpe_reload_retains_regional_posteriors_and_support_predictions(tmp_path):
    from ilg_toolkit import (
        ObservationPartition,
        PairwiseObservations,
        load_predictor,
        save_predictor,
    )

    region, observations, _, _ = problem("mlpe-reload")
    # Reserve two populations for support and query, preserving genuine unseen effects.
    labels = set(region.sampling_unit_ids[:4])
    pairs = tuple(pair for pair in observations.observed_pairs if set(pair) <= labels)
    training = ObservationPartition(region.name, pairs, role="training")
    other = replace(region, name="second-calibration")
    other_observations = PairwiseObservations.from_matrix(
        observations.sampling_unit_ids, observations.values * 2.0, target=observations.target
    )
    with jax.enable_x64():
        model = UNetEmbeddingDistance(
            2,
            patch_size=1,
            base_channels=2,
            embedding_dim=3,
            dropout=0.2,
            key=jax.random.key(3),
        )
        predictor = fit(
            (region, other),
            (observations, other_observations),
            model=model,
            config=FitConfig(epochs=0, objective="mlpe"),
            partition=(training, ObservationPartition(other.name, pairs, role="training")),
        ).predictor
        path = tmp_path / "mlpe.ilg"
        save_predictor(path, predictor)
        restored = load_predictor(path)
        np.testing.assert_array_equal(
            restored.predict(region).values, predictor.predict(region).values
        )
        np.testing.assert_array_equal(
            restored.predict(other).values, predictor.predict(other).values
        )
        query = (("population-4", "population-2"), ("population-5", "population-3"))
        before = predictor.predict_known_effects(region, query)
        after = restored.predict_known_effects(region, query)
        np.testing.assert_array_equal(after.model_values, before.model_values)
        np.testing.assert_array_equal(after.model_variance, before.model_variance)
        assert after.provenance == before.provenance
        support = PairwiseObservations.from_pairs(
            [("population-4", "population-0")], [1.5], target=predictor.target
        )
        declared = ObservationPartition(region.name, support.observed_pairs, role="support")
        before = predictor.predict_with_support(region, query, support, support_partition=declared)
        after = restored.predict_with_support(region, query, support, support_partition=declared)
        np.testing.assert_array_equal(after.model_values, before.model_values)
        np.testing.assert_array_equal(after.model_variance, before.model_variance)
        assert after.provenance == before.provenance
        assert restored.calibrations[region.name].converged is False


def test_conductance_archive_retains_actual_graph_scores_and_surface(tmp_path):
    from ilg_toolkit import (
        Predictor,
        PreparedRegion,
        SolverConfig,
        TargetSpec,
        load_predictor,
        save_predictor,
    )
    from ilg_toolkit.models import ResNet9Conductance

    region = PreparedRegion(
        "graph-reload",
        np.random.default_rng(1).normal(size=(8, 8, 2)),
        ("west", "north", "east"),
        np.array([[0, 0], [0, 7], [7, 7]]),
    )
    with jax.enable_x64():
        predictor = Predictor(
            ResNet9Conductance(2, patch_size=4, min_conductance=0.02, key=jax.random.key(3)),
            TargetSpec("divergence"),
            feature_count=2,
            solver_config=SolverConfig(rtol=1e-9, atol=1e-10),
        )
        path = tmp_path / "conductance.ilg"
        save_predictor(path, predictor)
        restored = load_predictor(path)
        np.testing.assert_array_equal(
            restored.conductance_surface(region), predictor.conductance_surface(region)
        )
        np.testing.assert_allclose(
            restored.landscape_scores(region),
            predictor.landscape_scores(region),
            atol=1e-12,
        )
        assert restored.solver_config == predictor.solver_config


def test_reload_rejects_incomplete_incompatible_and_corrupt_content(tmp_path):
    import json
    import zipfile

    from ilg_toolkit import ArtifactError, Predictor, TargetSpec, load_predictor, save_predictor

    predictor = Predictor(
        UNetEmbeddingDistance(
            2, patch_size=1, base_channels=2, embedding_dim=2, key=jax.random.key(0)
        ),
        TargetSpec("synthetic"),
        feature_count=2,
    )
    path = tmp_path / "valid.ilg"
    save_predictor(path, predictor)
    with zipfile.ZipFile(path) as archive:
        original = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(original["manifest.json"])
    mutations = [
        lambda m: m.update(schema=999),
        lambda m: m.update(kind="training_checkpoint"),
        lambda m: m["payload"].pop("target"),
        lambda m: m["payload"].update(feature_count=3),
        lambda m: m["payload"].update(objective="mlpe"),  # no regional calibration
        lambda m: m["payload"]["encoder"].update(kind="arbitrary.module.Class"),
    ]
    for index, mutate in enumerate(mutations):
        modified = json.loads(json.dumps(manifest))
        mutate(modified)
        bad = tmp_path / f"invalid-{index}.ilg"
        with zipfile.ZipFile(bad, "w") as archive:
            for name, data in original.items():
                archive.writestr(name, json.dumps(modified) if name == "manifest.json" else data)
        with pytest.raises(ArtifactError):
            load_predictor(bad)
    for missing in (False, True):
        bad = tmp_path / f"bad-payload-{missing}.ilg"
        first = next(name for name in original if name.startswith("arrays/"))
        with zipfile.ZipFile(bad, "w") as archive:
            for name, data in original.items():
                if name == first:
                    if missing:
                        continue
                    data = data[:-1] + bytes([data[-1] ^ 1])
                archive.writestr(name, data)
        with pytest.raises(ArtifactError, match="missing|integrity"):
            load_predictor(bad)


def test_unsupported_custom_encoder_fails_without_a_pickle_fallback(tmp_path):
    import jax.numpy as jnp
    from test_recalibration import ScalarEmbedding

    from ilg_toolkit import ArtifactError, Predictor, TargetSpec, save_predictor

    predictor = Predictor(ScalarEmbedding(jnp.array(1.0)), TargetSpec("divergence"), 2)
    with pytest.raises(ArtifactError, match="Unsupported custom model architecture"):
        save_predictor(tmp_path / "custom.ilg", predictor)


def test_model_precision_is_preserved_and_never_silently_truncated(tmp_path):
    import equinox as eqx
    import jax.numpy as jnp

    from ilg_toolkit import ArtifactError, Predictor, TargetSpec, load_predictor, save_predictor

    path = tmp_path / "float64.ilg"
    with jax.enable_x64():
        model = UNetEmbeddingDistance(
            2, patch_size=1, base_channels=2, embedding_dim=2, key=jax.random.key(0)
        )
        model = jax.tree.map(
            lambda value: value.astype(jnp.float64) if eqx.is_inexact_array(value) else value, model
        )
        predictor = Predictor(model, TargetSpec("divergence"), feature_count=2)
        save_predictor(path, predictor)
        restored = load_predictor(path)
        assert restored.encoder.patch_embedding.weight.dtype == np.float64
        region, _, _, _ = problem()
        # This hand-constructed predictor has no named feature contract.
        region = replace(region, feature_names=None)
        np.testing.assert_array_equal(
            restored.landscape_scores(region), predictor.landscape_scores(region)
        )
    with jax.enable_x64(False), pytest.raises(ArtifactError, match="JAX_ENABLE_X64"):
        load_predictor(path)


@pytest.mark.parametrize("shared_training", [False, True])
def test_partial_regional_calibration_roundtrip_retains_available_predictions(
    tmp_path, shared_training
):
    from ilg_toolkit import ObservationPartition, load_predictor, save_predictor

    original, observations, _, _ = problem("encoder-training")
    other = replace(original, name="second-region")
    model = UNetEmbeddingDistance(
        2,
        patch_size=1,
        base_channels=2,
        embedding_dim=3,
        dropout=0.2,
        key=jax.random.key(3),
    )
    if shared_training:
        fitted = fit(
            (original, other), (observations, observations), model=model, config=FitConfig(epochs=0)
        )
    else:
        fitted = fit(original, observations, model=model, config=FitConfig(epochs=0))
    declared = ObservationPartition(other.name, observations.observed_pairs, role="calibration")
    partial = fitted.predictor.recalibrate(other, observations, partitions=declared)
    assert set(partial.calibrations) == {other.name}
    before = partial.predict(other)
    path = tmp_path / "partial-calibration.ilg"
    save_predictor(path, partial)
    restored = load_predictor(path)
    assert restored.calibrations == partial.calibrations
    assert restored.training_pairs == fitted.predictor.training_pairs
    np.testing.assert_array_equal(restored.predict(other).values, before.values)
    np.testing.assert_array_equal(
        restored.landscape_scores(original), partial.landscape_scores(original)
    )
    with pytest.raises(ValueError, match="no MLPE calibration"):
        restored.predict(original)


def test_numpy_numeric_config_values_survive_inference_and_checkpoint_reload(tmp_path):
    from ilg_toolkit import (
        MLPEConfig,
        SolverConfig,
        load_checkpoint,
        load_predictor,
        save_checkpoint,
        save_predictor,
    )

    region, observations, _, _ = problem("numeric-config")
    config = FitConfig(
        epochs=0,
        learning_rate=np.float32(0.01),
        solver=SolverConfig(rtol=np.float32(1e-6), atol=np.float32(1e-7), max_steps=np.int64(200)),
        mlpe_variance_floor=np.float32(1e-10),
        mlpe_jitter=np.float32(1e-8),
        mlpe_initial_variances=(np.float32(0.05), np.float32(0.1)),
    )
    model = UNetEmbeddingDistance(
        2, patch_size=1, base_channels=2, embedding_dim=3, key=jax.random.key(3)
    )
    fitted = fit(region, observations, model=model, config=config)
    inference = tmp_path / "numpy-config-predictor.ilg"
    save_predictor(inference, fitted.predictor)
    restored = load_predictor(inference)
    assert restored.solver_config == config.solver
    np.testing.assert_array_equal(
        restored.predict(region).values, fitted.predictor.predict(region).values
    )
    checkpoint = tmp_path / "numpy-config-checkpoint.ilg"
    save_checkpoint(checkpoint, fitted.state)
    state = load_checkpoint(checkpoint)
    assert state.config == config
    assert state.config.learning_rate == float(np.float32(0.01))
    assert state.config.solver.rtol == float(np.float32(1e-6))
    calibrated = fitted.predictor.recalibrate(
        region,
        observations,
        config=MLPEConfig(
            variance_floor=np.float32(1e-10),
            min_score_scale=np.float32(1e-12),
            jitter=np.float32(1e-8),
        ),
    )
    calibration = tmp_path / "numpy-config-calibration.ilg"
    save_predictor(calibration, calibrated)
    reloaded = load_predictor(calibration)
    assert reloaded.calibrations == calibrated.calibrations
    np.testing.assert_array_equal(
        reloaded.predict(region).values, calibrated.predict(region).values
    )


@pytest.mark.parametrize("setting", ["learning_rate", "rtol", "variance_floor", "jitter"])
@pytest.mark.parametrize("value", [np.array(0.01), np.array([0.01]), object(), np.nan, np.inf])
def test_numeric_config_scalars_reject_arrays_objects_and_nonfinite_values(setting, value):
    from ilg_toolkit import MLPEConfig, SolverConfig

    config_type = {
        "learning_rate": FitConfig,
        "rtol": SolverConfig,
        "variance_floor": MLPEConfig,
        "jitter": MLPEConfig,
    }[setting]
    with pytest.raises(ValueError):
        config_type(**{setting: value})
