"""Frozen model recalibration and regional transfer at the public seam."""

from dataclasses import replace
from typing import final

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ilg_toolkit import (
    DirectCalibrationModule,
    MLPECalibrationModule,
    ObservationPartition,
    PairwiseObservations,
    RegionBatch,
    TargetSpec,
    TrainingConfig,
    fit,
)
from ilg_toolkit.models import EmbeddingDistanceModel


@final
class ScalarEmbedding(EmbeddingDistanceModel):
    weight: jax.Array
    patch_size: int = eqx.field(static=True, default=1)

    def __init__(self, weight: jax.Array):
        self.weight = weight

    def embedding_grid(self, features, *, inference=True, key=None, patch_batch_size=None):
        return features[..., :1] * self.weight


def problem(
    name: str = "original",
) -> tuple[RegionBatch, PairwiseObservations, ObservationPartition, ObservationPartition]:
    coordinates = np.array([0, 0.2, 0.8, 1.1, 1.9, 2.8])
    features = np.stack((coordinates, coordinates * 0.1 + 1), axis=-1).reshape(2, 3, 2)
    ids = tuple(f"population-{index}" for index in range(6))
    region = RegionBatch(
        name,
        cx.field(features, "row", "column", "feature"),
        ids,
        np.array(list(np.ndindex(2, 3))),
        feature_names=("east_covariate", "canopy_covariate"),
    )
    scores = (coordinates[:, None] - coordinates[None, :]) ** 2
    effects = np.array([0.2, -0.15, 0.1, 0.3, -0.2, 0.05])
    rng = np.random.default_rng(17)
    observed = 2 + 0.7 * scores + effects[:, None] + effects[None, :]
    noise = rng.normal(scale=0.12, size=(6, 6))
    observed += (noise + noise.T) / 2
    np.fill_diagonal(observed, 0)
    observations = PairwiseObservations.from_matrix(
        ids, observed, target=TargetSpec("synthetic divergence", units="index")
    )
    training = ObservationPartition(name, observations.observed_pairs[:10], role="training")
    validation = ObservationPartition(name, observations.observed_pairs[10:], role="validation")
    return region, observations, training, validation


def test_explicit_development_recalibration_keeps_encoder_and_prior_history_frozen():
    from ilg_toolkit import calibrate

    region, observations, training, validation = problem()
    result = fit(
        region,
        observations,
        distance_model=ScalarEmbedding(jnp.asarray(1.0)),
        config=TrainingConfig(objective="direct_log1p", epochs=0),
        partition=training,
        validation=(region, observations),
        validation_partition=validation,
    )
    history = result.history
    from ilg_toolkit import MLPEConfig

    calibrated = calibrate(result.model, region, observations, config=MLPEConfig(jitter=1e-7))
    assert isinstance(calibrated.calibration, MLPECalibrationModule)
    assert calibrated.training_objective == "direct_log1p"
    assert eqx.tree_equal(calibrated.distance_model, result.model.distance_model)
    assert isinstance(calibrated.calibration, MLPECalibrationModule)
    assert calibrated.calibration.regions[region.name].calibration_pairs == training.pairs
    assert isinstance(calibrated.calibration, MLPECalibrationModule)
    assert set(calibrated.calibration.regions[region.name].calibration_roles) == {"training"}
    developed = calibrate(calibrated, region, observations, partitions=(training, validation))
    assert eqx.tree_equal(developed.distance_model, calibrated.distance_model)
    assert isinstance(calibrated.calibration, MLPECalibrationModule)
    assert isinstance(developed.calibration, MLPECalibrationModule)
    assert (
        developed.calibration.regions[region.name].config
        is calibrated.calibration.regions[region.name].config
    )
    assert result.history is history
    assert result.model.training_objective == "direct_log1p"
    assert isinstance(result.model.calibration, DirectCalibrationModule)
    assert isinstance(developed.calibration, MLPECalibrationModule)
    assert len(developed.calibration.regions[region.name].calibration_pairs) == 15
    assert isinstance(developed.calibration, MLPECalibrationModule)
    assert set(developed.calibration.regions[region.name].calibration_roles) == {
        "training",
        "validation",
    }
    assert not np.allclose(calibrated.predict(region).values, developed.predict(region).values)
    np.testing.assert_array_equal(
        developed.landscape_scores(region), calibrated.landscape_scores(region)
    )


def test_composed_calibration_owns_inference_and_preserves_training_objective(tmp_path):
    import pytest

    from ilg_toolkit import (
        DirectCalibrationModule,
        LandscapeGeneticsModel,
        MLPECalibrationModule,
        calibrate,
        load_model,
        save_model,
    )

    region, observations, training, _ = problem()
    state = fit(
        region,
        observations,
        distance_model=ScalarEmbedding(jnp.asarray(1.0)),
        config=TrainingConfig(objective="direct_log1p", epochs=0),
        partition=training,
    )
    assert isinstance(state.model, LandscapeGeneticsModel)
    assert isinstance(state.model.calibration, DirectCalibrationModule)
    assert state.model.target is state.model.calibration.target
    converted = calibrate(state.model, region, observations)
    assert isinstance(converted.calibration, MLPECalibrationModule)
    assert converted.training_objective == state.config.objective == "direct_log1p"
    assert eqx.tree_equal(converted.distance_model, state.model.distance_model)
    unseen = replace(region, name="unseen")
    assert np.isfinite(state.model.predict(unseen).values).all()
    with pytest.raises(ValueError, match="unseen.*no MLPE calibration"):
        converted.predict(unseen)
    path = tmp_path / "converted.ilg"
    save_model(path, converted)
    restored = load_model(path, distance_model=ScalarEmbedding(jnp.asarray(0.0)))
    assert restored.training_objective == "direct_log1p"
    assert restored.target == observations.target
    np.testing.assert_array_equal(restored.predict(region).values, converted.predict(region).values)


def test_regional_mlpe_installation_requires_matching_region_and_target_contracts():
    import pytest

    from ilg_toolkit import MLPEError, RegionalMLPECalibration

    region, observations, training, _ = problem()
    model = fit(
        region,
        observations,
        distance_model=ScalarEmbedding(jnp.asarray(1.0)),
        config=TrainingConfig(objective="mlpe", epochs=0),
        partition=training,
    ).model
    assert isinstance(model.calibration, MLPECalibrationModule)
    regional = model.calibration.for_region(region.name)
    assert isinstance(regional, RegionalMLPECalibration)
    assert regional.target == model.target == observations.target
    with pytest.raises(ValueError, match="names and target contracts"):
        MLPECalibrationModule(model.target, {"wrong-region": regional})
    with pytest.raises(ValueError, match="names and target contracts"):
        MLPECalibrationModule(TargetSpec("another target"), {region.name: regional})
    with pytest.raises(MLPEError, match="dissimilarity TargetSpec"):
        replace(regional, target=TargetSpec("relatedness", kind="relatedness"))


def test_transfer_scores_are_label_free_and_each_region_requires_its_own_head():
    import pytest

    from ilg_toolkit import calibrate

    region, observations, training, _ = problem()
    original = fit(
        region,
        observations,
        distance_model=ScalarEmbedding(jnp.asarray(1.0)),
        config=TrainingConfig(objective="direct_log1p", epochs=0),
        partition=training,
    ).model
    calibrated = calibrate(original, region, observations)
    new_region = replace(region, name="unseen-catchment")
    scores = calibrated.landscape_scores(new_region)
    assert np.isfinite(scores).all()
    with pytest.raises(ValueError, match="unseen-catchment.*no MLPE calibration"):
        calibrated.predict(new_region)
    with pytest.raises(ValueError, match="declare a calibration partition"):
        calibrate(calibrated, new_region, observations)
    new_observations = PairwiseObservations.from_matrix(
        observations.sampling_unit_ids, observations.values * 2, target=observations.target
    )
    development = ObservationPartition(
        new_region.name, observations.observed_pairs, role="calibration"
    )
    transferred = calibrated.calibrate(new_region, new_observations, partitions=development)
    assert eqx.tree_equal(transferred.distance_model, calibrated.distance_model)
    assert isinstance(calibrated.calibration, MLPECalibrationModule)
    assert isinstance(transferred.calibration, MLPECalibrationModule)
    assert eqx.tree_equal(
        transferred.calibration.regions[region.name], calibrated.calibration.regions[region.name]
    )
    np.testing.assert_array_equal(
        transferred.predict(region).values, calibrated.predict(region).values
    )
    assert np.isfinite(transferred.predict(new_region).values).all()
    assert not np.allclose(
        transferred.predict(region).values, transferred.predict(new_region).values
    )
    assert isinstance(calibrated.calibration, MLPECalibrationModule)
    assert new_region.name not in calibrated.calibration.regions


def test_default_recalibration_ignores_validation_targets_and_records_encoder_access():
    from ilg_toolkit import calibrate

    region, observations, training, validation = problem()
    model = fit(
        region,
        observations,
        distance_model=ScalarEmbedding(jnp.asarray(1.0)),
        config=TrainingConfig(objective="direct_log1p", epochs=0),
        partition=training,
        validation=(region, observations),
        validation_partition=validation,
    ).model
    assert model.training_pairs[region.name] == training.pairs
    assert model.validation_pairs[region.name] == validation.pairs
    altered = np.array(observations.values)
    lookup = {label: index for index, label in enumerate(region.sampling_unit_ids)}
    for a, b in validation.pairs:
        altered[lookup[a], lookup[b]] = altered[lookup[b], lookup[a]] = 100
    altered = PairwiseObservations.from_matrix(
        region.sampling_unit_ids, altered, target=observations.target
    )
    original = calibrate(model, region, observations)
    changed = calibrate(model, region, altered)
    np.testing.assert_array_equal(original.predict(region).values, changed.predict(region).values)


def test_recalibration_rejects_incompatible_features_roles_targets_and_individual_effects():
    import pytest

    from ilg_toolkit import calibrate

    region, observations, training, _ = problem()
    model = fit(
        region,
        observations,
        distance_model=ScalarEmbedding(jnp.asarray(1.0)),
        config=TrainingConfig(objective="direct_log1p", epochs=0),
        partition=training,
    ).model
    assert region.feature_names is not None
    with pytest.raises(ValueError, match="feature contract"):
        calibrate(
            model,
            replace(
                region,
                features=cx.field(region.feature_array, "row", "column", "feature"),
                feature_names=tuple(reversed(region.feature_names)),
            ),
            observations,
        )
    incompatible = PairwiseObservations.from_matrix(
        region.sampling_unit_ids,
        observations.values,
        target=TargetSpec("another target", units="fraction"),
    )
    with pytest.raises(ValueError, match="target meaning"):
        calibrate(model, region, incompatible)
    for role in ("query", "support"):
        with pytest.raises(ValueError, match="query or support"):
            calibrate(
                model,
                region,
                observations,
                partitions=ObservationPartition(region.name, training.pairs, role=role),
            )
    with pytest.raises(ValueError, match="overlap"):
        calibrate(model, region, observations, partitions=(training, training))
    individuals = replace(region, sampling_unit_kinds=("individual",) + ("population",) * 5)
    with pytest.raises(ValueError, match="population sampling units"):
        calibrate(model, individuals, observations)
    calibrated = calibrate(model, region, observations)
    assert np.isfinite(calibrated.landscape_scores(individuals)).all()
    with pytest.raises(ValueError, match="population sampling units"):
        calibrated.predict(individuals)
