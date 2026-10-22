"""Scientific behavior of calibration."""

from dataclasses import replace
from typing import cast

import coordax as cx
import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ilg_toolkit import (
    MLPECalibrationModule,
    ObservationPartition,
    PairwiseObservations,
    TargetSpec,
    TrainingConfig,
    fit,
)


def test_explicit_development_recalibration_keeps_encoder_and_prior_history_frozen(
    ScalarEmbedding, calibrated, calibration_problem
):
    from ilg_toolkit import calibrate

    region, observations, training, validation = calibration_problem()
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
    assert calibrated.training_objective == "direct_log1p"
    assert eqx.tree_equal(calibrated.distance_model, result.model.distance_model)
    assert (
        cast(MLPECalibrationModule, calibrated.calibration).regions[region.name].calibration_pairs
        == training.pairs
    )
    assert set(
        cast(MLPECalibrationModule, calibrated.calibration).regions[region.name].calibration_roles
    ) == {"training"}
    developed = calibrate(calibrated, region, observations, partitions=(training, validation))
    assert eqx.tree_equal(developed.distance_model, calibrated.distance_model)
    assert (
        cast(MLPECalibrationModule, developed.calibration).regions[region.name].config
        is cast(MLPECalibrationModule, calibrated.calibration).regions[region.name].config
    )
    assert result.history is history
    assert result.model.training_objective == "direct_log1p"
    assert (
        len(
            cast(MLPECalibrationModule, developed.calibration)
            .regions[region.name]
            .calibration_pairs
        )
        == 15
    )
    assert set(
        cast(MLPECalibrationModule, developed.calibration).regions[region.name].calibration_roles
    ) == {
        "training",
        "validation",
    }
    assert not np.allclose(calibrated.predict(region).values, developed.predict(region).values)
    np.testing.assert_array_equal(
        developed.ecological_distances(region), calibrated.ecological_distances(region)
    )


def test_regional_mlpe_installation_requires_matching_region_and_target_contracts(
    ScalarEmbedding, calibration_problem
):
    import pytest

    from ilg_toolkit import MLPEError, RegionalMLPECalibration

    region, observations, training, _ = calibration_problem()
    model = fit(
        region,
        observations,
        distance_model=ScalarEmbedding(jnp.asarray(1.0)),
        config=TrainingConfig(objective="mlpe", epochs=0),
        partition=training,
    ).model
    regional = cast(MLPECalibrationModule, model.calibration).for_region(region.name)
    assert cast(RegionalMLPECalibration, regional).target == model.target == observations.target
    with pytest.raises(ValueError):
        MLPECalibrationModule(
            model.target, {"wrong-region": cast(RegionalMLPECalibration, regional)}
        )
    with pytest.raises(ValueError):
        MLPECalibrationModule(
            TargetSpec("another target"), {region.name: cast(RegionalMLPECalibration, regional)}
        )
    with pytest.raises(MLPEError):
        replace(
            cast(RegionalMLPECalibration, regional),
            target=TargetSpec("relatedness", kind="relatedness"),
        )


def test_transfer_distances_are_label_free_and_each_region_requires_its_own_head(
    ScalarEmbedding, calibrated, calibration_problem
):
    import pytest

    from ilg_toolkit import calibrate

    region, observations, training, _ = calibration_problem()
    original = fit(
        region,
        observations,
        distance_model=ScalarEmbedding(jnp.asarray(1.0)),
        config=TrainingConfig(objective="direct_log1p", epochs=0),
        partition=training,
    ).model
    calibrated = calibrate(original, region, observations)
    new_region = replace(region, name="unseen-catchment")
    distances = calibrated.ecological_distances(new_region)
    assert np.isfinite(distances).all()
    with pytest.raises(ValueError):
        calibrated.predict(new_region)
    with pytest.raises(ValueError):
        calibrate(calibrated, new_region, observations)
    new_observations = PairwiseObservations.from_matrix(
        observations.sampling_unit_ids, observations.values * 2, target=observations.target
    )
    development = ObservationPartition(
        new_region.name, observations.observed_pairs, role="calibration"
    )
    transferred = calibrated.calibrate(new_region, new_observations, partitions=development)
    assert eqx.tree_equal(transferred.distance_model, calibrated.distance_model)
    assert eqx.tree_equal(
        cast(MLPECalibrationModule, transferred.calibration).regions[region.name],
        cast(MLPECalibrationModule, calibrated.calibration).regions[region.name],
    )
    np.testing.assert_array_equal(
        transferred.predict(region).values, calibrated.predict(region).values
    )
    assert np.isfinite(transferred.predict(new_region).values).all()
    assert not np.allclose(
        transferred.predict(region).values, transferred.predict(new_region).values
    )
    assert new_region.name not in cast(MLPECalibrationModule, calibrated.calibration).regions


def test_default_recalibration_ignores_validation_targets_and_records_encoder_access(
    ScalarEmbedding, calibration_problem
):
    from ilg_toolkit import calibrate

    region, observations, training, validation = calibration_problem()
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


def test_recalibration_rejects_incompatible_features_roles_targets_and_individual_effects(
    ScalarEmbedding, calibrated, calibration_problem
):
    import pytest

    from ilg_toolkit import calibrate

    region, observations, training, _ = calibration_problem()
    model = fit(
        region,
        observations,
        distance_model=ScalarEmbedding(jnp.asarray(1.0)),
        config=TrainingConfig(objective="direct_log1p", epochs=0),
        partition=training,
    ).model
    assert region.feature_names is not None
    with pytest.raises(ValueError):
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
    with pytest.raises(ValueError):
        calibrate(model, region, incompatible)
    for role in ("query", "support"):
        with pytest.raises(ValueError):
            calibrate(
                model,
                region,
                observations,
                partitions=ObservationPartition(region.name, training.pairs, role=role),
            )
    with pytest.raises(ValueError):
        calibrate(model, region, observations, partitions=(training, training))
    individuals = replace(region, sampling_unit_kinds=("individual",) + ("population",) * 5)
    with pytest.raises(ValueError):
        calibrate(model, individuals, observations)
    calibrated = calibrate(model, region, observations)
    assert np.isfinite(calibrated.ecological_distances(individuals)).all()
    with pytest.raises(ValueError):
        calibrated.predict(individuals)
