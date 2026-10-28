"""MLPE calibrations fitted by calibration heads, against the frozen R reference."""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from ilg_toolkit import fit_calibration_head, MLPECalibrationHead, MLPELoss, PairwiseObservations
from ilg_toolkit.mlpe import MLPECalibration, MLPEError


def _fit(mlpe_reference, observations, distances, **options) -> MLPECalibration:
    region, distance_model = mlpe_reference.problem(observations.observed_pairs, distances)
    head = fit_calibration_head(region, observations, distance_model, **options)
    assert isinstance(head, MLPECalibrationHead)
    return head.for_region(region.name)


def test_calibration_matches_frozen_r_full_ml_and_predicts_without_query_targets(
    reference_observations, mlpe_reference
):
    observations = reference_observations()
    shift = mlpe_reference.shift
    calibration = _fit(
        mlpe_reference, observations, mlpe_reference.distances + shift, loss=MLPELoss()
    )
    np.testing.assert_allclose(
        [calibration.intercept, calibration.slope], mlpe_reference.fixed, rtol=3e-5, atol=3e-6
    )
    np.testing.assert_allclose(
        [calibration.unit_variance, calibration.residual_variance],
        mlpe_reference.variances,
        rtol=2e-4,
        atol=2e-6,
    )
    assert calibration.ml_log_likelihood == pytest.approx(mlpe_reference.log_likelihood, abs=2e-5)
    np.testing.assert_allclose(calibration.effect_mean, mlpe_reference.blups, rtol=2e-4, atol=2e-5)
    assert calibration.distance_center == pytest.approx(-0.14328369260805038 + shift, abs=1e-14)
    assert calibration.distance_scale == pytest.approx(1.012896471531856, abs=1e-14)
    assert calibration.calibration_pairs == observations.observed_pairs
    head = MLPECalibrationHead(observations.target, {("alpine", None): calibration})
    query = head.predict(
        np.array([2.5, -2.0]) + shift, [("new", "p01"), ("p02", "new")], region_name="alpine"
    )
    # Independent frozen predictions from the R coefficients and distance moments.
    np.testing.assert_allclose(query.values, [-0.49407802616878604, 2.4619456646531583], atol=1e-5)
    assert query.values[0] < 0  # Signed calibration is not clipped or repaired.


def test_calibration_honors_the_declared_distance_scale_threshold(
    reference_observations, mlpe_reference
):
    distances = (mlpe_reference.distances + mlpe_reference.shift) * 1e-14
    observations = reference_observations()
    with pytest.raises(MLPEError):
        _fit(mlpe_reference, observations, distances, loss=MLPELoss())
    calibration = _fit(
        mlpe_reference, observations, distances, loss=MLPELoss(min_distance_scale=1e-16)
    )
    np.testing.assert_allclose(
        [calibration.intercept, calibration.slope], mlpe_reference.fixed, rtol=3e-5, atol=3e-6
    )
    np.testing.assert_allclose(
        [calibration.unit_variance, calibration.residual_variance],
        mlpe_reference.variances,
        rtol=2e-4,
        atol=2e-6,
    )


def test_calibration_rejects_unidentified_variances(reference_observations, mlpe_reference):
    observations = reference_observations()
    disjoint = PairwiseObservations.from_pairs(
        [("a", "b"), ("c", "d"), ("e", "f")], [1, 3, 2], target=observations.target
    )
    with pytest.raises(ValueError, match="unidentifiable"):
        _fit(mlpe_reference, disjoint, [1, 2, 3], loss=MLPELoss())
    with pytest.raises(MLPEError):
        _fit(mlpe_reference, observations, np.ones(len(mlpe_reference.distances)), loss=MLPELoss())


def test_explicit_numerical_constraints_and_optimization_failure_are_visible(
    reference_observations, mlpe_reference
):
    invalid_options: tuple[dict[str, Any], ...] = (
        {"variance_floor": 0},
        {"min_distance_scale": float("nan")},
        {"jitter": 1e-3},
    )
    for options in invalid_options:
        with pytest.raises(ValueError):
            MLPELoss(**options)
    distances = mlpe_reference.distances + mlpe_reference.shift
    for max_iterations in (1, 0, 2.5):
        with pytest.raises(MLPEError):
            _fit(
                mlpe_reference,
                reference_observations(),
                distances,
                loss=MLPELoss(),
                max_iterations=max_iterations,
            )


def test_mlpe_calibrations_reject_invalid_fitted_values(calibrated):
    head, _ = calibrated
    calibration = head.for_region("alpine")
    count = len(calibration.population_ids)
    for changes in (
        {"slope": float("nan")},
        {"distance_scale": 0.0},
        {"unit_variance": 0.0},
        {"effect_mean": np.zeros(count + 1)},
        {"effect_covariance": np.full((count, count), np.inf)},
        {"population_ids": ("p01",) * count},
    ):
        with pytest.raises(MLPEError):
            replace(calibration, **changes)
