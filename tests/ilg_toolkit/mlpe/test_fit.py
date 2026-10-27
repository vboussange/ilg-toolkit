"""Scientific behavior of mlpe.fit."""

from typing import Any

import numpy as np
import pytest

from ilg_toolkit import PairwiseObservations
from ilg_toolkit.mlpe import calibrate_mlpe


def test_calibration_matches_frozen_r_full_ml_and_predicts_without_query_targets(
    reference_observations, mlpe_reference
):
    observations = reference_observations()
    head = calibrate_mlpe(mlpe_reference.distances, observations, region_name="alpine")
    np.testing.assert_allclose(
        [head.intercept, head.slope], mlpe_reference.fixed, rtol=3e-5, atol=3e-6
    )
    np.testing.assert_allclose(
        [head.unit_variance, head.residual_variance], mlpe_reference.variances, rtol=2e-4, atol=2e-6
    )
    assert head.ml_log_likelihood == pytest.approx(mlpe_reference.log_likelihood, abs=2e-5)
    np.testing.assert_allclose(head.effect_mean, mlpe_reference.blups, rtol=2e-4, atol=2e-5)
    assert head.distance_center == pytest.approx(-0.14328369260805038, abs=1e-14)
    assert head.distance_scale == pytest.approx(1.012896471531856, abs=1e-14)
    query = head.predict([2.5, -2.0], [("new", "p01"), ("p02", "new")])
    # Independent frozen predictions from the R coefficients and distance moments.
    np.testing.assert_allclose(query.values, [-0.49407802616878604, 2.4619456646531583], atol=1e-5)
    assert query.values[0] < 0  # Signed calibration is not clipped or repaired.
    assert head.region_name == "alpine"
    assert head.calibration_pairs == observations.observed_pairs


def test_calibration_honors_the_declared_distance_scale_threshold(
    reference_observations, mlpe_reference
):
    from ilg_toolkit import MLPEError, MLPELoss

    distances = mlpe_reference.distances * 1e-14
    observations = reference_observations()
    with pytest.raises(MLPEError):
        calibrate_mlpe(distances, observations, region_name="alpine")
    head = calibrate_mlpe(
        distances, observations, region_name="alpine", loss=MLPELoss(min_distance_scale=1e-16)
    )
    np.testing.assert_allclose(
        [head.intercept, head.slope], mlpe_reference.fixed, rtol=3e-5, atol=3e-6
    )
    np.testing.assert_allclose(
        [head.unit_variance, head.residual_variance], mlpe_reference.variances, rtol=2e-4, atol=2e-6
    )


def test_calibration_rejects_unidentified_variances(reference_observations, mlpe_reference):
    from ilg_toolkit.mlpe import MLPEError

    observations = reference_observations()
    disjoint = PairwiseObservations.from_pairs(
        [("a", "b"), ("c", "d"), ("e", "f")], [1, 3, 2], target=observations.target
    )
    with pytest.raises(MLPEError):
        calibrate_mlpe([1, 2, 3], disjoint, region_name="alpine")
    with pytest.raises(MLPEError):
        calibrate_mlpe(np.ones(len(mlpe_reference.distances)), observations, region_name="alpine")


def test_explicit_numerical_constraints_and_optimization_failure_are_visible(
    reference_observations, mlpe_reference
):
    from ilg_toolkit import MLPEError, MLPELoss

    invalid_options: tuple[dict[str, Any], ...] = (
        {"variance_floor": 0},
        {"min_distance_scale": float("nan")},
        {"jitter": 1e-3},
    )
    for options in invalid_options:
        with pytest.raises(ValueError):
            MLPELoss(**options)
    with pytest.raises(MLPEError):
        calibrate_mlpe(
            mlpe_reference.distances,
            reference_observations(),
            region_name="alpine",
            max_iterations=1,
        )
