"""Scientific behavior of mlpe.fit."""

from typing import Any

import numpy as np
import pytest

from ilg_toolkit import PairwiseObservations, TargetSpec
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
    query = head.predict_marginal([2.5, -2.0], [("new", "p01"), ("p02", "new")])
    # Independent frozen predictions from the R coefficients and distance moments.
    np.testing.assert_allclose(
        query.prediction.values, [-0.49407802616878604, 2.4619456646531583], atol=1e-5
    )
    assert query.prediction.values[0] < 0  # Signed calibration is not clipped or repaired.
    assert head.region_name == "alpine"
    assert head.calibration_pairs == observations.observed_pairs
    assert set(head.calibration_roles) == {"calibration"}


def test_calibration_honors_the_declared_distance_scale_threshold(
    reference_observations, mlpe_reference
):
    from ilg_toolkit import MLPEConfig, MLPEError

    distances = mlpe_reference.distances * 1e-14
    observations = reference_observations()
    with pytest.raises(MLPEError):
        calibrate_mlpe(distances, observations, region_name="alpine")
    head = calibrate_mlpe(
        distances, observations, region_name="alpine", config=MLPEConfig(min_distance_scale=1e-16)
    )
    np.testing.assert_allclose(
        [head.intercept, head.slope], mlpe_reference.fixed, rtol=3e-5, atol=3e-6
    )
    np.testing.assert_allclose(
        [head.unit_variance, head.residual_variance], mlpe_reference.variances, rtol=2e-4, atol=2e-6
    )


def test_calibration_rejects_query_targets_and_unidentified_variances(
    reference_observations, mlpe_reference
):
    from ilg_toolkit import ObservationPartition
    from ilg_toolkit.mlpe import MLPEError

    observations = reference_observations()
    query = ObservationPartition("alpine", observations.observed_pairs, role="query")
    with pytest.raises(MLPEError):
        calibrate_mlpe(
            mlpe_reference.distances, observations, region_name="alpine", partition=query
        )
    disjoint = PairwiseObservations.from_pairs(
        [("a", "b"), ("c", "d"), ("e", "f")], [1, 3, 2], target=observations.target
    )
    with pytest.raises(MLPEError):
        calibrate_mlpe([1, 2, 3], disjoint, region_name="alpine")
    with pytest.raises(MLPEError):
        calibrate_mlpe(np.ones(len(mlpe_reference.distances)), observations, region_name="alpine")


def test_incomplete_partition_calibration_records_roles_and_original_target_scale(
    dense_profile, reference_observations, mlpe_reference
):

    from ilg_toolkit import MLPEConfig, ObservationPartition, calibrate_mlpe

    reference = reference_observations()
    target = TargetSpec("divergence", units="fraction", transform="log1p")
    observations = PairwiseObservations.from_pairs(
        reference.observed_pairs,
        np.expm1(mlpe_reference.targets + 1),
        target=target,
    )
    selected = tuple(
        pair for i, pair in enumerate(observations.observed_pairs) if i not in {2, 7, 11}
    )
    partition = ObservationPartition("river", selected, role="training")
    head = calibrate_mlpe(
        mlpe_reference.distances,
        observations,
        region_name="river",
        partition=partition,
        config=MLPEConfig(jitter=1e-7),
    )
    lookup = {label: i for i, label in enumerate(head.population_ids)}
    left = np.array([lookup[a] for a, b in selected])
    right = np.array([lookup[b] for a, b in selected])
    mask = np.array([pair in selected for pair in observations.observed_pairs])
    expected_nll, beta, _, _ = dense_profile(
        mlpe_reference.distances[mask],
        mlpe_reference.targets[mask] + 1,
        left,
        right,
        len(lookup),
        [head.unit_variance, head.residual_variance],
        jitter=1e-7,
    )
    assert head.ml_log_likelihood == pytest.approx(-expected_nll, abs=1e-10)
    np.testing.assert_allclose([head.intercept, head.slope], beta, atol=1e-10)
    query = head.predict_marginal([0.1], [("new-north", "new-south")])
    assert query.prediction.values[0] == pytest.approx(np.expm1(query.model_values[0]))
    assert head.calibration_pairs == selected
    assert set(head.calibration_roles) == {"training"}
    # No extra measurements from absent pairs enter the fit or posterior.
    assert len(head.calibration_pairs) == 25
    z = np.zeros((len(selected), len(lookup)))
    z[np.arange(len(selected)), left] = z[np.arange(len(selected)), right] = 1
    expected_covariance = np.linalg.inv(
        np.eye(len(lookup)) / head.unit_variance + z.T @ z / (head.residual_variance + 1e-7)
    )
    np.testing.assert_allclose(head.effect_covariance, expected_covariance, atol=1e-11)


def test_explicit_numerical_constraints_and_optimization_failure_are_visible(
    reference_observations, mlpe_reference
):
    from ilg_toolkit.mlpe import MLPEConfig, MLPEError

    invalid_options: tuple[dict[str, Any], ...] = (
        {"variance_floor": 0},
        {"min_distance_scale": float("nan")},
        {"jitter": 1e-3},
    )
    for options in invalid_options:
        with pytest.raises(MLPEError):
            MLPEConfig(**options)
    with pytest.raises(MLPEError):
        calibrate_mlpe(
            mlpe_reference.distances,
            reference_observations(),
            region_name="alpine",
            config=MLPEConfig(max_iterations=1),
        )
