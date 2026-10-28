"""Scientific behavior of maximum-likelihood population effects (MLPE) prediction."""

import numpy as np
import pytest
from ilg_toolkit import MLPEError, PairwiseObservations, TargetSpec
from ilg_toolkit.mlpe import calibrate_mlpe


def test_known_effects_and_unseen_prior_match_joint_gaussian(
    joint_gaussian_reference, frozen_r_conditional_reference, mlpe_reference, calibrated
):
    head, observations = calibrated
    pairs = (("p01", "p02"), ("new", "p03"), ("new", "other-new"))
    distances = np.array([0.2, 0.8, -0.1])
    expected_mean, expected_variance = joint_gaussian_reference(
        head,
        observations.observed_pairs,
        mlpe_reference.distances,
        observations.observed_values,
        pairs,
        distances,
    )
    result = head.predict(distances, pairs, known_effects=True)
    assert all(value.dtype == np.float64 for value in (result.values, *result.extras.values()))
    np.testing.assert_allclose(result.extras["model_values"], expected_mean, atol=2e-11)
    np.testing.assert_allclose(result.extras["model_variance"], expected_variance, atol=2e-11)
    # Frozen R coefficients and population effects provide a separate mean reference.
    r_fixed = (
        mlpe_reference.fixed[0]
        + mlpe_reference.fixed[1] * (distances + 0.14328369260805038) / 1.012896471531856
    )
    r_effects = [mlpe_reference.blups[0] + mlpe_reference.blups[1], mlpe_reference.blups[2], 0]
    np.testing.assert_allclose(result.values, r_fixed + r_effects, atol=2e-5)
    frozen_expected = frozen_r_conditional_reference(
        observations.observed_pairs,
        mlpe_reference.distances,
        mlpe_reference.targets,
        pairs,
        distances,
        jitter=1e-7,
    )
    for quantity, expected in frozen_expected.items():
        # Frozen R parameters stay fixed; the jittered fit differs by at most 1.3e-7 here.
        np.testing.assert_allclose(result.extras[quantity], expected, rtol=0, atol=2e-6)
    marginal = head.predict(distances, pairs)
    assert result.values[0] != pytest.approx(marginal.values[0])
    assert result.values[-1] == pytest.approx(marginal.values[-1])


def test_supplied_support_updates_effects_and_matches_joint_gaussian(
    joint_gaussian_reference, frozen_r_conditional_reference, mlpe_reference, calibrated
):
    head, calibration = calibrated
    support = PairwiseObservations.from_pairs(
        [("new", "p01"), ("p04", "new"), ("second-new", "p02")],
        [1.4, 0.8, 2.1],
        target=head.target,
    )
    support_distances = np.array([0.7, -0.2, 0.3])  # Aligned with observed_pairs.
    pairs = (("new", "p05"), ("second-new", "new"), ("third-new", "p01"))
    distances = np.array([0.1, 0.6, 1.2])
    known = head.predict(distances, pairs, known_effects=True)
    result = head.predict(distances, pairs, support=support, support_distances=support_distances)
    assert all(value.dtype == np.float64 for value in (result.values, *result.extras.values()))
    expected_mean, expected_variance = joint_gaussian_reference(
        head,
        calibration.observed_pairs + support.observed_pairs,
        np.r_[mlpe_reference.distances, support_distances],
        np.r_[calibration.observed_values, support.observed_values],
        pairs,
        distances,
    )
    np.testing.assert_allclose(result.extras["model_values"], expected_mean, atol=3e-11)
    np.testing.assert_allclose(result.extras["model_variance"], expected_variance, atol=3e-11)
    frozen_expected = frozen_r_conditional_reference(
        calibration.observed_pairs + support.observed_pairs,
        np.r_[mlpe_reference.distances, support_distances],
        np.r_[mlpe_reference.targets, support.observed_values],
        pairs,
        distances,
        jitter=1e-7,
    )
    for quantity, expected in frozen_expected.items():
        # Support amplifies the jittered-fit difference to 7.3e-7; R literals have 16+ digits.
        np.testing.assert_allclose(result.extras[quantity], expected, rtol=0, atol=2e-6)
    np.testing.assert_array_equal(
        head.predict(distances, pairs, known_effects=True).values, known.values
    )


def test_support_cannot_reuse_calibration_or_query_pairs_even_when_reversed(calibrated):
    head, _ = calibrated
    reused = PairwiseObservations.from_pairs([("p02", "p01")], [1.0], target=head.target)
    with pytest.raises(MLPEError):
        head.predict([0.2], [("new", "p02")], support=reused, support_distances=[0.3])
    support = PairwiseObservations.from_pairs([("new", "p01")], [1.0], target=head.target)
    for distances, pairs in (
        ([0.3], [("p01", "new")]),
        ([0.2, 0.2], [("p02", "new"), ("new", "p02")]),
    ):
        with pytest.raises(MLPEError):
            head.predict(distances, pairs, support=support, support_distances=[0.3])


def test_nonlinear_points_inverse_transform_mean_but_variance_stays_model_scale(
    joint_gaussian_reference, reference_observations, mlpe_reference
):
    reference = reference_observations()
    target = TargetSpec("divergence", units="fraction", transform="log1p")
    observations = PairwiseObservations.from_pairs(
        reference.observed_pairs, np.expm1(reference.observed_values + 2.0), target=target
    )
    head = calibrate_mlpe(mlpe_reference.distances, observations, region_name="log-region")
    support = PairwiseObservations.from_pairs([("new", "p02")], [np.expm1(2.3)], target=target)
    result = head.predict([0.2], [("new", "p01")], support=support, support_distances=[0.4])
    expected_mean, expected_variance = joint_gaussian_reference(
        head,
        observations.observed_pairs + support.observed_pairs,
        np.r_[mlpe_reference.distances, 0.4],
        np.r_[target.forward(observations.observed_values), 2.3],
        [("new", "p01")],
        [0.2],
    )
    np.testing.assert_allclose(result.values, np.expm1(expected_mean), atol=1e-10)
    np.testing.assert_allclose(result.extras["model_variance"], expected_variance, atol=1e-11)
    np.testing.assert_allclose(
        result.extras["model_variance"],
        result.extras["effect_variance"] + head.residual_variance + head.loss.jitter,
    )
    assert result.values[0] != pytest.approx(np.expm1(expected_mean[0] + expected_variance[0] / 2))


def test_conditioning_rejects_incompatible_support(calibrated):
    head, _ = calibrated
    support = PairwiseObservations.from_pairs([("new", "p01")], [1.0], target=head.target)
    incompatible = PairwiseObservations.from_pairs(
        support.observed_pairs, [1.0], target=TargetSpec("different measurement")
    )
    for supplied, distances in ((incompatible, [0.1]), (support, [np.nan]), (support, [0.1, 0.3])):
        with pytest.raises(MLPEError):
            head.predict([0.2], [("new", "p02")], support=supplied, support_distances=distances)
    with pytest.raises(MLPEError):
        head.predict(
            [0.2], [("new", "p02")], known_effects=True, support=support, support_distances=[0.1]
        )
    with pytest.raises(MLPEError):
        head.predict([0.2], [("new", "p02")], support=support)
    with pytest.raises(MLPEError):
        head.predict([0.2], [("new", "p02")], support_distances=[0.1])
