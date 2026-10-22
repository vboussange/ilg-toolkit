"""Scientific behavior of mlpe.prediction."""

import numpy as np
import pytest

from ilg_toolkit.mlpe import calibrate_mlpe


def test_known_effects_and_unseen_prior_match_joint_gaussian(
    joint_gaussian_reference, mlpe_reference, calibrated
):

    from ilg_toolkit.mlpe import predict_known_effects

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
    result = predict_known_effects(head, distances, pairs)
    assert result.prediction.pairs == pairs
    assert result.prediction.sampling_unit_ids == ("p01", "p02", "new", "p03", "other-new")
    assert all(
        value.dtype == np.float64
        for value in (
            result.prediction.values,
            result.model_values,
            result.model_variance,
            result.effect_variance,
        )
    )
    np.testing.assert_allclose(result.model_values, expected_mean, atol=2e-11)
    np.testing.assert_allclose(result.model_variance, expected_variance, atol=2e-11)
    marginal = head.predict_marginal(distances, pairs)
    assert result.model_values[0] != pytest.approx(marginal.model_values[0])
    assert result.model_values[-1] == pytest.approx(marginal.model_values[-1])
    assert result.provenance.unseen_effect_policy == "independent_prior"
    assert result.provenance.mode == "known_effects"


def test_supplied_support_updates_effects_and_matches_joint_gaussian(
    joint_gaussian_reference, mlpe_reference, calibrated
):

    from ilg_toolkit import ObservationPartition, PairwiseObservations
    from ilg_toolkit.mlpe import condition_on_support

    head, calibration = calibrated
    support = PairwiseObservations.from_pairs(
        [("new", "p01"), ("p04", "new"), ("second-new", "p02")],
        [1.4, 0.8, 2.1],
        target=head.target,
    )
    support_distances = np.array([0.7, -0.2, 0.3])  # aligned to observed_pairs, not input order
    partition = ObservationPartition("alpine", support.observed_pairs, role="support")
    conditioned = condition_on_support(head, support_distances, support, partition=partition)
    pairs = (("new", "p05"), ("second-new", "new"), ("third-new", "p01"))
    distances = np.array([0.1, 0.6, 1.2])
    result = conditioned.predict(distances, pairs)
    assert all(
        value.dtype == np.float64
        for value in (
            result.prediction.values,
            result.model_values,
            result.model_variance,
            result.effect_variance,
        )
    )
    expected_mean, expected_variance = joint_gaussian_reference(
        head,
        calibration.observed_pairs + support.observed_pairs,
        np.r_[mlpe_reference.distances, support_distances],
        np.r_[calibration.observed_values, support.observed_values],
        pairs,
        distances,
    )
    np.testing.assert_allclose(result.model_values, expected_mean, atol=3e-11)
    np.testing.assert_allclose(result.model_variance, expected_variance, atol=3e-11)
    assert result.provenance.mode == "support"
    assert set(result.provenance.support_pairs) == set(partition.pairs)
    assert result.provenance.support_roles == ("support",) * 3
    assert result.provenance.calibration_pairs == head.calibration_pairs
    assert conditioned.head is head


def test_support_cannot_reuse_calibration_or_query_pairs_even_when_reversed(calibrated):
    from operator import methodcaller

    from ilg_toolkit import (
        MLPEError,
        ObservationPartition,
        PairwiseObservations,
        condition_on_support,
    )

    head, _ = calibrated
    reused = PairwiseObservations.from_pairs([("p02", "p01")], [1.0], target=head.target)
    partition = ObservationPartition("alpine", (("p01", "p02"),), role="support")
    with pytest.raises(MLPEError):
        condition_on_support(head, [0.3], reused, partition=partition)

    support = PairwiseObservations.from_pairs([("new", "p01")], [1.0], target=head.target)
    partition = ObservationPartition("alpine", (("new", "p01"),), role="support")
    conditioned = condition_on_support(head, [0.3], support, partition=partition)
    with pytest.raises(MLPEError):
        conditioned.predict([0.3], [("p01", "new")])
    with pytest.raises(MLPEError):
        conditioned.predict([0.2, 0.2], [("p02", "new"), ("new", "p02")])
    with pytest.raises(TypeError):
        methodcaller("predict", [0.3], [("p02", "new")], query_targets=[7.0])(conditioned)
    with pytest.raises(TypeError):
        methodcaller(
            "condition_on_support", [0.3], support, partition=partition, query_targets=[7.0]
        )(head)


def test_nonlinear_points_inverse_transform_mean_but_variance_stays_model_scale(
    joint_gaussian_reference, reference_observations, mlpe_reference
):
    from ilg_toolkit import ObservationPartition, PairwiseObservations, TargetSpec

    reference = reference_observations()
    target = TargetSpec("divergence", units="fraction", transform="log1p")
    observations = PairwiseObservations.from_pairs(
        reference.observed_pairs, np.expm1(reference.observed_values + 2.0), target=target
    )
    head = calibrate_mlpe(mlpe_reference.distances, observations, region_name="log-region")
    known = head.predict_known_effects([0.2], [("new", "p01")])
    support = PairwiseObservations.from_pairs([("new", "p02")], [np.expm1(2.3)], target=target)
    conditioned = head.condition_on_support(
        [0.4],
        support,
        partition=ObservationPartition("log-region", support.observed_pairs, role="support"),
    )
    result = conditioned.predict([0.2], [("new", "p01")])
    expected_mean, expected_variance = joint_gaussian_reference(
        head,
        observations.observed_pairs + support.observed_pairs,
        np.r_[mlpe_reference.distances, 0.4],
        np.r_[target.forward(observations.observed_values), 2.3],
        [("new", "p01")],
        [0.2],
    )
    np.testing.assert_allclose(result.prediction.values, np.expm1(expected_mean), atol=1e-10)
    np.testing.assert_allclose(result.model_variance, expected_variance, atol=1e-11)
    np.testing.assert_allclose(
        result.model_variance, result.effect_variance + result.residual_variance + result.jitter
    )
    assert result.prediction.scale == known.prediction.scale == "original"
    assert result.variance_scale == known.variance_scale == "model"
    assert result.prediction.target == target
    assert set(result.excluded_uncertainty) == {
        "encoder",
        "fixed_effect_coefficients",
        "variance_parameters",
    }
    assert result.prediction.values[0] != pytest.approx(
        np.expm1(expected_mean[0] + expected_variance[0] / 2)
    )


def test_conditioning_rejects_undeclared_targets_and_incompatible_support(calibrated):
    from ilg_toolkit import MLPEError, ObservationPartition, PairwiseObservations, TargetSpec

    head, _ = calibrated
    support = PairwiseObservations.from_pairs(
        [("new", "p01"), ("new", "p02")], [1.0, 2.0], target=head.target
    )
    for role in ("training", "calibration", "validation", "query"):
        with pytest.raises(MLPEError):
            head.condition_on_support(
                [0.1, 0.3],
                support,
                partition=ObservationPartition("alpine", support.observed_pairs, role=role),
            )
    with pytest.raises(MLPEError):
        head.condition_on_support(
            [0.1, 0.3],
            support,
            partition=ObservationPartition("alpine", support.observed_pairs[:1], role="support"),
        )
    partition = ObservationPartition("alpine", support.observed_pairs, role="support")
    incompatible = PairwiseObservations.from_pairs(
        support.observed_pairs, [1.0, 2.0], target=TargetSpec("different measurement")
    )
    with pytest.raises(MLPEError):
        head.condition_on_support([0.1, 0.3], incompatible, partition=partition)
    with pytest.raises(MLPEError):
        head.condition_on_support(
            [0.1, 0.3],
            support,
            partition=ObservationPartition(
                "another-region", support.observed_pairs, role="support"
            ),
        )
    with pytest.raises(MLPEError):
        head.condition_on_support([np.nan, 0.3], support, partition=partition)
