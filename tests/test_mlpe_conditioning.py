"""Prediction seam: independently condition the joint observation Gaussian."""

import numpy as np
import pytest
from mlpe_reference import _R_SCORES
from test_mlpe import reference_observations

from ilg_toolkit.mlpe import MLPEConfig, calibrate_mlpe


@pytest.fixture(scope="module")
def calibrated():
    observations = reference_observations()
    head = calibrate_mlpe(
        _R_SCORES, observations, region_name="alpine", config=MLPEConfig(jitter=1e-7)
    )
    return head, observations


def joint_gaussian_reference(head, observed_pairs, observed_scores, targets, query_pairs, scores):
    """Dense joint-observation oracle; never uses the stored effect posterior."""
    all_pairs = tuple(observed_pairs) + tuple(query_pairs)
    labels = sorted({label for pair in all_pairs for label in pair})
    incidence = np.array([[float(label in pair) for label in labels] for pair in all_pairs])
    covariance = head.unit_variance * incidence @ incidence.T
    covariance += (head.residual_variance + head.config.jitter) * np.eye(len(all_pairs))
    all_scores = np.r_[observed_scores, scores]
    fixed = head.intercept + head.slope * (all_scores - head.score_center) / head.score_scale
    n = len(observed_pairs)
    cross = covariance[n:, :n]
    mean = fixed[n:] + cross @ np.linalg.solve(covariance[:n, :n], targets - fixed[:n])
    conditional = covariance[n:, n:] - cross @ np.linalg.solve(covariance[:n, :n], cross.T)
    return mean, np.diag(conditional)


def test_known_effects_and_unseen_prior_match_joint_gaussian(calibrated):
    from ilg_toolkit.mlpe import predict_known_effects

    head, observations = calibrated
    pairs = (("p01", "p02"), ("new", "p03"), ("new", "other-new"))
    scores = np.array([0.2, 0.8, -0.1])
    expected_mean, expected_variance = joint_gaussian_reference(
        head, observations.observed_pairs, _R_SCORES, observations.observed_values, pairs, scores
    )
    result = predict_known_effects(head, scores, pairs)
    np.testing.assert_allclose(result.model_values, expected_mean, atol=2e-11)
    np.testing.assert_allclose(result.model_variance, expected_variance, atol=2e-11)
    marginal = head.predict_marginal(scores, pairs)
    assert result.model_values[0] != pytest.approx(marginal.model_values[0])
    assert result.model_values[-1] == pytest.approx(marginal.model_values[-1])
    assert result.provenance.unseen_effect_policy == "independent_prior"
    assert result.provenance.mode == "known_effects"


def test_supplied_support_updates_effects_and_matches_joint_gaussian(calibrated):
    from ilg_toolkit import ObservationPartition, PairwiseObservations
    from ilg_toolkit.mlpe import condition_on_support

    head, calibration = calibrated
    support = PairwiseObservations.from_pairs(
        [("new", "p01"), ("p04", "new"), ("second-new", "p02")],
        [1.4, 0.8, 2.1],
        target=head.target,
    )
    support_scores = np.array([0.7, -0.2, 0.3])  # aligned to observed_pairs, not input order
    partition = ObservationPartition("alpine", support.observed_pairs, role="support")
    conditioned = condition_on_support(head, support_scores, support, partition=partition)
    pairs = (("new", "p05"), ("second-new", "new"), ("third-new", "p01"))
    scores = np.array([0.1, 0.6, 1.2])
    result = conditioned.predict(scores, pairs)
    expected_mean, expected_variance = joint_gaussian_reference(
        head,
        calibration.observed_pairs + support.observed_pairs,
        np.r_[_R_SCORES, support_scores],
        np.r_[calibration.observed_values, support.observed_values],
        pairs,
        scores,
    )
    np.testing.assert_allclose(result.model_values, expected_mean, atol=3e-11)
    np.testing.assert_allclose(result.model_variance, expected_variance, atol=3e-11)
    assert result.provenance.mode == "support"
    assert set(result.provenance.support_pairs) == set(partition.pairs)
    assert result.provenance.support_roles == ("support",) * 3
    assert result.provenance.calibration_pairs == head.calibration_pairs
    assert conditioned.head is head


def test_support_cannot_reuse_calibration_or_query_pairs_even_when_reversed(calibrated):
    from ilg_toolkit import (
        MLPEError,
        ObservationPartition,
        PairwiseObservations,
        condition_on_support,
    )

    head, _ = calibrated
    reused = PairwiseObservations.from_pairs([("p02", "p01")], [1.0], target=head.target)
    partition = ObservationPartition("alpine", [("p01", "p02")], role="support")
    with pytest.raises(MLPEError, match="calibration.*twice|reuses.*calibration"):
        condition_on_support(head, [0.3], reused, partition=partition)

    support = PairwiseObservations.from_pairs([("new", "p01")], [1.0], target=head.target)
    partition = ObservationPartition("alpine", [("new", "p01")], role="support")
    conditioned = condition_on_support(head, [0.3], support, partition=partition)
    with pytest.raises(MLPEError, match="Support/query overlap"):
        conditioned.predict([0.3], [("p01", "new")])
    with pytest.raises(MLPEError, match="duplicate unordered"):
        conditioned.predict([0.2, 0.2], [("p02", "new"), ("new", "p02")])
    with pytest.raises(TypeError):
        conditioned.predict([0.3], [("p02", "new")], query_targets=[7.0])
    with pytest.raises(TypeError):
        condition_on_support(head, [0.3], support, partition=partition, query_targets=[7.0])


def test_nonlinear_points_inverse_transform_mean_but_variance_stays_model_scale():
    from ilg_toolkit import ObservationPartition, PairwiseObservations, TargetSpec

    reference = reference_observations()
    target = TargetSpec("divergence", units="fraction", transform="log1p")
    observations = PairwiseObservations.from_pairs(
        reference.observed_pairs, np.expm1(reference.observed_values + 2.0), target=target
    )
    head = calibrate_mlpe(_R_SCORES, observations, region_name="log-region")
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
        np.r_[_R_SCORES, 0.4],
        np.r_[target.forward(observations.observed_values), 2.3],
        [("new", "p01")],
        [0.2],
    )
    np.testing.assert_allclose(result.values, np.expm1(expected_mean), atol=1e-10)
    np.testing.assert_allclose(result.model_variance, expected_variance, atol=1e-11)
    np.testing.assert_allclose(
        result.model_variance, result.effect_variance + result.residual_variance + result.jitter
    )
    assert result.scale == known.scale == "original"
    assert result.variance_scale == known.variance_scale == "model"
    assert result.target == target
    assert set(result.excluded_uncertainty) == {
        "encoder",
        "fixed_effect_coefficients",
        "variance_parameters",
    }
    assert result.values[0] != pytest.approx(np.expm1(expected_mean[0] + expected_variance[0] / 2))


def test_conditioning_rejects_undeclared_targets_and_incompatible_support(calibrated):
    from ilg_toolkit import MLPEError, ObservationPartition, PairwiseObservations, TargetSpec

    head, _ = calibrated
    support = PairwiseObservations.from_pairs(
        [("new", "p01"), ("new", "p02")], [1.0, 2.0], target=head.target
    )
    for role in ("training", "calibration", "validation", "query"):
        with pytest.raises(MLPEError, match="support-role"):
            head.condition_on_support(
                [0.1, 0.3],
                support,
                partition=ObservationPartition("alpine", support.observed_pairs, role=role),
            )
    with pytest.raises(MLPEError, match="exactly.*declared"):
        head.condition_on_support(
            [0.1, 0.3],
            support,
            partition=ObservationPartition("alpine", support.observed_pairs[:1], role="support"),
        )
    partition = ObservationPartition("alpine", support.observed_pairs, role="support")
    incompatible = PairwiseObservations.from_pairs(
        support.observed_pairs, [1.0, 2.0], target=TargetSpec("different measurement")
    )
    with pytest.raises(MLPEError, match="target metadata"):
        head.condition_on_support([0.1, 0.3], incompatible, partition=partition)
    with pytest.raises(MLPEError, match="region_name"):
        head.condition_on_support(
            [0.1, 0.3],
            support,
            partition=ObservationPartition(
                "another-region", support.observed_pairs, role="support"
            ),
        )
    with pytest.raises(MLPEError, match="finite.*aligned"):
        head.condition_on_support([np.nan, 0.3], support, partition=partition)


def test_model_conditions_real_landscape_scores_without_updating_encoder():
    import itertools

    import jax.numpy as jnp
    from test_recalibration import ScalarEmbedding, problem

    from ilg_toolkit import (
        TrainingConfig,
        ObservationPartition,
        PairwiseObservations,
        fit,
        recalibrate,
    )

    region, observations, _, _ = problem("conditioned-landscape")
    calibration_pairs = tuple(itertools.combinations(region.sampling_unit_ids[:4], 2))
    training = ObservationPartition(region.name, calibration_pairs, role="training")
    direct = fit(
        region,
        observations,
        model=ScalarEmbedding(jnp.asarray(1.0)),
        config=TrainingConfig(epochs=0),
        partition=training,
    ).model
    model = recalibrate(direct, region, observations)
    head = model.calibrations[region.name]
    pairs = (("population-4", "population-2"), ("population-5", "population-3"))
    known = model.predict_known_effects(region, pairs)
    scores = model.landscape_scores(region)
    lookup = {label: index for index, label in enumerate(region.sampling_unit_ids)}

    def selected_scores(selected):
        return np.array([scores[lookup[a], lookup[b]] for a, b in selected])

    expected_mean, expected_variance = joint_gaussian_reference(
        head,
        calibration_pairs,
        selected_scores(calibration_pairs),
        observations.aligned_values(region)[
            [lookup[a] for a, _ in calibration_pairs], [lookup[b] for _, b in calibration_pairs]
        ],
        pairs,
        selected_scores(pairs),
    )
    np.testing.assert_allclose(known.model_values, expected_mean, atol=1e-10)
    np.testing.assert_allclose(known.model_variance, expected_variance, atol=1e-10)
    support = PairwiseObservations.from_pairs(
        [("population-4", "population-0"), ("population-4", "population-1")],
        [2.0, 2.5],
        target=head.target,
    )
    result = model.predict_with_support(
        region,
        pairs,
        support,
        support_partition=ObservationPartition(region.name, support.observed_pairs, role="support"),
    )
    expected_mean, expected_variance = joint_gaussian_reference(
        head,
        calibration_pairs + support.observed_pairs,
        np.r_[selected_scores(calibration_pairs), selected_scores(support.observed_pairs)],
        np.r_[
            observations.aligned_values(region)[
                [lookup[a] for a, _ in calibration_pairs], [lookup[b] for _, b in calibration_pairs]
            ],
            support.observed_values,
        ],
        pairs,
        selected_scores(pairs),
    )
    np.testing.assert_allclose(result.model_values, expected_mean, atol=1e-10)
    np.testing.assert_allclose(result.model_variance, expected_variance, atol=1e-10)
    assert model.encoder is direct.encoder
    assert result.provenance.support_pairs == tuple(
        tuple(sorted(pair)) for pair in support.observed_pairs
    )
    with pytest.raises(ValueError, match="MLPE"):
        direct.predict_known_effects(region, pairs)
    with pytest.raises(ValueError, match="sampling-unit.*region"):
        model.predict_known_effects(region, [("missing-location", "population-0")])
