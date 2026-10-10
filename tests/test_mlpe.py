"""Standalone calibration and independent numerical seams accepted in spec #1."""

import itertools
from typing import Any

import numpy as np
import pytest
from mlpe_reference import (
    _R_BLUPS,
    _R_FIXED,
    _R_LOG_LIKELIHOOD,
    _R_SCORES,
    _R_TARGETS,
    _R_VARIANCES,
)

from ilg_toolkit import PairwiseObservations, TargetSpec
from ilg_toolkit.mlpe import calibrate_mlpe


def reference_observations():
    ids = tuple(f"p{index:02d}" for index in range(1, 9))
    pairs = tuple(itertools.combinations(ids, 2))
    return PairwiseObservations.from_pairs(
        pairs, _R_TARGETS, target=TargetSpec("synthetic dissimilarity", units="index")
    )


def test_calibration_matches_frozen_r_full_ml_and_predicts_without_query_targets():
    observations = reference_observations()
    head = calibrate_mlpe(_R_SCORES, observations, region_name="alpine")
    np.testing.assert_allclose([head.intercept, head.slope], _R_FIXED, rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(
        [head.unit_variance, head.residual_variance], _R_VARIANCES, rtol=2e-4, atol=2e-6
    )
    assert head.ml_log_likelihood == pytest.approx(_R_LOG_LIKELIHOOD, abs=2e-5)
    np.testing.assert_allclose(head.effect_mean, _R_BLUPS, rtol=2e-4, atol=2e-5)
    assert head.score_center == pytest.approx(-0.14328369260805038, abs=1e-14)
    assert head.score_scale == pytest.approx(1.012896471531856, abs=1e-14)
    query = head.predict_marginal([2.5, -2.0], [("new", "p01"), ("p02", "new")])
    # Independent frozen predictions from the R coefficients and score moments.
    np.testing.assert_allclose(query.values, [-0.49407802616878604, 2.4619456646531583], atol=1e-5)
    assert query.values[0] < 0  # Signed calibration is not clipped or repaired.
    assert query.target.units == "index"
    assert head.region_name == "alpine"
    assert head.calibration_pairs == observations.observed_pairs
    assert set(head.calibration_roles) == {"calibration"}


def test_marginal_prediction_retains_float64_jax_values_without_changing_global_precision():
    import jax
    import jax.numpy as jnp

    with jax.enable_x64(False):
        head = calibrate_mlpe(_R_SCORES, reference_observations(), region_name="alpine")
        query = head.predict_marginal([2.5, -2.0], [("new", "p01"), ("p02", "new")])
        assert not jax.config.read("jax_enable_x64")
    assert isinstance(query.values, jax.Array)
    assert isinstance(query.model_values, jax.Array)
    assert query.values.dtype == query.model_values.dtype == jnp.float64
    np.testing.assert_allclose(query.values, [-0.49407802616878604, 2.4619456646531583], atol=1e-5)


def test_differentiable_full_ml_matches_dense_incomplete_pair_oracle():
    import jax
    import jax.numpy as jnp
    from jax import enable_x64
    from mlpe_reference import dense_profile

    from ilg_toolkit.mlpe import profiled_mlpe_ml_fit

    # Arbitrary order and orientation with two absent pairs.
    scores = np.array([0.2, 1.1, -0.3, 2.0])
    targets = np.array([0.5, 0.1, 0.9, 0.3])
    left, right = np.array([3, 0, 2, 0]), np.array([1, 2, 1, 1])
    raw = np.array([-0.7, -1.4])
    variances = np.logaddexp(0, raw) + 1e-10
    expected_nll, expected_beta = dense_profile(scores, targets, left, right, 4, variances)
    with enable_x64():

        def objective(values):
            return profiled_mlpe_ml_fit(
                values,
                jnp.asarray(targets),
                jnp.asarray(left),
                jnp.asarray(right),
                n_populations=4,
                raw_variances=jnp.asarray(raw),
            )

        nll, beta = objective(jnp.asarray(scores))
        np.testing.assert_allclose([nll, *beta], [expected_nll, *expected_beta], atol=1e-11)
        compiled = jax.jit(objective)(jnp.asarray(scores))
        np.testing.assert_allclose(compiled[0], nll, atol=1e-11)
        gradient = jax.grad(lambda values: objective(values)[0])(jnp.asarray(scores))
    plus, minus = scores.copy(), scores.copy()
    plus[2] += 1e-5
    minus[2] -= 1e-5
    expected_gradient = (
        dense_profile(plus, targets, left, right, 4, variances)[0]
        - dense_profile(minus, targets, left, right, 4, variances)[0]
    ) / 2e-5
    assert gradient[2] == pytest.approx(expected_gradient, rel=1e-5, abs=1e-8)


def test_calibration_rejects_query_targets_and_unidentified_variances():
    from ilg_toolkit import ObservationPartition
    from ilg_toolkit.mlpe import MLPEError

    observations = reference_observations()
    query = ObservationPartition("alpine", observations.observed_pairs, role="query")
    with pytest.raises(MLPEError, match="query|support"):
        calibrate_mlpe(_R_SCORES, observations, region_name="alpine", partition=query)
    disjoint = PairwiseObservations.from_pairs(
        [("a", "b"), ("c", "d"), ("e", "f")], [1, 3, 2], target=observations.target
    )
    with pytest.raises(MLPEError, match="unidentifiable"):
        calibrate_mlpe([1, 2, 3], disjoint, region_name="alpine")
    with pytest.raises(MLPEError, match="constant|singular"):
        calibrate_mlpe(np.ones(len(_R_SCORES)), observations, region_name="alpine")


def test_incomplete_partition_calibration_records_roles_and_original_target_scale():
    from mlpe_reference import dense_profile

    from ilg_toolkit import MLPEConfig, ObservationPartition, calibrate_mlpe

    reference = reference_observations()
    target = TargetSpec("divergence", units="fraction", transform="log1p")
    observations = PairwiseObservations.from_pairs(
        reference.observed_pairs,
        np.expm1(_R_TARGETS + 1),
        target=target,
    )
    selected = tuple(
        pair for i, pair in enumerate(observations.observed_pairs) if i not in {2, 7, 11}
    )
    partition = ObservationPartition("river", selected, role="training")
    head = calibrate_mlpe(
        _R_SCORES,
        observations,
        region_name="river",
        partition=partition,
        config=MLPEConfig(jitter=1e-7),
    )
    lookup = {label: i for i, label in enumerate(head.population_ids)}
    left = np.array([lookup[a] for a, b in selected])
    right = np.array([lookup[b] for a, b in selected])
    mask = np.array([pair in selected for pair in observations.observed_pairs])
    expected_nll, beta = dense_profile(
        _R_SCORES[mask],
        _R_TARGETS[mask] + 1,
        left,
        right,
        len(lookup),
        [head.unit_variance, head.residual_variance],
        jitter=1e-7,
    )
    assert head.ml_log_likelihood == pytest.approx(-expected_nll, abs=1e-10)
    np.testing.assert_allclose([head.intercept, head.slope], beta, atol=1e-10)
    query = head.predict_marginal([0.1], [("new-north", "new-south")])
    assert query.values[0] == pytest.approx(np.expm1(query.model_values[0]))
    assert query.target == target
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


def test_explicit_numerical_constraints_and_optimization_failure_are_visible():
    from ilg_toolkit.mlpe import MLPEConfig, MLPEError

    invalid_options: tuple[dict[str, Any], ...] = (
        {"variance_floor": 0},
        {"min_score_scale": float("nan")},
        {"jitter": 1e-3},
    )
    for options in invalid_options:
        with pytest.raises(MLPEError):
            MLPEConfig(**options)
    with pytest.raises(MLPEError, match="converged"):
        calibrate_mlpe(
            _R_SCORES,
            reference_observations(),
            region_name="alpine",
            config=MLPEConfig(max_iterations=1),
        )


@pytest.mark.parametrize("field", ["variance_floor", "min_score_scale", "jitter", "max_iterations"])
@pytest.mark.parametrize("value", [False, True, np.bool_(False), np.bool_(True)])
def test_mlpe_config_rejects_booleans_at_construction(field, value):
    from ilg_toolkit import MLPEConfig, MLPEError

    with pytest.raises(MLPEError, match="boolean"):
        MLPEConfig(**{field: value})
