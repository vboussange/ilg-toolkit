"""Tests of the private population-covariance likelihood kernel against a joint Gaussian."""

import numpy as np
import pytest


def test_fixed_coefficient_likelihood_matches_direct_joint_gaussian():
    import jax
    import jax.numpy as jnp
    from ilg_toolkit.mlpe.likelihood import _mlpe_ml_negative_log_likelihood

    distances = np.array([0.2, 1.1, -0.3, 2.0])
    y = np.array([0.5, 0.1, 0.9, 0.3])
    left, right = np.array([3, 0, 2, 0]), np.array([1, 2, 1, 1])
    beta, variances = np.array([0.4, -0.2]), np.array([0.6, 0.12])
    raw = np.log(np.expm1(variances - 1e-10))
    z = np.zeros((4, 4))
    z[np.arange(4), left] = z[np.arange(4), right] = 1
    x = np.column_stack((np.ones(4), (distances - distances.mean()) / distances.std(ddof=1)))
    residual = y - x @ beta
    v = variances[0] * (z @ z.T) + (variances[1] + 1e-7) * np.eye(4)
    expected_nll = 0.5 * (
        4 * np.log(2 * np.pi) + np.linalg.slogdet(v)[1] + residual @ np.linalg.solve(v, residual)
    )
    with jax.enable_x64():
        nll = _mlpe_ml_negative_log_likelihood(
            distances,
            y,
            left,
            right,
            n_populations=4,
            fixed_effects=jnp.asarray(beta),
            raw_variances=jnp.asarray(raw),
            distance_center=distances.mean(),
            distance_scale=distances.std(ddof=1),
            jitter=1e-7,
            variance_floor=1e-10,
            min_distance_scale=1e-12,
        )
    assert nll == pytest.approx(expected_nll, abs=1e-12)
