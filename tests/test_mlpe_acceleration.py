"""The accepted independent MLPE numerical seam, including missing pair masks."""

import numpy as np
import pytest
from mlpe_reference import dense_profile

from ilg_toolkit.mlpe import profiled_mlpe_ml_fit


def test_masked_pairs_are_absent_from_likelihood_and_score_standardization_under_jit():
    import jax
    import jax.numpy as jnp

    scores = np.array([0.2, np.nan, 1.1, -0.3, 2.0, np.inf])
    targets = np.array([0.5, np.nan, 0.1, 0.9, 0.3, np.inf])
    left, right = np.array([3, -999, 0, 2, 0, 999]), np.array([1, 999, 2, 1, 1, -999])
    mask = np.array([True, False, True, True, True, False])
    raw = np.array([-0.7, -1.4])
    expected_nll, expected_beta = dense_profile(
        scores[mask], targets[mask], left[mask], right[mask], 4, np.logaddexp(0, raw) + 1e-10
    )
    with jax.enable_x64():

        def objective(s):
            return profiled_mlpe_ml_fit(
                s,
                jnp.asarray(targets),
                jnp.asarray(left),
                jnp.asarray(right),
                n_populations=4,
                raw_variances=jnp.asarray(raw),
                pair_mask=jnp.asarray(mask),
            )

        actual = objective(jnp.asarray(scores))
        compiled = jax.jit(objective)(jnp.asarray(scores))
        gradient = jax.grad(lambda s: objective(s)[0])(jnp.asarray(scores))
    np.testing.assert_allclose([actual[0], *actual[1]], [expected_nll, *expected_beta], atol=1e-10)
    np.testing.assert_allclose([compiled[0], *compiled[1]], [actual[0], *actual[1]], atol=1e-10)
    assert np.isfinite(gradient).all()
    np.testing.assert_array_equal(np.asarray(gradient)[~mask], 0)


@pytest.mark.parametrize("incomplete", [False, True])
@pytest.mark.parametrize("variances", [(1e-8, 1.0), (1.0, 1.0), (1.0, 1e-8)])
def test_endpoint_likelihood_gls_and_gradients_agree_with_dense_extreme_variance_reference(
    incomplete, variances
):
    import jax
    import jax.numpy as jnp
    from mlpe_reference import decimal_dense_profile_and_gradients, dense_jax_profile

    rng = np.random.default_rng(12)
    left, right = np.triu_indices(6, 1)
    order = rng.permutation(len(left))
    if incomplete:
        order = order[3:]
    left, right = right[order], left[order]  # Reversed, shuffled endpoints.
    scores = rng.normal(size=len(left))
    targets = 0.5 - 0.4 * scores + rng.normal(scale=0.3, size=len(left))
    raw = np.log(np.expm1(np.asarray(variances) - 1e-10))
    with jax.enable_x64():
        s, y, raw = jnp.asarray(scores), jnp.asarray(targets), jnp.asarray(raw)

        def accelerated(s, raw):
            return profiled_mlpe_ml_fit(s, y, left, right, n_populations=6, raw_variances=raw)

        def dense(s, raw):
            return dense_jax_profile(s, y, left, right, 6, raw)

        actual_nll, actual_beta = accelerated(s, raw)
        dense_nll, dense_beta = dense(s, raw)
        actual_gradients = jax.grad(lambda s, r: accelerated(s, r)[0], argnums=(0, 1))(s, raw)
        dense_gradients = jax.grad(lambda s, r: dense(s, r)[0], argnums=(0, 1))(s, raw)
    np.testing.assert_allclose(actual_nll, dense_nll, rtol=2e-7, atol=1e-8)
    np.testing.assert_allclose(actual_beta, dense_beta, rtol=2e-7, atol=1e-8)
    # At u/e=1e8 the dense float64 unit-variance gradient subtracts terms of
    # order 1/e^2. Use the independent 60-digit dense oracle for that cancellation.
    if variances[1] < 1e-7:
        _, _, dense_gradients = decimal_dense_profile_and_gradients(
            scores, targets, left, right, np.asarray(raw)
        )
    for actual, expected in zip(actual_gradients, dense_gradients, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=5e-7)


def test_population_posterior_and_fixed_likelihood_match_direct_joint_gaussian():
    import jax
    import jax.numpy as jnp

    from ilg_toolkit.mlpe import mlpe_effect_posterior, mlpe_ml_negative_log_likelihood

    scores = np.array([0.2, 1.1, -0.3, 2.0])
    y = np.array([0.5, 0.1, 0.9, 0.3])
    left, right = np.array([3, 0, 2, 0]), np.array([1, 2, 1, 1])
    beta, variances = np.array([0.4, -0.2]), np.array([0.6, 0.12])
    raw = np.log(np.expm1(variances - 1e-10))
    z = np.zeros((4, 4))
    z[np.arange(4), left] = z[np.arange(4), right] = 1
    x = np.column_stack((np.ones(4), (scores - scores.mean()) / scores.std(ddof=1)))
    residual = y - x @ beta
    v = variances[0] * (z @ z.T) + (variances[1] + 1e-7) * np.eye(4)
    expected_mean = variances[0] * z.T @ np.linalg.solve(v, residual)
    expected_covariance = variances[0] * np.eye(4) - variances[0] ** 2 * z.T @ np.linalg.solve(v, z)
    expected_nll = 0.5 * (
        4 * np.log(2 * np.pi) + np.linalg.slogdet(v)[1] + residual @ np.linalg.solve(v, residual)
    )
    with jax.enable_x64():
        mean, covariance, factor = mlpe_effect_posterior(
            scores,
            y,
            left,
            right,
            n_populations=4,
            fixed_effects=jnp.asarray(beta),
            raw_variances=jnp.asarray(raw),
            score_center=scores.mean(),
            score_scale=scores.std(ddof=1),
            jitter=1e-7,
        )
        nll = mlpe_ml_negative_log_likelihood(
            scores,
            y,
            left,
            right,
            n_populations=4,
            fixed_effects=jnp.asarray(beta),
            raw_variances=jnp.asarray(raw),
            score_center=scores.mean(),
            score_scale=scores.std(ddof=1),
            jitter=1e-7,
        )
        mean, covariance, factor = map(np.asarray, (mean, covariance, factor))
    np.testing.assert_allclose(mean, expected_mean, atol=1e-12)
    np.testing.assert_allclose(covariance, expected_covariance, atol=1e-12)
    np.testing.assert_allclose(factor @ factor.T @ covariance, np.eye(4), atol=1e-12)
    assert nll == pytest.approx(expected_nll, abs=1e-12)


def test_profiled_score_and_variance_gradients_match_independent_finite_differences():
    import jax
    import jax.numpy as jnp

    scores = np.array([0.2, 1.1, -0.3, 2.0])
    targets = np.array([0.5, 0.1, 0.9, 0.3])
    left, right = np.array([3, 0, 2, 0]), np.array([1, 2, 1, 1])
    raw = np.array([-0.3, -1.2])

    def oracle(s, r):
        return dense_profile(s, targets, left, right, 4, np.logaddexp(0, r) + 1e-10)[0]

    with jax.enable_x64():

        def objective(s, r):
            return profiled_mlpe_ml_fit(s, targets, left, right, n_populations=4, raw_variances=r)[
                0
            ]

        gradients = jax.grad(objective, argnums=(0, 1))(jnp.asarray(scores), jnp.asarray(raw))
    for array_number, coordinate in [(0, 2), (1, 0), (1, 1)]:
        plus, minus = [scores.copy(), raw.copy()], [scores.copy(), raw.copy()]
        plus[array_number][coordinate] += 1e-5
        minus[array_number][coordinate] -= 1e-5
        expected = (oracle(*plus) - oracle(*minus)) / 2e-5
        assert gradients[array_number][coordinate] == pytest.approx(expected, rel=1e-6, abs=1e-8)


def test_unsupported_precision_and_singular_scores_have_explicit_nonfinite_diagnostics():
    import jax
    import jax.numpy as jnp

    # Low-level numerical kernels preserve their documented nonfinite-result
    # diagnostic contract; the public calibration/head boundaries reject it.
    left, right = np.triu_indices(4, 1)
    scores = np.arange(6, dtype=np.float32)
    with jax.enable_x64(False):
        raw = jnp.asarray(np.log(np.expm1(np.array([1.0, 1e-8]) - 1e-10)), dtype=jnp.float32)

        def objective(s):
            return profiled_mlpe_ml_fit(
                s,
                jnp.asarray(scores),
                left,
                right,
                n_populations=4,
                raw_variances=raw,
            )

        eager, beta = objective(jnp.asarray(scores))
        compiled, compiled_beta = jax.jit(objective)(jnp.asarray(scores))
    assert not np.isfinite(eager)
    assert not np.isfinite(compiled)
    assert not np.isfinite(beta).any()
    assert not np.isfinite(compiled_beta).any()
    with jax.enable_x64():
        nll, beta = profiled_mlpe_ml_fit(
            np.ones(6), scores, left, right, n_populations=4, raw_variances=jnp.zeros(2)
        )
    assert not np.isfinite(nll)
    assert not np.isfinite(beta).any()


def test_unidentifiable_profile_is_rejected_but_one_pair_frozen_scoring_is_valid():
    import jax
    import jax.numpy as jnp

    from ilg_toolkit.mlpe import mlpe_ml_negative_log_likelihood

    with jax.enable_x64():
        raw = jnp.log(jnp.expm1(jnp.array([0.6, 0.12]) - 1e-10))

        def unidentifiable():
            return profiled_mlpe_ml_fit(
                jnp.array([0.2, 0.5, 1.0]),
                jnp.array([0.3, 0.4, 0.5]),
                jnp.array([0, 2, 4]),
                jnp.array([1, 3, 5]),
                n_populations=6,
                raw_variances=raw,
            )[0]

        assert not np.isfinite(unidentifiable())
        assert not np.isfinite(jax.jit(unidentifiable)())
        actual = mlpe_ml_negative_log_likelihood(
            jnp.array([0.2]),
            jnp.array([0.3]),
            jnp.array([0]),
            jnp.array([1]),
            n_populations=2,
            fixed_effects=jnp.array([0.4, -0.2]),
            raw_variances=raw,
            score_center=0.0,
            score_scale=1.0,
        )
    # One query's marginal Gaussian variance is 2u+e; no covariance fitting occurs.
    expected = 0.5 * (np.log(2 * np.pi * 1.32) + (0.3 - 0.36) ** 2 / 1.32)
    assert actual == pytest.approx(expected, abs=1e-12)
