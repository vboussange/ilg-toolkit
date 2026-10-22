"""Scientific behavior of mlpe.likelihood."""

import numpy as np
import pytest

from ilg_toolkit.mlpe import profiled_mlpe_ml_fit


@pytest.mark.parametrize("jit", [False, True])
def test_complete_mlpe_finalization_matches_incomplete_pair_dense_gaussian(dense_profile, jit):
    import jax
    import jax.numpy as jnp

    from ilg_toolkit.mlpe import finalize_mlpe_fit

    distances = np.array([0.2, 1.1, -0.3, 2.0])
    targets = np.array([0.5, 0.1, 0.9, 0.3])
    left, right = np.array([3, 0, 2, 0]), np.array([1, 2, 1, 1])
    variances, jitter = np.array([0.6, 0.12]), 1e-7
    expected_nll, beta, expected_mean, expected_covariance = dense_profile(
        distances, targets, left, right, 4, variances, jitter=jitter
    )
    with jax.enable_x64():
        raw = jnp.log(jnp.expm1(jnp.asarray(variances) - 1e-10))

        def finalize():
            return finalize_mlpe_fit(
                distances, targets, left, right, n_populations=4, raw_variances=raw, jitter=jitter
            )

        result = jax.jit(finalize)() if jit else finalize()
    mean, posterior, factor = result.posterior
    assert result.nll == pytest.approx(expected_nll, abs=1e-12)
    np.testing.assert_allclose(result.coefficients, beta, atol=1e-12)
    np.testing.assert_allclose(result.variances, variances, atol=1e-15)
    assert result.distance_center == pytest.approx(distances.mean(), abs=1e-15)
    assert result.distance_scale == pytest.approx(distances.std(ddof=1), abs=1e-15)
    np.testing.assert_allclose(mean, expected_mean, atol=1e-12)
    np.testing.assert_allclose(posterior, expected_covariance, atol=1e-12)
    np.testing.assert_allclose(factor @ factor.T @ posterior, np.eye(4), atol=1e-12)


def test_masked_pairs_are_absent_from_likelihood_and_distance_standardization_under_jit(
    dense_profile,
):
    import jax
    import jax.numpy as jnp

    distances = np.array([0.2, np.nan, 1.1, -0.3, 2.0, np.inf])
    targets = np.array([0.5, np.nan, 0.1, 0.9, 0.3, np.inf])
    left, right = np.array([3, -999, 0, 2, 0, 999]), np.array([1, 999, 2, 1, 1, -999])
    mask = np.array([True, False, True, True, True, False])
    raw = np.array([-0.7, -1.4])
    expected_nll, expected_beta, _, _ = dense_profile(
        distances[mask], targets[mask], left[mask], right[mask], 4, np.logaddexp(0, raw) + 1e-10
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

        actual = objective(jnp.asarray(distances))
        compiled = jax.jit(objective)(jnp.asarray(distances))
        gradient = jax.grad(lambda s: objective(s)[0])(jnp.asarray(distances))
    np.testing.assert_allclose([actual[0], *actual[1]], [expected_nll, *expected_beta], atol=1e-10)
    np.testing.assert_allclose([compiled[0], *compiled[1]], [actual[0], *actual[1]], atol=1e-10)
    assert np.isfinite(gradient).all()
    np.testing.assert_array_equal(np.asarray(gradient)[~mask], 0)


@pytest.mark.parametrize("incomplete", [False, True])
@pytest.mark.parametrize("variances", [(1e-8, 1.0), (1.0, 1.0), (1.0, 1e-8)])
def test_endpoint_likelihood_gls_and_gradients_agree_with_dense_extreme_variance_reference(
    decimal_dense_profile_and_gradients, dense_jax_profile, incomplete, variances
):
    import jax
    import jax.numpy as jnp

    rng = np.random.default_rng(12)
    left, right = np.triu_indices(6, 1)
    order = rng.permutation(len(left))
    if incomplete:
        order = order[3:]
    left, right = right[order], left[order]  # Reversed, shuffled endpoints.
    distances = rng.normal(size=len(left))
    targets = 0.5 - 0.4 * distances + rng.normal(scale=0.3, size=len(left))
    raw = np.log(np.expm1(np.asarray(variances) - 1e-10))
    with jax.enable_x64():
        s, y, raw = jnp.asarray(distances), jnp.asarray(targets), jnp.asarray(raw)

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
            distances, targets, left, right, np.asarray(raw)
        )
    for actual, expected in zip(actual_gradients, dense_gradients, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=5e-7)


def test_unsupported_precision_and_singular_distances_have_explicit_nonfinite_diagnostics():
    import jax
    import jax.numpy as jnp

    # Low-level numerical kernels preserve their documented nonfinite-result
    # diagnostic contract; the public calibration/head boundaries reject it.
    left, right = np.triu_indices(4, 1)
    distances = np.arange(6, dtype=np.float32)
    with jax.enable_x64(False):
        raw = jnp.asarray(np.log(np.expm1(np.array([1.0, 1e-8]) - 1e-10)), dtype=jnp.float32)

        def objective(s):
            return profiled_mlpe_ml_fit(
                s,
                jnp.asarray(distances),
                left,
                right,
                n_populations=4,
                raw_variances=raw,
            )

        eager, beta = objective(jnp.asarray(distances))
        compiled, compiled_beta = jax.jit(objective)(jnp.asarray(distances))
    assert not np.isfinite(eager)
    assert not np.isfinite(compiled)
    assert not np.isfinite(beta).any()
    assert not np.isfinite(compiled_beta).any()
    with jax.enable_x64():
        nll, beta = profiled_mlpe_ml_fit(
            np.ones(6), distances, left, right, n_populations=4, raw_variances=jnp.zeros(2)
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
            distance_center=0.0,
            distance_scale=1.0,
        )
    # One query's marginal Gaussian variance is 2u+e; no covariance fitting occurs.
    expected = 0.5 * (np.log(2 * np.pi * 1.32) + (0.3 - 0.36) ** 2 / 1.32)
    assert actual == pytest.approx(expected, abs=1e-12)
