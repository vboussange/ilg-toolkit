"""Tests of the private maximum-likelihood population effects (MLPE) likelihood kernels.

The tests cover profile gradients at extreme variance ratios and the float32
precision guard; public tests cover likelihood values, coefficients and posteriors.
"""

import jax
import jax.numpy as jnp
import numpy as np
from ilg_toolkit import MLPELoss
from ilg_toolkit.mlpe.likelihood import _profiled_mlpe_ml_fit


# The kernels take MLPELoss's numerical settings explicitly.
_SETTINGS = dict(
    variance_floor=MLPELoss().variance_floor, min_distance_scale=MLPELoss().min_distance_scale
)


def test_profile_gradients_agree_with_dense_extreme_variance_reference(
    dense_jax_profile, decimal_dense_profile_and_gradients
):
    rng = np.random.default_rng(12)
    left, right = np.triu_indices(6, 1)
    # Three absent pairs make the population design incomplete.
    order = rng.permutation(len(left))[3:]
    left, right = right[order], left[order]  # Reversed, shuffled endpoints.
    distances = rng.normal(size=len(left))
    targets = 0.5 - 0.4 * distances + rng.normal(scale=0.3, size=len(left))
    s, y = jnp.asarray(distances), jnp.asarray(targets)

    def accelerated(s, raw):
        return _profiled_mlpe_ml_fit(
            s, y, left, right, n_populations=6, raw_variances=raw, jitter=0.0, **_SETTINGS
        )[0]

    def dense(s, raw):
        return dense_jax_profile(s, y, left, right, 6, raw)[0]

    evaluate = jax.jit(jax.grad(accelerated, argnums=(0, 1)))
    reference = jax.jit(jax.grad(dense, argnums=(0, 1)))
    for variances in ((1e-8, 1.0), (1.0, 1.0), (1.0, 1e-8)):
        raw = jnp.asarray(np.log(np.expm1(np.asarray(variances) - 1e-10)))
        actual_gradients = evaluate(s, raw)
        dense_gradients = reference(s, raw)
        # At u/e=1e8 the dense float64 unit-variance gradient subtracts terms of
        # order 1/e^2. The 60-digit dense oracle resolves that cancellation.
        if variances[1] < 1e-7:
            _, _, dense_gradients = decimal_dense_profile_and_gradients(
                distances, targets, left, right, np.asarray(raw)
            )
        for actual, expected in zip(actual_gradients, dense_gradients, strict=True):
            np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=5e-7)


def test_unsupported_float32_precision_gives_nonfinite_diagnostics():
    left, right = np.triu_indices(4, 1)
    distances = np.arange(6, dtype=np.float32)
    with jax.enable_x64(False):
        raw = jnp.asarray(np.log(np.expm1(np.array([1.0, 1e-8]) - 1e-10)), dtype=jnp.float32)

        def profiled(s):
            return _profiled_mlpe_ml_fit(
                s,
                jnp.asarray(distances),
                left,
                right,
                n_populations=4,
                raw_variances=raw,
                jitter=0.0,
                **_SETTINGS,
            )

        compiled, compiled_beta = jax.jit(profiled)(jnp.asarray(distances))
    assert not np.isfinite(compiled)
    assert not np.isfinite(compiled_beta).any()
