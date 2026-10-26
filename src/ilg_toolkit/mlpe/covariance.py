"""Population-sized covariance calculations for pairwise genetic observations."""

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import cho_solve

# The maximum number of observed pairs per sampling unit bounds
# eps*cond(I + u/d Z.T Z), including incomplete and bipartite pair sets.
# Calculations outside this precision bound return nonfinite results.
MAX_ROUNDOFF_BOUND = 0.01


def endpoint_sums(values, left, right, weights, n_populations):
    """Sums of pairwise values for each sampling unit."""
    values = jnp.asarray(values)
    weighted = jnp.where(weights[:, None], values, 0)
    result = jnp.zeros((n_populations, values.shape[1]), dtype=values.dtype)
    return result.at[left].add(weighted).at[right].add(weighted)


class PopulationCovarianceFactor(NamedTuple):
    """Factor of I + (u/d) Z.T Z and the observed sampling-unit indices.

    For V = d I + u Z Z.T, the determinant lemma gives
    logdet(V) = m log(d) + logdet(I + (u/d) Z.T Z). For columns B,
    A = (u/d) K^-1 Z.T B and R = B - Z A give the positive identity
    B.T V^-1 B = R.T R / d + A.T A / u. This avoids cancellation from
    subtracting large, nearly equal terms in the Woodbury form. Accumulation
    uses O(n^2 + m) storage and the population-sized factorization O(n^3) work.
    """

    left: Array
    right: Array
    weights: Array
    factor: Array
    unit: Array
    residual: Array
    pair_count: Array
    logdet: Array
    valid: Array
    identifiable: Array

    def residual_components(self, values):
        """Positive representation for B.T V^-1 C; stable against subtraction."""
        values = jnp.where(self.weights[:, None], values, 0)
        sums = endpoint_sums(values, self.left, self.right, self.weights, len(self.factor))
        effects = (self.unit / self.residual) * cho_solve((self.factor, True), sums)
        remaining = jnp.where(
            self.weights[:, None], values - effects[self.left] - effects[self.right], 0
        )
        return remaining, effects

    def inner_products(self, values):
        """Covariance-weighted inner products, residuals, and population effects."""
        remaining, effects = self.residual_components(values)
        inner = remaining.T @ remaining / self.residual + effects.T @ effects / self.unit
        return inner, remaining, effects


def population_covariance_factor(left, right, *, n_populations, unit, residual, weights, dtype):
    """Population covariance factor for observed unordered pairs.

    ``residual`` includes the configured jitter.
    The observation mask selects the sampling-unit indices used in the calculation.
    """
    unit, residual = jnp.asarray(unit, dtype=dtype), jnp.asarray(residual, dtype=dtype)
    left, right = jnp.asarray(left, dtype=jnp.int32), jnp.asarray(right, dtype=jnp.int32)
    valid_endpoints = jnp.all(
        (~weights)
        | (
            (left >= 0)
            & (left < n_populations)
            & (right >= 0)
            & (right < n_populations)
            & (left != right)
        )
    )
    left, right = jnp.where(weights, left, 0), jnp.where(weights, right, 0)
    value = weights.astype(dtype)
    gram = jnp.zeros((n_populations, n_populations), dtype=dtype)
    gram = gram.at[left, left].add(value).at[right, right].add(value)
    gram = gram.at[left, right].add(value).at[right, left].add(value)
    ratio = unit / residual
    matrix = jnp.eye(n_populations, dtype=dtype) + ratio * gram
    factor = jnp.linalg.cholesky(matrix)
    count = jnp.sum(value)
    logdet = count * jnp.log(residual) + 2 * jnp.log(jnp.diag(factor)).sum()
    roundoff_bound = jnp.finfo(dtype).eps * (1 + 2 * ratio * jnp.max(jnp.diag(gram)))
    valid = (
        valid_endpoints
        & jnp.isfinite(factor).all()
        & jnp.isfinite(unit)
        & (unit > 0)
        & jnp.isfinite(residual)
        & (residual > 0)
        & (roundoff_bound <= MAX_ROUNDOFF_BOUND)
    )
    return PopulationCovarianceFactor(
        left,
        right,
        weights,
        factor,
        unit,
        residual,
        count,
        logdet,
        valid,
        jnp.max(jnp.diag(gram)) > 1,
    )
