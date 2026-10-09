"""Exact endpoint-system arithmetic without pair-sized covariance allocation."""

from typing import NamedTuple

import jax.numpy as jnp
from jax.scipy.linalg import cho_solve

# A conservative precision bound protects incomplete/bipartite endpoint systems.
# It bounds eps*cond(I + u/d Z.T Z) using the maximum endpoint degree. No
# variance or diagonal is altered when the arithmetic lies outside this domain.
MAX_ROUNDOFF_BOUND = 0.01


def endpoint_sums(values, left, right, weights, n_populations):
    """Scatter Z.T values without materializing the pair-to-unit incidence."""
    values = jnp.asarray(values)
    weighted = jnp.where(weights[:, None], values, 0)
    result = jnp.zeros((n_populations, values.shape[1]), dtype=values.dtype)
    return result.at[left].add(weighted).at[right].add(weighted)


class PopulationSystem(NamedTuple):
    """Factor of I + (u/d) Z.T Z and its observed endpoint representation."""

    left: object
    right: object
    weights: object
    factor: object
    unit: object
    residual: object
    count: object
    logdet: object
    valid: object
    identifiable: object

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
        remaining, effects = self.residual_components(values)
        inner = remaining.T @ remaining / self.residual + effects.T @ effects / self.unit
        return inner, remaining, effects


def population_system(left, right, *, n_populations, unit, residual, weights, dtype):
    """Build and factor an n*n system for any observed unordered pair set.

    ``residual`` already includes the caller's explicitly declared jitter.
    Masked indices are sanitized and never used to index a sampling unit.
    """
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
    return PopulationSystem(
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
