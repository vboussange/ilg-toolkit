"""Tests of the population-sized covariance algebra's precision limit."""

import jax.numpy as jnp
import numpy as np
import pytest
from ilg_toolkit.mlpe.covariance import _max_log_variance_ratio, _population_covariance_factor


@pytest.mark.parametrize("residual", [1e-10, 1e-3, 1.0, 1e3])
def test_variance_ratio_limit_is_the_edge_of_the_valid_factorizations(residual):
    left, right = np.triu_indices(8, 1)
    limit = _max_log_variance_ratio(7, float(jnp.finfo(jnp.float64).eps))

    def valid(log_ratio):
        return _population_covariance_factor(
            left,
            right,
            n_populations=8,
            unit=residual * np.exp(log_ratio),
            residual=residual,
            dtype=jnp.float64,
        ).valid

    assert valid(limit)
    assert not valid(limit + 1e-6)
