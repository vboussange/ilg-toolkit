"""Tests of penalties: the ridge penalty's value, gradient and arguments."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from ilg_toolkit import RidgePenalty
from ilg_toolkit.models import LogLinearConductance


def _distance_model() -> LogLinearConductance:
    """Returns a log-linear patch encoder with coefficients (3, 4) and intercept 5."""
    return eqx.tree_at(
        lambda model: (model.coefficients, model.intercepts),
        LogLinearConductance(2, patch_size=1),
        (jnp.asarray([[3.0, 4.0]]), jnp.asarray([5.0])),
    )


@pytest.mark.parametrize(
    ("where", "expected"),
    [
        (lambda model: model.coefficients, 25 / 8),
        # The selector reaches slices and several fields.
        (lambda model: model.coefficients[:, 1:], 16 / 8),
        (lambda model: (model.coefficients, model.intercepts), 50 / 8),
        # An empty selection is a zero penalty.
        (lambda model: (), 0.0),
    ],
    ids=["field", "slice", "fields", "empty"],
)
def test_ridge_penalty_is_the_gaussian_negative_log_density_without_its_constant(where, expected):
    assert float(RidgePenalty(prior_sd=2.0, where=where)(_distance_model())) == expected


def test_ridge_gradient_is_the_selected_leaf_over_the_squared_prior_sd():
    penalty = RidgePenalty(prior_sd=2.0, where=lambda model: model.coefficients[:, 1:])
    gradient = eqx.filter_grad(penalty)(_distance_model())
    np.testing.assert_array_equal(gradient.coefficients, [[0.0, 4 / 4]])
    np.testing.assert_array_equal(gradient.intercepts, [0.0])


@pytest.mark.parametrize("prior_sd", [0, -1.0, float("nan"), float("inf")])
def test_ridge_penalty_rejects_a_prior_sd_that_is_not_finite_and_positive(prior_sd):
    with pytest.raises(ValueError, match="prior_sd"):
        RidgePenalty(prior_sd=prior_sd, where=lambda model: model.coefficients)
