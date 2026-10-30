"""Tests of penalties: the ridge penalty's value, leaf selection and arguments."""

from typing import Any

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
    value = RidgePenalty(prior_sd=2.0, where=where)(_distance_model())
    assert value.shape == ()
    assert float(value) == expected


@pytest.mark.parametrize("prior_sd", [0, -1.0, float("nan"), float("inf"), True, "1.5", None])
def test_ridge_penalty_rejects_a_prior_sd_that_is_not_finite_and_positive(prior_sd: Any):
    with pytest.raises(ValueError, match="prior_sd"):
        RidgePenalty(prior_sd=prior_sd, where=lambda model: model.coefficients)


@pytest.mark.parametrize("where", [None, "coefficients", 3])
def test_ridge_penalty_rejects_a_where_that_is_not_callable(where: Any):
    with pytest.raises(ValueError, match="where"):
        RidgePenalty(prior_sd=1.0, where=where)


def test_ridge_penalty_stores_prior_sd_as_a_float():
    penalty = RidgePenalty(prior_sd=np.float64(2.0), where=lambda model: model.coefficients)
    assert type(penalty.prior_sd) is float
