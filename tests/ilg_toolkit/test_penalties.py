"""Tests of penalties: the ridge penalty's value, gradient, arguments and training."""

from dataclasses import replace

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from ilg_toolkit import Log1pLoss, RidgePenalty, train, TrainingConfig
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
        # A selected subtree takes one prior standard deviation for all its leaves.
        (lambda model: model, 50 / 8),
        # An empty selection is a zero penalty.
        (lambda model: (), 0.0),
    ],
    ids=["field", "slice", "fields", "subtree", "empty"],
)
def test_a_default_ridge_penalty_is_a_zero_mean_gaussian_with_one_prior_sd(where, expected):
    assert float(RidgePenalty(prior_sd=2.0, where=where)(_distance_model())) == expected


@pytest.mark.parametrize(
    ("prior_sd", "where", "prior_mean", "expected"),
    [
        # (5 - 1)^2 / (2 * 2^2)
        (2.0, lambda model: model.intercepts, 1.0, 16 / 8),
        # ((3 - 1)^2 + (4 - 1)^2) / (2 * 2^2) + (5 - 4)^2 / (2 * 0.5^2)
        (
            (2.0, 0.5),
            lambda model: (model.coefficients, model.intercepts),
            (1.0, 4.0),
            13 / 8 + 2,
        ),
        # Each prior mean is broadcast over its leaf's entries.
        (2.0, lambda model: model.coefficients, 1.0, 13 / 8),
        # One number serves every leaf, given alone or in a tuple of one.
        ((2.0, 0.5), lambda model: (model.coefficients, model.intercepts), 1.0, 13 / 8 + 32),
        (2.0, lambda model: (model.coefficients, model.intercepts), (1.0, 4.0), 13 / 8 + 1 / 8),
        ((2.0,), lambda model: (model.coefficients, model.intercepts), (1.0,), 13 / 8 + 16 / 8),
    ],
    ids=[
        "one-leaf",
        "several-leaves",
        "broadcast-prior-mean",
        "one-prior-mean",
        "one-prior-sd",
        "tuples-of-one",
    ],
)
def test_ridge_penalty_is_the_gaussian_negative_log_density_about_its_prior_means(
    prior_sd, where, prior_mean, expected
):
    penalty = RidgePenalty(prior_sd=prior_sd, where=where, prior_mean=prior_mean)
    assert float(penalty(_distance_model())) == expected


def test_ridge_gradient_is_the_selected_leaf_over_the_squared_prior_sd():
    penalty = RidgePenalty(prior_sd=2.0, where=lambda model: model.coefficients[:, 1:])
    gradient = eqx.filter_grad(penalty)(_distance_model())
    np.testing.assert_array_equal(gradient.coefficients, [[0.0, 4 / 4]])
    np.testing.assert_array_equal(gradient.intercepts, [0.0])


@pytest.mark.parametrize("in_tuple", [False, True], ids=["number", "tuple"])
@pytest.mark.parametrize("prior_sd", [0, -1.0, float("nan"), float("inf")])
def test_ridge_penalty_rejects_a_prior_sd_that_is_not_finite_and_positive(prior_sd, in_tuple):
    with pytest.raises(ValueError, match="prior_sd must be finite and positive"):
        RidgePenalty(
            prior_sd=(1.0, prior_sd) if in_tuple else prior_sd,
            where=lambda model: (model.coefficients, model.intercepts),
        )


@pytest.mark.parametrize("in_tuple", [False, True], ids=["number", "tuple"])
@pytest.mark.parametrize("prior_mean", [float("nan"), float("inf"), -float("inf")])
def test_ridge_penalty_rejects_a_prior_mean_that_is_not_finite(prior_mean, in_tuple):
    with pytest.raises(ValueError, match="prior_mean must be finite"):
        RidgePenalty(
            prior_sd=1.0,
            where=lambda model: (model.coefficients, model.intercepts),
            prior_mean=(0.0, prior_mean) if in_tuple else prior_mean,
        )


@pytest.mark.parametrize(
    ("numbers", "where", "message"),
    [
        (
            {"prior_sd": (1.0, 2.0)},
            lambda model: model.coefficients,
            "prior_sd has 2 numbers for the 1 leaves the selector selects",
        ),
        (
            {"prior_sd": 1.0, "prior_mean": (0.0, 1.0)},
            lambda model: (model.coefficients, model.intercepts, model.coefficients),
            "prior_mean has 2 numbers for the 3 leaves the selector selects",
        ),
        (
            {"prior_sd": (1.0, 2.0), "prior_mean": (0.0, 1.0, 2.0)},
            lambda model: (model.coefficients, model.intercepts),
            "prior_mean has 3 numbers for the 2 leaves the selector selects",
        ),
    ],
    ids=["prior-sd", "prior-mean", "both"],
)
def test_ridge_penalty_rejects_a_selection_of_another_number_of_leaves(numbers, where, message):
    penalty = RidgePenalty(where=where, **numbers)
    with pytest.raises(ValueError, match=message):
        penalty(_distance_model())


def test_training_keeps_a_pinned_leaf_within_a_few_standard_deviations_of_its_prior_mean(
    ScalarEmbedding, regions_and_observations
):
    regions, observations = regions_and_observations()
    # The weight starts at 0, a hundred prior standard deviations below its prior mean.
    penalty = RidgePenalty(prior_sd=0.01, where=lambda model: model.weight, prior_mean=1.0)
    state = train(
        regions[0],
        observations[0],
        distance_model=ScalarEmbedding(jnp.asarray(0.0)),
        config=TrainingConfig(loss=Log1pLoss(), steps=5, optimizer=optax.lbfgs(), penalty=penalty),
    )
    np.testing.assert_allclose(state.latest_model.distance_model.weight, 1.0, atol=3 * 0.01)


def test_a_second_ridge_penalty_with_equal_numbers_reuses_the_compiled_update(
    ScalarEmbedding, regions_and_observations
):
    regions, observations = regions_and_observations()
    traces = []

    def where(model):
        traces.append(None)
        return (model.weight,)

    config = TrainingConfig(
        loss=Log1pLoss(),
        steps=2,
        optimizer=optax.adam(0.1),
        penalty=RidgePenalty(prior_sd=(0.5,), where=where, prior_mean=(2,)),
    )
    model = ScalarEmbedding(jnp.asarray(1.0))
    train(regions, observations, distance_model=model, config=config)
    assert traces
    traces.clear()
    # A number and a tuple of one number are equal numbers.
    second = RidgePenalty(prior_sd=np.float64(0.5), where=where, prior_mean=2.0)
    train(regions, observations, distance_model=model, config=replace(config, penalty=second))
    assert not traces
