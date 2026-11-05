"""Tests of the summed training objective, the Laplace approximation and the Newton step."""

import itertools
import logging
import math
from types import SimpleNamespace

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from ilg_toolkit import (
    laplace_approximation,
    Log1pLoss,
    MLPELoss,
    newton_step,
    PairwiseObservations,
    RegionBatch,
    ResistanceSolverConfig,
    RidgePenalty,
    summed_training_objective,
    TargetSpec,
    train,
    TrainingConfig,
)
from ilg_toolkit.models import LogLinearConductance


# Central differences of the gradient resolve the solver's error divided by their step.
_TIGHT_SOLVER = ResistanceSolverConfig(rtol=1e-10, atol=1e-10, max_steps=5000)
_RIDGE = RidgePenalty(prior_sd=1.0, where=lambda model: model.coefficients)
_PINNED_INTERCEPT = 1.0


def _coefficients(model: LogLinearConductance) -> jax.Array:
    """Selects the coefficients of a log-linear patch encoder, leaving its intercept fixed."""
    return model.coefficients


def _penalty(model: LogLinearConductance) -> jnp.ndarray:
    """Returns the ridge penalty on the coefficients plus a pin of the intercept."""
    return _RIDGE(model) + jnp.sum((model.intercepts - _PINNED_INTERCEPT) ** 2) / (2 * 0.1**2)


def _smooth_field(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    field = rng.normal(size=shape)
    for axis in (0, 1):
        field = (field + np.roll(field, 1, axis) + np.roll(field, -1, axis)) / 3
    return field / field.std()


def _region_and_observations() -> tuple[RegionBatch, PairwiseObservations]:
    """Returns a small region whose dissimilarities follow MLPE on a log-linear surface."""
    rng = np.random.default_rng(1)
    shape = (6, 7)
    features = np.stack([_smooth_field(rng, shape), _smooth_field(rng, shape)], axis=-1)
    positions = np.array([[0, 0], [0, 6], [5, 0], [5, 6], [2, 2], [3, 5], [1, 3], [4, 3]])
    units = tuple(f"unit-{index}" for index in range(len(positions)))
    region = RegionBatch(
        "valley",
        cx.field(
            features, "row", "column", cx.LabeledAxis("feature", np.array(["elevation", "canopy"]))
        ),
        units,
        positions,
    )
    truth = eqx.tree_at(
        lambda model: model.coefficients,
        LogLinearConductance(2, patch_size=1, solver_config=_TIGHT_SOLVER),
        jnp.asarray([[1.0, -0.6]]),
    )
    pairs = list(itertools.combinations(units, 2))
    resistance = np.asarray(truth.predict(region, pairs).values)
    effects = dict(zip(units, rng.normal(scale=0.03, size=len(units)), strict=True))
    values = (
        0.2
        + 0.06 * (resistance - resistance.mean()) / resistance.std(ddof=1)
        + np.array([effects[a] + effects[b] for a, b in pairs])
        + rng.normal(scale=0.01, size=len(pairs))
    )
    observations = PairwiseObservations.from_pairs(
        pairs,
        values,
        target=TargetSpec("synthetic divergence", units="index"),
        sampling_unit_ids=units,
    )
    return region, observations


@pytest.fixture(scope="module")
def penalized_fit() -> SimpleNamespace:
    """Returns a penalized fit of the log-linear patch encoder with MLPE, trained to convergence."""
    region, observations = _region_and_observations()
    start = eqx.tree_at(
        lambda model: model.intercepts,
        LogLinearConductance(2, patch_size=1, solver_config=_TIGHT_SOLVER),
        jnp.asarray([_PINNED_INTERCEPT]),
    )
    config = TrainingConfig(
        loss=MLPELoss(),
        steps=100,
        eval_every=100,
        optimizer=optax.lbfgs(),
        gradient_tolerance=1e-9,
        penalty=_penalty,
    )
    state = train(region, observations, distance_model=start, config=config)
    assert state.converged
    approximation = laplace_approximation(region, observations, state, where=_coefficients)
    return SimpleNamespace(
        region=region, observations=observations, state=state, approximation=approximation
    )


def _newton_step(
    fit: SimpleNamespace,
    distance_model: LogLinearConductance,
    observations: PairwiseObservations | None = None,
) -> jax.Array:
    """Returns the Newton step of a distance model, with the fit's statistical parameters.

    The observations are the fit's unless given.
    """
    return newton_step(
        fit.region,
        fit.observations if observations is None else observations,
        distance_model,
        fit.state.loss_state,
        config=fit.state.config,
        where=_coefficients,
        hessian=fit.approximation.hessian,
    )


def test_summed_training_objective_is_the_pair_count_times_the_training_objective(penalized_fit):
    state = penalized_fit.state
    value = summed_training_objective(
        penalized_fit.region,
        penalized_fit.observations,
        state.latest_model.distance_model,
        state.loss_state,
        config=state.config,
    )
    pair_count = len(penalized_fit.observations.observed_pairs)
    assert float(value) == pytest.approx(
        pair_count * state.history[-1].training_objective, rel=1e-12
    )


def test_where_nothing_is_profiled_the_hessian_is_that_of_automatic_differentiation(
    WeightedEmbedding, embedding_problem
):
    region, observations = embedding_problem()
    start = WeightedEmbedding(jnp.asarray([0.3, -0.2]))
    penalty = RidgePenalty(prior_sd=0.5, where=lambda model: model.log_weights)
    config = TrainingConfig(loss=Log1pLoss(), steps=0, penalty=penalty)
    state = train(region, observations, distance_model=start, config=config)

    approximation = laplace_approximation(
        region, observations, state, where=lambda model: model.log_weights
    )

    def objective(log_weights):
        model = eqx.tree_at(lambda current: current.log_weights, start, log_weights)
        return summed_training_objective(region, observations, model, None, config=config)

    expected = jax.hessian(objective)(start.log_weights)
    assert approximation.parameter_names == ("log_weights[0]", "log_weights[1]")
    np.testing.assert_allclose(approximation.hessian, expected, rtol=1e-7)


def test_where_the_hessian_is_not_positive_definite_no_standard_deviation_is_a_number(
    ScalarEmbedding, regions_and_observations
):
    regions, observations = regions_and_observations()
    config = TrainingConfig(loss=Log1pLoss(), steps=0)
    state = train(
        regions[0], observations[0], distance_model=ScalarEmbedding(jnp.asarray(0.0)), config=config
    )

    approximation = laplace_approximation(
        regions[0], observations[0], state, where=lambda model: model.weight
    )

    # At weight 0, the squared distances 1, 4 and 1 vanish, and the summed loss
    # sum((log1p(w**2 * d) - log1p(0.1)) ** 2) has second derivative -4 * log1p(0.1) * 6.
    assert approximation.hessian[0, 0] == pytest.approx(-24 * math.log1p(0.1), rel=1e-6)
    assert not approximation.positive_definite
    assert np.isnan(approximation.standard_deviations).all()


def test_at_a_converged_penalized_fit_the_hessian_is_positive_definite_and_the_step_small(
    penalized_fit,
):
    approximation = penalized_fit.approximation
    assert approximation.positive_definite
    assert np.all(np.linalg.eigvalsh(approximation.hessian) > 0)
    np.testing.assert_allclose(
        approximation.covariance @ approximation.hessian, np.eye(4), atol=1e-10
    )
    np.testing.assert_allclose(
        approximation.standard_deviations, np.sqrt(np.diag(approximation.covariance))
    )

    step = _newton_step(penalized_fit, penalized_fit.state.latest_model.distance_model)

    assert step.shape == (4,)
    assert np.abs(step / approximation.standard_deviations).max() < 1e-6


def test_from_a_perturbed_model_the_newton_step_points_back_to_the_fit(penalized_fit):
    fit = penalized_fit.state.latest_model.distance_model
    deviations = penalized_fit.approximation.standard_deviations
    perturbation = jnp.asarray([0.1, -0.1]) * deviations[:2]
    perturbed = eqx.tree_at(_coefficients, fit, fit.coefficients + perturbation)

    step = _newton_step(penalized_fit, perturbed)

    # Newton's method leaves an error of second order in the perturbation.
    remaining = (np.concatenate([perturbation, np.zeros(2)]) + step) / deviations
    assert np.abs(remaining).max() < 0.02


def test_leaves_the_selector_does_not_pick_are_no_parameters_and_keep_their_values(
    penalized_fit,
):
    approximation = penalized_fit.approximation
    with_intercept = laplace_approximation(
        penalized_fit.region,
        penalized_fit.observations,
        penalized_fit.state,
        where=lambda model: (model.coefficients, model.intercepts),
    )

    assert approximation.parameter_names == (
        "coefficients[0, 0]",
        "coefficients[0, 1]",
        "raw_variances[0]",
        "raw_variances[1]",
    )
    assert with_intercept.parameter_names == (
        "coefficients[0, 0]",
        "coefficients[0, 1]",
        "intercepts[0]",
        "raw_variances[0]",
        "raw_variances[1]",
    )
    # Holding the intercept at its fitted value gives the same second derivatives as
    # differentiating through it at that value.
    kept = [0, 1, 3, 4]
    np.testing.assert_allclose(
        approximation.hessian, with_intercept.hessian[np.ix_(kept, kept)], rtol=1e-7, atol=1e-7
    )


def test_several_regions_raise(ScalarEmbedding, regions_and_observations):
    regions, observations = regions_and_observations()
    model = ScalarEmbedding(jnp.asarray(1.0))
    config = TrainingConfig(loss=Log1pLoss(), steps=0)
    state = train(regions, observations, distance_model=model, config=config)

    with pytest.raises(ValueError, match="one region–species group"):
        laplace_approximation(regions, observations, state, where=lambda model: model.weight)
    with pytest.raises(ValueError, match="one region–species group"):
        laplace_approximation(regions[0], observations[0], state, where=lambda model: model.weight)
    with pytest.raises(ValueError, match="one region–species group"):
        newton_step(
            regions,
            observations,
            model,
            None,
            config=config,
            where=lambda model: model.weight,
            hessian=jnp.eye(1),
        )


def test_several_species_raise(training_problem):
    region, observations, model = training_problem("embedding", species=("frog", "newt"))
    config = TrainingConfig(loss=Log1pLoss(), steps=0)
    state = train(region, observations, distance_model=model, config=config)

    with pytest.raises(ValueError, match="one region–species group"):
        laplace_approximation(region, observations, state, where=lambda model: model.log_weights)
    with pytest.raises(ValueError, match="one region–species group"):
        newton_step(
            region,
            observations,
            model,
            None,
            config=config,
            where=lambda model: model.log_weights,
            hessian=jnp.eye(4),
        )


def test_a_failed_solve_raises_the_failure_that_training_raises(PointConductance):
    region = RegionBatch(
        "bad-solve",
        cx.field(
            np.linspace(0.1, 2, 12).reshape(3, 4, 1),
            "row",
            "column",
            cx.LabeledAxis("feature", np.array(["elevation"])),
        ),
        ("first", "middle", "last"),
        np.array([[0, 0], [1, 2], [2, 3]]),
    )
    observations = PairwiseObservations.from_pairs(
        [("first", "middle"), ("first", "last"), ("middle", "last")],
        [0.2, 0.5, 0.4],
        target=TargetSpec("synthetic divergence", units="index"),
    )
    config = TrainingConfig(loss=Log1pLoss(), steps=0)
    state = train(
        region, observations, distance_model=PointConductance(jnp.asarray(0.5)), config=config
    )
    # Conjugate gradient runs out of steps at every conductance surface.
    exhausted = PointConductance(
        jnp.asarray(0.5), solver_config=ResistanceSolverConfig(rtol=1e-12, atol=1e-12, max_steps=1)
    )
    state = eqx.tree_at(lambda current: current.latest_model.distance_model, state, exhausted)
    reason = "The maximum number of solver steps was reached"

    with pytest.raises(RuntimeError) as caught:
        laplace_approximation(region, observations, state, where=lambda model: model.log_scale)
    assert caught.type is RuntimeError
    assert str(caught.value).startswith(
        f"region 'bad-solve', species 0: Laplace approximation: {reason}"
    )
    with pytest.raises(RuntimeError) as caught:
        newton_step(
            region,
            observations,
            exhausted,
            None,
            config=config,
            where=lambda model: model.log_scale,
            hessian=jnp.eye(1),
        )
    assert caught.type is RuntimeError
    assert str(caught.value).startswith(f"region 'bad-solve', species 0: Newton step: {reason}")


def test_a_later_fold_on_as_many_pairs_compiles_nothing(penalized_fit, caplog):
    observations = penalized_fit.observations
    # Two population-holdout folds, each without one sampling unit's pairs.
    first, second = (
        observations.select_pairs(
            [pair for pair in observations.observed_pairs if unit not in pair]
        )
        for unit in penalized_fit.region.sampling_unit_ids[:2]
    )
    fit = penalized_fit.state.latest_model.distance_model
    _newton_step(penalized_fit, fit, first)

    with jax.log_compiles(), caplog.at_level(logging.WARNING):
        _newton_step(penalized_fit, fit, second)

    assert not [record for record in caplog.records if "Compiling" in record.getMessage()]
