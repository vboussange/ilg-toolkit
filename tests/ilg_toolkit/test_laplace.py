"""Tests of the summed training objective, the Laplace approximation and the Newton step."""

import itertools
import logging
import math
from typing import NamedTuple

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from ilg_toolkit import (
    laplace_approximation,
    LaplaceApproximation,
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
    TrainingState,
)
from ilg_toolkit.models import LogLinearConductance


# Central differences of the gradient resolve the solver's error divided by their step.
_TIGHT_SOLVER = ResistanceSolverConfig(rtol=1e-10, atol=1e-10, max_steps=5000)
_GENERATING_COEFFICIENTS = jnp.asarray([[1.0, -0.6]])
_PINNED_INTERCEPT = 1.0
# A ridge on the coefficients and a pin of the intercept.
_PENALTY = RidgePenalty(
    prior_sd=(1.0, 0.1),
    where=lambda model: (model.coefficients, model.intercepts),
    prior_mean=(0.0, _PINNED_INTERCEPT),
)


def _coefficients(model: LogLinearConductance) -> jax.Array:
    """Selects the coefficients of a log-linear patch encoder, leaving its intercept fixed."""
    return model.coefficients


def _smooth_field(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    field = rng.normal(size=shape)
    for axis in (0, 1):
        field = (field + np.roll(field, 1, axis) + np.roll(field, -1, axis)) / 3
    return field / field.std()


def _region_and_observations() -> tuple[RegionBatch, PairwiseObservations]:
    """Returns a small region whose dissimilarities follow a log-linear surface.

    The dissimilarities follow maximum-likelihood population effects (MLPE) on the
    surface's effective resistances.
    """
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
        _GENERATING_COEFFICIENTS,
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


class _TrainedWithPenalty(NamedTuple):
    """A region–species group, a training state on it and the state's Laplace approximation."""

    region: RegionBatch
    observations: PairwiseObservations
    state: TrainingState[LogLinearConductance]
    approximation: LaplaceApproximation


@pytest.fixture(scope="module")
def trained_with_penalty() -> _TrainedWithPenalty:
    """Returns the log-linear patch encoder trained with MLPE and a penalty to convergence.

    Training starts from the generating coefficients and the pinned intercept.
    """
    region, observations = _region_and_observations()
    start = eqx.tree_at(
        lambda model: (model.coefficients, model.intercepts),
        LogLinearConductance(2, patch_size=1, solver_config=_TIGHT_SOLVER),
        (_GENERATING_COEFFICIENTS, jnp.asarray([_PINNED_INTERCEPT])),
    )
    config = TrainingConfig(
        loss=MLPELoss(),
        steps=100,
        eval_every=100,
        optimizer=optax.lbfgs(),
        gradient_tolerance=1e-9,
        penalty=_PENALTY,
    )
    state = train(region, observations, distance_model=start, config=config)
    assert state.converged
    approximation = laplace_approximation(region, observations, state, where=_coefficients)
    return _TrainedWithPenalty(region, observations, state, approximation)


def _identity_approximation(where, size: int) -> LaplaceApproximation:
    """Returns a Laplace approximation with an identity Hessian over ``size`` parameters."""
    identity = jnp.eye(size)
    return LaplaceApproximation(
        where=where,
        parameter_names=tuple(f"parameter {index}" for index in range(size)),
        hessian=identity,
        covariance=identity,
        standard_deviations=jnp.ones(size),
        positive_definite=True,
    )


def test_summed_training_objective_is_the_pair_count_times_the_training_objective(
    trained_with_penalty,
):
    region, observations, state, _ = trained_with_penalty
    value = summed_training_objective(
        region,
        observations,
        state.latest_model.distance_model,
        state.loss_state,
        config=state.config,
    )
    pair_count = len(observations.observed_pairs)
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


def test_after_converged_training_with_a_penalty_the_hessian_is_positive_definite_and_step_small(
    trained_with_penalty,
):
    region, observations, state, approximation = trained_with_penalty
    assert approximation.positive_definite
    assert np.all(np.linalg.eigvalsh(approximation.hessian) > 0)
    np.testing.assert_allclose(
        approximation.covariance @ approximation.hessian, np.eye(4), atol=1e-10
    )
    np.testing.assert_allclose(
        approximation.standard_deviations, np.sqrt(np.diag(approximation.covariance))
    )

    step = newton_step(region, observations, state, approximation)

    assert step.shape == (4,)
    assert np.abs(step / approximation.standard_deviations).max() < 1e-6


def test_from_a_perturbed_model_the_newton_step_points_back_to_the_trained_model(
    trained_with_penalty,
):
    region, observations, state, approximation = trained_with_penalty
    deviations = approximation.standard_deviations
    perturbation = jnp.asarray([0.1, -0.1]) * deviations[:2]
    perturbed = eqx.tree_at(
        lambda current: current.latest_model.distance_model.coefficients,
        state,
        state.latest_model.distance_model.coefficients + perturbation,
    )

    step = newton_step(region, observations, perturbed, approximation)

    # Newton's method leaves an error of second order in the perturbation.
    remaining = (np.concatenate([perturbation, np.zeros(2)]) + step) / deviations
    assert np.abs(remaining).max() < 0.02


def test_leaves_the_selector_does_not_select_are_no_parameters_and_keep_their_values(
    trained_with_penalty,
):
    region, observations, state, approximation = trained_with_penalty
    with_intercept = laplace_approximation(
        region, observations, state, where=lambda model: (model.coefficients, model.intercepts)
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
    # Holding the intercept at its trained value gives the same second derivatives as
    # differentiating through it at that value.
    kept = [0, 1, 3, 4]
    np.testing.assert_allclose(
        approximation.hessian, with_intercept.hessian[np.ix_(kept, kept)], rtol=1e-7, atol=1e-7
    )


@pytest.mark.parametrize("step", [0.0, -1e-4, float("nan"), float("inf")])
def test_laplace_approximation_rejects_a_step_that_is_not_finite_and_positive(
    trained_with_penalty, step
):
    region, observations, state, _ = trained_with_penalty
    with pytest.raises(ValueError, match="step must be finite and positive"):
        laplace_approximation(region, observations, state, where=_coefficients, step=step)


def test_several_regions_raise(ScalarEmbedding, regions_and_observations):
    regions, observations = regions_and_observations()
    model = ScalarEmbedding(jnp.asarray(1.0))
    config = TrainingConfig(loss=Log1pLoss(), steps=0)
    state = train(regions, observations, distance_model=model, config=config)

    with pytest.raises(ValueError, match="one region–species group"):
        summed_training_objective(regions, observations, model, None, config=config)
    with pytest.raises(ValueError, match="one region–species group"):
        laplace_approximation(regions, observations, state, where=lambda model: model.weight)
    with pytest.raises(ValueError, match="one region–species group"):
        laplace_approximation(regions[0], observations[0], state, where=lambda model: model.weight)
    approximation = _identity_approximation(lambda model: model.weight, 1)
    with pytest.raises(ValueError, match="one region–species group"):
        newton_step(regions, observations, state, approximation)
    with pytest.raises(ValueError, match="one region–species group"):
        newton_step(regions[0], observations[0], state, approximation)


def test_several_species_raise(training_problem):
    region, observations, model = training_problem("embedding", species=("frog", "newt"))
    config = TrainingConfig(loss=Log1pLoss(), steps=0)
    state = train(region, observations, distance_model=model, config=config)

    with pytest.raises(ValueError, match="one region–species group"):
        summed_training_objective(region, observations, model, None, config=config)
    with pytest.raises(ValueError, match="one region–species group"):
        laplace_approximation(region, observations, state, where=lambda model: model.log_weights)
    with pytest.raises(ValueError, match="one region–species group"):
        newton_step(
            region,
            observations,
            state,
            _identity_approximation(lambda model: model.log_weights, 4),
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
            state,
            _identity_approximation(lambda model: model.log_scale, 1),
        )
    assert caught.type is RuntimeError
    assert str(caught.value).startswith(f"region 'bad-solve', species 0: Newton step: {reason}")


def test_a_later_fold_on_as_many_pairs_compiles_nothing(trained_with_penalty, caplog):
    region, observations, state, approximation = trained_with_penalty
    # Two population-holdout folds, each without one sampling unit's pairs, and a
    # training state of each, with its own latest model.
    first, second = (
        observations.select_pairs(
            [pair for pair in observations.observed_pairs if unit not in pair]
        )
        for unit in region.sampling_unit_ids[:2]
    )
    coefficients = state.latest_model.distance_model.coefficients
    first_state, second_state = (
        eqx.tree_at(
            lambda current: current.latest_model.distance_model.coefficients,
            state,
            coefficients + shift,
        )
        for shift in (0.01, -0.01)
    )
    newton_step(region, first, first_state, approximation)

    with jax.log_compiles(), caplog.at_level(logging.WARNING):
        newton_step(region, second, second_state, approximation)

    assert not [record for record in caplog.records if "Compiling" in record.getMessage()]
