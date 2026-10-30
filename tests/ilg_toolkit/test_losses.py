"""Tests of the log1p and maximum-likelihood population effects (MLPE) losses."""

from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from ilg_toolkit import (
    fit_calibration_head,
    LandscapeGeneticsModel,
    Log1pLoss,
    MLPECalibrationHead,
    MLPELoss,
    PairwiseObservations,
    ResistanceSolverConfig,
    RidgePenalty,
    train,
    TrainingConfig,
)


_PENALTY = RidgePenalty(prior_sd=0.5, where=lambda model: model.log_weights)


def _trained(region, observations, distance_model, loss, *, penalty=None):
    """Returns the training state of three updates, evaluated at the final step."""
    return train(
        region,
        observations,
        distance_model=distance_model,
        config=TrainingConfig(
            loss=loss, steps=3, eval_every=3, optimizer=optax.adam(0.05), penalty=penalty
        ),
    )


@pytest.fixture(scope="module", params=[Log1pLoss(), MLPELoss()], ids=["log1p", "mlpe"])
def trained_group(request, WeightedEmbedding, embedding_problem):
    """Returns one region–species group, its loss and its ridge-penalized training state."""
    region, observations = embedding_problem()
    model = WeightedEmbedding(jnp.asarray([0.3, -0.2]))
    state = _trained(region, observations, model, request.param, penalty=_PENALTY)
    return SimpleNamespace(
        region=region,
        observations=observations,
        loss=request.param,
        state=state,
        model=state.latest_model.distance_model,
    )


def _summed_loss(group, distance_model):
    """Returns the summed loss of a trained group at a distance model."""
    return group.loss.sum_over_pairs(
        group.region, group.observations, distance_model, group.state.loss_state
    )


def test_summed_loss_is_the_pair_count_times_the_training_loss_and_objective(trained_group):
    group = trained_group
    pair_count = len(group.observations.observed_pairs)
    record = group.state.history[-1]
    summed = float(_summed_loss(group, group.model))
    assert summed == pytest.approx(pair_count * record.training_loss, rel=1e-12)
    assert summed + float(_PENALTY(group.model)) == pytest.approx(
        pair_count * record.training_objective, rel=1e-12
    )


def test_summed_loss_gradient_compiles_and_matches_central_differences(trained_group):
    group = trained_group

    def shifted(index, shift):
        weights = group.model.log_weights.at[index].add(shift)
        return _summed_loss(group, eqx.tree_at(lambda m: m.log_weights, group.model, weights))

    step = 1e-6
    differences = [(shifted(i, step) - shifted(i, -step)) / (2 * step) for i in range(2)]
    gradient = eqx.filter_jit(eqx.filter_grad(lambda model: _summed_loss(group, model)))(
        group.model
    )
    np.testing.assert_allclose(gradient.log_weights, differences, rtol=1e-7)


def test_summed_loss_is_unchanged_when_observations_list_their_units_in_another_order(
    trained_group,
):
    group = trained_group
    ids = group.region.sampling_unit_ids
    shuffled = PairwiseObservations.from_pairs(
        group.observations.observed_pairs,
        group.observations.observed_values,
        target=group.observations.target,
        sampling_unit_ids=tuple(ids[index] for index in (2, 0, 5, 1, 4, 3)),
    )
    value = group.loss.sum_over_pairs(group.region, shuffled, group.model, group.state.loss_state)
    assert float(value) == pytest.approx(float(_summed_loss(group, group.model)), rel=1e-13)


def test_summed_log1p_loss_is_the_sum_of_squared_differences(
    ScalarEmbedding, regions_and_observations
):
    regions, observations = regions_and_observations()
    # Weight 1 predicts the distances 1, 4 and 1 of the wide region, whose targets are 0.1.
    value = Log1pLoss().sum_over_pairs(
        regions[0], observations[0], ScalarEmbedding(jnp.asarray(1.0)), None
    )
    expected = sum((np.log1p(distance) - np.log1p(0.1)) ** 2 for distance in (1.0, 4.0, 1.0))
    assert float(value) == pytest.approx(expected, rel=1e-12)


def _assert_direct_summed_loss_matches_conjugate_gradient(training_problem, PointConductance):
    """Checks the compiled summed loss and its gradient on cuDSS against conjugate gradient."""
    region, observations, _ = training_problem("conductance")
    value_and_gradient = eqx.filter_jit(
        eqx.filter_value_and_grad(
            lambda model: Log1pLoss().sum_over_pairs(region, observations, model, None)
        )
    )
    value, gradient = value_and_gradient(
        PointConductance(jnp.asarray(0.2), solver_config=ResistanceSolverConfig(solver="cudss"))
    )
    expected_value, expected_gradient = value_and_gradient(
        PointConductance(
            jnp.asarray(0.2), solver_config=ResistanceSolverConfig(rtol=1e-11, atol=1e-11)
        )
    )
    assert float(value) == pytest.approx(float(expected_value), rel=1e-8)
    assert float(gradient.log_scale) == pytest.approx(float(expected_gradient.log_scale), rel=1e-6)


def test_summed_loss_on_the_direct_solver_matches_conjugate_gradient(
    cudss_double, training_problem, PointConductance
):
    _assert_direct_summed_loss_matches_conjugate_gradient(training_problem, PointConductance)


def test_summed_loss_on_the_direct_solver_matches_conjugate_gradient_on_a_gpu(
    cudss_gpu, training_problem, PointConductance
):
    _assert_direct_summed_loss_matches_conjugate_gradient(training_problem, PointConductance)


def test_summed_mlpe_loss_is_the_negative_log_likelihood_of_the_dense_reference(
    WeightedEmbedding, embedding_problem, dense_profile
):
    region, observations = embedding_problem()
    loss = MLPELoss(jitter=1e-7)
    state = _trained(region, observations, WeightedEmbedding(jnp.zeros(2)), loss)
    model = state.latest_model.distance_model
    calibration = cast(MLPECalibrationHead, state.latest_model.calibration_head).for_region(
        region.name
    )
    pairs = observations.observed_pairs
    lookup = {label: index for index, label in enumerate(region.sampling_unit_ids)}
    nll, *_ = dense_profile(
        np.asarray(model.predict(region, pairs).values),
        np.asarray(observations.observed_values),
        [lookup[a] for a, _ in pairs],
        [lookup[b] for _, b in pairs],
        len(region.sampling_unit_ids),
        (calibration.unit_variance, calibration.residual_variance),
        jitter=1e-7,
    )
    value = loss.sum_over_pairs(region, observations, model, state.loss_state)
    assert float(value) == pytest.approx(nll, rel=1e-9)


def test_central_differences_of_the_summed_mlpe_gradient_match_second_differences_of_the_value(
    WeightedEmbedding, embedding_problem
):
    region, observations = embedding_problem()
    loss = MLPELoss()
    state = _trained(region, observations, WeightedEmbedding(jnp.asarray([0.3, -0.2])), loss)
    trained = state.latest_model.distance_model

    @eqx.filter_jit
    def value(log_weights):
        model = eqx.tree_at(lambda m: m.log_weights, trained, log_weights)
        return loss.sum_over_pairs(region, observations, model, state.loss_state)

    gradient = eqx.filter_jit(jax.grad(value))
    start, unit = trained.log_weights, jnp.asarray([1.0, 0.0])
    gradient_step, value_step = 1e-5 * unit, 1e-4 * unit
    from_gradient = (gradient(start + gradient_step)[0] - gradient(start - gradient_step)[0]) / 2e-5
    from_value = (value(start + value_step) - 2 * value(start) + value(start - value_step)) / 1e-8
    assert float(from_gradient) == pytest.approx(float(from_value), rel=1e-6)
    # Second-order automatic differentiation is conditional on the coefficients.
    assert float(jax.jacfwd(gradient)(start)[0, 0]) != pytest.approx(float(from_value), rel=1e-3)


def test_summed_mlpe_loss_reads_the_observed_species_group(training_problem):
    species = ("frog", "newt")
    loss = MLPELoss()
    region, observations, distance_model = training_problem("embedding", species=species)
    state = _trained(region, observations, distance_model, loss)
    model = state.latest_model.distance_model
    record = state.history[-1]
    assert record.training_by_species is not None
    for name, observed in zip(species, observations, strict=True):
        summed = eqx.filter_jit(
            lambda current, group=observed: loss.sum_over_pairs(
                region, group, current, state.loss_state
            )
        )
        expected = len(observed.observed_pairs) * record.training_by_species[name]
        assert float(summed(model)) == pytest.approx(expected, rel=1e-12)


def test_summed_loss_rejects_inputs_outside_the_statistical_parameter_state_and_loss(
    WeightedEmbedding, embedding_problem, training_problem
):
    loss = MLPELoss()
    region, observations = embedding_problem()
    model = WeightedEmbedding(jnp.zeros(2))
    state = _trained(region, observations, model, loss)
    with pytest.raises(ValueError, match="region–species group"):
        loss.sum_over_pairs(
            replace(region, name="elsewhere"), observations, model, state.loss_state
        )
    with pytest.raises(ValueError, match="at least three"):
        selected = observations.select_pairs(observations.observed_pairs[:2])
        loss.sum_over_pairs(region, selected, model, state.loss_state)
    labeled, groups, species_model = training_problem("embedding", species=("frog", "newt"))
    with pytest.raises(ValueError, match="species"):
        loss.sum_over_pairs(labeled, replace(groups[0], species=None), species_model, None)


def test_summed_loss_requires_float64(ScalarEmbedding, regions_and_observations, float64_required):
    regions, observations = regions_and_observations()
    with float64_required():
        Log1pLoss().sum_over_pairs(
            regions[0], observations[0], ScalarEmbedding(jnp.asarray(1.0)), None
        )


def test_direct_loss_trains_without_statistical_parameters(
    ScalarEmbedding, regions_and_observations
):
    regions, observations = regions_and_observations()
    model = ScalarEmbedding(jnp.asarray(1.0))
    state = train(
        regions[0],
        observations[0],
        distance_model=model,
        config=TrainingConfig(loss=Log1pLoss(), steps=2),
    )
    assert state.loss_state is None
    assert state.history[-1].training_loss < state.history[0].training_loss
    prediction = state.selected_model.predict(regions[0], observations[0].observed_pairs)
    assert np.isfinite(prediction.values).all()


def test_mlpe_losses_reject_individual_based_regions_and_log1p_accepts_them(
    ScalarEmbedding, four_population_calibration
):
    from dataclasses import replace

    region, _, observations = four_population_calibration
    individuals = replace(region, individual_based=True)
    model = ScalarEmbedding(jnp.asarray(1.0))
    with pytest.raises(ValueError, match="individual-based"):
        train(
            individuals,
            observations,
            distance_model=model,
            config=TrainingConfig(loss=MLPELoss(), steps=0),
        )
    with pytest.raises(ValueError, match="individual-based"):
        fit_calibration_head(individuals, observations, model, loss=MLPELoss())
    state = train(
        individuals,
        observations,
        distance_model=model,
        config=TrainingConfig(loss=Log1pLoss(), steps=1),
    )
    assert np.isfinite(state.selected_model.predict(individuals).values).all()
    head = fit_calibration_head(individuals, observations, model, loss=Log1pLoss())
    composed = LandscapeGeneticsModel(model, head, individuals.feature_names)
    assert np.isfinite(composed.predict(individuals).values).all()


def test_training_mlpe_calibrations_match_the_dense_reference_profile(
    ScalarEmbedding, four_population_calibration, dense_profile
):
    region, observations, _ = four_population_calibration
    # Four of the six pairs among four populations leave the pair design incomplete.
    pairs = (
        ("population-1", "population-3"),
        ("population-0", "population-2"),
        ("population-1", "population-2"),
        ("population-0", "population-1"),
    )
    selected = observations.select_pairs(pairs)
    distance_model = ScalarEmbedding(jnp.asarray(1.0))
    head = train(
        region,
        selected,
        distance_model=distance_model,
        config=TrainingConfig(loss=MLPELoss(jitter=1e-7), steps=0),
    ).selected_model.calibration_head
    assert isinstance(head, MLPECalibrationHead)
    calibration = head.for_region(region.name)
    assert calibration.population_ids == region.sampling_unit_ids
    assert not calibration.converged and calibration.optimizer_iterations == 0
    lookup = {label: index for index, label in enumerate(region.sampling_unit_ids)}
    distances = np.asarray(distance_model.predict(region, selected.observed_pairs).values)
    nll, beta, effect_mean, effect_covariance = dense_profile(
        distances,
        np.asarray(selected.observed_values),
        [lookup[a] for a, _ in selected.observed_pairs],
        [lookup[b] for _, b in selected.observed_pairs],
        len(region.sampling_unit_ids),
        (calibration.unit_variance, calibration.residual_variance),
        jitter=1e-7,
    )
    assert calibration.ml_log_likelihood == pytest.approx(-nll, abs=1e-10)
    np.testing.assert_allclose([calibration.intercept, calibration.slope], beta, atol=1e-10)
    np.testing.assert_allclose(calibration.effect_mean, effect_mean, atol=1e-10)
    np.testing.assert_allclose(calibration.effect_covariance, effect_covariance, atol=1e-10)
    assert calibration.distance_center == pytest.approx(distances.mean(), abs=1e-14)
    assert calibration.distance_scale == pytest.approx(distances.std(ddof=1), abs=1e-14)


def test_mlpe_validation_loss_is_the_joint_gaussian_at_training_coefficients(
    ScalarEmbedding, four_population_calibration
):
    region, observations, training = four_population_calibration
    held_out = [
        pair for pair in observations.observed_pairs if pair[1] in ("population-4", "population-5")
    ]
    distance_model = ScalarEmbedding(jnp.asarray(1.0))
    jitter = 1e-7
    lookup = {label: index for index, label in enumerate(region.sampling_unit_ids)}
    # One pair has the marginal variance 2u + e + j; several pairs share populations.
    for pairs in (held_out[:1], held_out):
        validation = observations.select_pairs(pairs)
        state = train(
            region,
            training,
            distance_model=distance_model,
            config=TrainingConfig(loss=MLPELoss(jitter=jitter), steps=0),
            validation=(region, validation),
        )
        head = state.latest_model.calibration_head
        assert isinstance(head, MLPECalibrationHead)
        calibration = head.for_region(region.name)
        distances = np.asarray(distance_model.predict(region, validation.observed_pairs).values)
        residual = np.asarray(validation.observed_values) - (
            calibration.intercept
            + calibration.slope
            * (distances - calibration.distance_center)
            / calibration.distance_scale
        )
        incidence = np.zeros((len(distances), len(lookup)))
        for row, (a, b) in enumerate(validation.observed_pairs):
            incidence[row, lookup[a]] = incidence[row, lookup[b]] = 1
        covariance = calibration.unit_variance * incidence @ incidence.T + (
            calibration.residual_variance + jitter
        ) * np.eye(len(distances))
        expected = 0.5 * (
            len(distances) * np.log(2 * np.pi)
            + np.linalg.slogdet(covariance)[1]
            + residual @ np.linalg.solve(covariance, residual)
        )
        if len(pairs) == 1:
            variance = 2 * calibration.unit_variance + calibration.residual_variance + jitter
            assert expected == pytest.approx(
                0.5 * (np.log(2 * np.pi * variance) + residual[0] ** 2 / variance), abs=1e-12
            )
        assert state.history[0].validation_loss == pytest.approx(
            expected / len(distances), abs=1e-12
        )


def test_mlpe_training_rejects_unidentifiable_variances_and_constant_distances(
    ScalarEmbedding, four_population_calibration
):
    region, observations, training = four_population_calibration
    config = TrainingConfig(loss=MLPELoss(), steps=0)
    disjoint = observations.select_pairs(
        [
            ("population-0", "population-1"),
            ("population-2", "population-3"),
            ("population-4", "population-5"),
        ]
    )
    with pytest.raises(ValueError, match="unidentifiable"):
        train(region, disjoint, distance_model=ScalarEmbedding(jnp.asarray(1.0)), config=config)
    # A zero embedding gives constant distances, which have no sample scale.
    with pytest.raises(FloatingPointError, match="nonfinite loss"):
        train(region, training, distance_model=ScalarEmbedding(jnp.asarray(0.0)), config=config)
