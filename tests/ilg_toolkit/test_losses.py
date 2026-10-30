"""Tests of the log1p and maximum-likelihood population effects (MLPE) losses."""

from dataclasses import replace

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
    RidgePenalty,
    train,
    TrainingConfig,
)


both_losses = pytest.mark.parametrize("loss", [Log1pLoss(), MLPELoss()], ids=["log1p", "mlpe"])


def _trained(region, observations, distance_model, loss, *, penalty=None):
    """Returns the training state of a few updates, evaluated at the final step."""
    return train(
        region,
        observations,
        distance_model=distance_model,
        config=TrainingConfig(
            loss=loss, steps=3, eval_every=3, optimizer=optax.adam(0.05), penalty=penalty
        ),
    )


@both_losses
def test_summed_value_is_the_pair_count_times_the_reported_training_loss(
    loss, WeightedEmbedding, persistence_problem
):
    region, observations = persistence_problem()
    state = _trained(region, observations, WeightedEmbedding(jnp.zeros(2)), loss)
    pairs = observations.observed_pairs
    distances = state.latest_model.distance_model.predict(region, pairs).values
    value = loss.summed_value(region, observations, distances, state.loss_state)
    assert float(value) == pytest.approx(len(pairs) * state.history[-1].training_loss, rel=1e-12)


def test_summed_log1p_loss_is_the_sum_of_squared_differences(regions_and_observations):
    regions, observations = regions_and_observations()
    distances = jnp.asarray([0.5, 1.0, 3.0])
    expected = sum((np.log1p(distance) - np.log1p(0.1)) ** 2 for distance in (0.5, 1.0, 3.0))
    value = Log1pLoss().summed_value(regions[0], observations[0], distances, None)
    assert float(value) == pytest.approx(expected, rel=1e-12)


def test_summed_mlpe_loss_is_the_negative_log_likelihood_of_the_dense_reference(
    WeightedEmbedding, embedding_problem, dense_profile
):
    region, observations = embedding_problem()
    loss = MLPELoss(jitter=1e-7)
    state = _trained(region, observations, WeightedEmbedding(jnp.zeros(2)), loss)
    head = state.latest_model.calibration_head
    assert isinstance(head, MLPECalibrationHead)
    calibration = head.for_region(region.name)
    pairs = observations.observed_pairs
    distances = state.latest_model.distance_model.predict(region, pairs).values
    lookup = {label: index for index, label in enumerate(region.sampling_unit_ids)}
    nll, *_ = dense_profile(
        np.asarray(distances),
        np.asarray(observations.observed_values),
        [lookup[a] for a, _ in pairs],
        [lookup[b] for _, b in pairs],
        len(region.sampling_unit_ids),
        (calibration.unit_variance, calibration.residual_variance),
        jitter=1e-7,
    )
    value = loss.summed_value(region, observations, distances, state.loss_state)
    assert float(value) == pytest.approx(nll, rel=1e-9)


@both_losses
def test_summed_value_plus_the_penalty_is_the_pair_count_times_the_training_objective(
    loss, WeightedEmbedding, embedding_problem
):
    region, observations = embedding_problem()
    penalty = RidgePenalty(prior_sd=0.5, where=lambda model: model.log_weights)
    initial = WeightedEmbedding(jnp.asarray([0.3, -0.2]))
    state = _trained(region, observations, initial, loss, penalty=penalty)
    pairs = observations.observed_pairs
    trained = state.latest_model.distance_model
    distances = trained.predict(region, pairs).values
    value = loss.summed_value(region, observations, distances, state.loss_state) + penalty(trained)
    record = state.history[-1]
    assert float(penalty(trained)) > 0
    assert float(value) == pytest.approx(len(pairs) * record.training_objective, rel=1e-12)


@both_losses
def test_summed_value_gradient_through_predict_matches_central_differences(
    loss, WeightedEmbedding, embedding_problem
):
    region, observations = embedding_problem()
    state = _trained(region, observations, WeightedEmbedding(jnp.asarray([0.3, -0.2])), loss)
    trained = state.latest_model.distance_model

    def summed_loss(distance_model):
        distances = distance_model.predict(region, observations.observed_pairs).values
        return loss.summed_value(region, observations, distances, state.loss_state)

    def shifted(index, shift):
        weights = trained.log_weights.at[index].add(shift)
        return summed_loss(eqx.tree_at(lambda model: model.log_weights, trained, weights))

    step = 1e-6
    differences = [(shifted(i, step) - shifted(i, -step)) / (2 * step) for i in range(2)]
    gradient = eqx.filter_grad(summed_loss)(trained).log_weights
    assert np.abs(gradient).min() > 1e-2
    np.testing.assert_allclose(gradient, differences, rtol=1e-7)


def test_central_differences_of_the_summed_mlpe_gradient_match_second_differences_of_the_value(
    WeightedEmbedding, embedding_problem
):
    region, observations = embedding_problem()
    loss = MLPELoss()
    state = _trained(region, observations, WeightedEmbedding(jnp.asarray([0.3, -0.2])), loss)
    trained = state.latest_model.distance_model

    @eqx.filter_jit
    def value(log_weights):
        distance_model = eqx.tree_at(lambda model: model.log_weights, trained, log_weights)
        distances = distance_model.predict(region, observations.observed_pairs).values
        return loss.summed_value(region, observations, distances, state.loss_state)

    gradient = eqx.filter_jit(jax.grad(value))
    start = trained.log_weights
    basis = jnp.eye(2)
    gradient_step = 1e-5
    from_gradient = np.column_stack(
        [
            (gradient(start + gradient_step * unit) - gradient(start - gradient_step * unit))
            / (2 * gradient_step)
            for unit in basis
        ]
    )
    value_step = 1e-4

    def second_difference(first, second):
        shift, other = value_step * first, value_step * second
        return (
            value(start + shift + other)
            - value(start + shift - other)
            - value(start - shift + other)
            + value(start - shift - other)
        ) / (4 * value_step**2)

    from_value = np.asarray([[second_difference(a, b) for b in basis] for a in basis])
    np.testing.assert_allclose(from_gradient, from_value, rtol=1e-6)
    # Second-order automatic differentiation is conditional on the coefficients.
    conditional = jax.jacfwd(gradient)(start)
    assert np.abs(conditional - from_value).max() > 0.5 * np.abs(from_value).max()


@both_losses
def test_summed_value_is_unchanged_when_observations_list_their_units_in_another_order(
    loss, WeightedEmbedding, embedding_problem
):
    region, observations = embedding_problem()
    state = _trained(region, observations, WeightedEmbedding(jnp.asarray([0.3, -0.2])), loss)
    trained = state.latest_model.distance_model
    ids = region.sampling_unit_ids
    shuffled = PairwiseObservations.from_pairs(
        observations.observed_pairs,
        observations.observed_values,
        target=observations.target,
        sampling_unit_ids=tuple(ids[index] for index in (2, 0, 5, 1, 4, 3)),
    )
    assert shuffled.observed_pairs != observations.observed_pairs

    def value(listed):
        distances = trained.predict(region, listed.observed_pairs).values
        return float(loss.summed_value(region, listed, distances, state.loss_state))

    assert value(shuffled) == pytest.approx(value(observations), rel=1e-13)


@both_losses
def test_summed_value_of_a_labeled_region_evaluates_the_observed_species_group(
    loss, training_problem
):
    species = ("frog", "newt")
    region, observations, distance_model = training_problem("embedding", species=species)
    state = _trained(region, observations, distance_model, loss)
    trained = state.latest_model.distance_model
    record = state.history[-1]
    assert record.training_by_species is not None
    single_species_region, _, _ = training_problem("embedding")
    for index, (name, observed) in enumerate(zip(species, observations, strict=True)):
        pairs = observed.observed_pairs
        distances = trained.predict(region, {name: pairs})[name].values
        value = loss.summed_value(
            region, observed, distances, state.loss_state, species_index=index
        )
        assert float(value) == pytest.approx(
            len(pairs) * record.training_by_species[name], rel=1e-12
        )
        # The species' own sampling units alone give the same value.
        alone = loss.summed_value(
            single_species_region,
            replace(observed, species=None),
            distances,
            state.loss_state,
            species_index=index,
        )
        assert float(value) == pytest.approx(float(alone), rel=1e-13)


def test_summed_mlpe_loss_reads_the_statistical_parameter_state_of_the_species_index(
    training_problem,
):
    species = ("frog", "newt")
    loss = MLPELoss()
    region, observations, distance_model = training_problem("embedding", species=species)
    state = _trained(region, observations, distance_model, loss)
    newt = observations[1]
    distances = state.latest_model.distance_model.predict(region, {"newt": newt.observed_pairs})[
        "newt"
    ].values
    with_frog_state, with_newt_state = (
        loss.summed_value(region, newt, distances, state.loss_state, species_index=index)
        for index in (0, 1)
    )
    assert float(with_frog_state) != pytest.approx(float(with_newt_state), rel=1e-3)


@both_losses
def test_summed_value_reads_the_only_species_of_a_labeled_region_for_unnamed_observations(
    loss, ScalarEmbedding, embedding_problem
):
    region, observations = embedding_problem()
    labeled = replace(region, species=("frog",) * len(region.sampling_unit_ids))
    state = _trained(
        labeled, observations, ScalarEmbedding(jnp.asarray(1.0), species=("frog",)), loss
    )
    pairs = observations.observed_pairs
    distances = state.latest_model.distance_model.predict(labeled, pairs).values
    value = loss.summed_value(labeled, observations, distances, state.loss_state)
    assert float(value) == pytest.approx(len(pairs) * state.history[-1].training_loss, rel=1e-12)


@both_losses
def test_summed_value_rejects_pairs_with_a_sampling_unit_of_another_species(
    loss, embedding_problem
):
    region, observations = embedding_problem()
    labeled = replace(region, species=("frog",) * 3 + ("newt",) * 3)
    ids = region.sampling_unit_ids
    # The first unit of these pairs belongs to frog in the region.
    newt = replace(
        observations.select_pairs([(ids[2], ids[3]), (ids[2], ids[4]), (ids[3], ids[4])]),
        species="newt",
    )
    with pytest.raises(ValueError, match="belong to their species"):
        loss.summed_value(labeled, newt, jnp.ones(3), None)


@both_losses
def test_summed_value_plus_the_penalty_compiles(loss, WeightedEmbedding, embedding_problem):
    region, observations = embedding_problem()
    state = _trained(region, observations, WeightedEmbedding(jnp.asarray([0.3, -0.2])), loss)
    penalty = RidgePenalty(prior_sd=0.5, where=lambda model: model.log_weights)

    def summed_loss_plus_penalty(distance_model):
        distances = distance_model.predict(region, observations.observed_pairs).values
        summed = loss.summed_value(region, observations, distances, state.loss_state)
        return summed + penalty(distance_model)

    trained = state.latest_model.distance_model
    value_and_gradient = eqx.filter_value_and_grad(summed_loss_plus_penalty)
    value, gradient = value_and_gradient(trained)
    compiled_value, compiled_gradient = eqx.filter_jit(value_and_gradient)(trained)
    assert float(compiled_value) == pytest.approx(float(value), rel=1e-12)
    np.testing.assert_allclose(compiled_gradient.log_weights, gradient.log_weights, rtol=1e-10)


@both_losses
def test_summed_value_compiles_for_a_labeled_region(loss, training_problem):
    region, observations, distance_model = training_problem("embedding", species=("frog", "newt"))
    state = _trained(region, observations, distance_model, loss)
    newt = observations[1]
    distances = state.latest_model.distance_model.predict(region, {"newt": newt.observed_pairs})[
        "newt"
    ].values

    def value(ecological_distances):
        return loss.summed_value(
            region, newt, ecological_distances, state.loss_state, species_index=1
        )

    assert float(eqx.filter_jit(value)(distances)) == pytest.approx(
        float(value(distances)), rel=1e-12
    )


@both_losses
def test_summed_value_rejects_distances_of_the_wrong_shape(
    loss, WeightedEmbedding, embedding_problem
):
    region, observations = embedding_problem()
    state = _trained(region, observations, WeightedEmbedding(jnp.zeros(2)), loss)
    # Shapes (1,) and (15, 1) broadcast against the 15 targets.
    for shape in [(14,), (16,), (15, 1), (1,), ()]:
        with pytest.raises(ValueError, match="observed_pairs"):
            loss.summed_value(region, observations, jnp.ones(shape), state.loss_state)


@pytest.mark.parametrize(
    ("region_name", "species_index"),
    [("elsewhere", 0), ("alpine", 1)],
    ids=["region", "species_index"],
)
def test_summed_mlpe_loss_rejects_a_group_absent_from_the_statistical_parameter_state(
    region_name, species_index, WeightedEmbedding, embedding_problem
):
    loss = MLPELoss()
    region, observations = embedding_problem()
    state = _trained(region, observations, WeightedEmbedding(jnp.zeros(2)), loss)
    distances = jnp.arange(1.0, 16.0)
    with pytest.raises(ValueError, match="region–species group"):
        loss.summed_value(
            replace(region, name=region_name),
            observations,
            distances,
            state.loss_state,
            species_index=species_index,
        )


def test_summed_mlpe_loss_rejects_the_state_of_a_loss_without_statistical_parameters(
    embedding_problem,
):
    region, observations = embedding_problem()
    with pytest.raises(ValueError, match="region–species group"):
        MLPELoss().summed_value(region, observations, jnp.arange(1.0, 16.0), None)


def test_summed_mlpe_loss_of_constant_distances_is_not_a_number(
    WeightedEmbedding, embedding_problem
):
    region, observations = embedding_problem()
    loss = MLPELoss()
    state = _trained(region, observations, WeightedEmbedding(jnp.zeros(2)), loss)
    assert np.isnan(loss.summed_value(region, observations, jnp.ones(15), state.loss_state))


def test_summed_mlpe_loss_rejects_too_few_pairs_to_identify_its_variances(
    WeightedEmbedding, embedding_problem
):
    region, observations = embedding_problem()
    loss = MLPELoss()
    state = _trained(region, observations, WeightedEmbedding(jnp.zeros(2)), loss)
    selected = observations.select_pairs(observations.observed_pairs[:2])
    with pytest.raises(ValueError, match="at least three"):
        loss.summed_value(region, selected, jnp.ones(2), state.loss_state)


@both_losses
def test_summed_value_requires_observations_to_name_one_species_of_a_labeled_region(
    loss, training_problem
):
    region, observations, distance_model = training_problem("embedding", species=("frog", "newt"))
    state = _trained(region, observations, distance_model, loss)
    with pytest.raises(ValueError, match="species"):
        loss.summed_value(
            region, replace(observations[0], species=None), jnp.ones(6), state.loss_state
        )


@both_losses
def test_summed_value_requires_float64(
    loss, WeightedEmbedding, embedding_problem, float64_required
):
    region, observations = embedding_problem()
    state = _trained(region, observations, WeightedEmbedding(jnp.zeros(2)), loss)
    distances = jnp.arange(1.0, 16.0)
    with float64_required():
        loss.summed_value(region, observations, distances, state.loss_state)


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
