"""Tests of the log1p and maximum-likelihood population effects (MLPE) losses."""

import jax.numpy as jnp
import numpy as np
import pytest
from ilg_toolkit import (
    fit_calibration_head,
    LandscapeGeneticsModel,
    Log1pLoss,
    MLPECalibrationHead,
    MLPELoss,
    train,
    TrainingConfig,
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
