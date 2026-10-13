"""Species-balanced complete-group MLPE against independent dense Gaussian fits."""

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from test_mlpe_training import WeightedEmbedding, dense_profile, embedding_problem

from ilg_toolkit import (
    ObservationPartition,
    PairwiseObservations,
    TrainingConfig,
    UnpooledMLPECalibrationModule,
    fit,
)


def irregular_problem():
    first, observed = embedding_problem()
    second, other_observed = embedding_problem("valley", offset=3, seed=71)
    first = replace(
        first,
        sampling_units={
            "fox": first.sampling_units["species"],
            "lynx": replace(
                first.sampling_units["species"],
                grid_positions=np.asarray(first.sampling_units["species"].grid_positions)[::-1],
            ),
        },
    )
    second = replace(second, sampling_units={"fox": second.sampling_units["species"]})
    lynx = PairwiseObservations.from_pairs(
        observed.observed_pairs[:7], 3 + observed.observed_values[:7] * 1.8, target=observed.target
    )
    sparse = PairwiseObservations.from_pairs(
        other_observed.observed_pairs[:9],
        other_observed.observed_values[:9],
        target=other_observed.target,
    )
    groups = (
        (first, "fox", observed, 0.25),
        (first, "lynx", lynx, 0.5),
        (second, "fox", sparse, 0.25),
    )
    return (
        [second, first],
        {first.name: {"fox": observed, "lynx": lynx}, second.name: {"fox": sparse}},
        groups,
    )


def oracle_group(weights, raw, prepared, species, observed):
    units = prepared.sampling_units[species]
    positions = np.asarray(units.grid_positions)
    embedding = np.asarray(prepared.feature_array)[positions[:, 0], positions[:, 1]] * np.exp(
        weights
    )
    left = np.asarray([units.ids.index(a) for a, b in observed.observed_pairs])
    right = np.asarray([units.ids.index(b) for a, b in observed.observed_pairs])
    distances = np.sum((embedding[left] - embedding[right]) ** 2, axis=1)
    variances = np.logaddexp(0, raw) + 1e-10
    result = dense_profile(
        distances, np.asarray(observed.observed_values), left, right, len(units.ids), variances
    )
    return distances, variances, result


def test_irregular_mlpe_species_objective_gradients_and_final_calibrations_match_dense_oracle():
    regions, observed, groups = irregular_problem()
    raw = np.log(np.expm1(np.array([0.05, 0.1]) - 1e-10))
    initial = np.concatenate((np.zeros(2), np.tile(raw, 3)))

    def objective(parameters):
        return sum(
            weight
            * oracle_group(parameters[:2], parameters[2 + 2 * i : 4 + 2 * i], r, s, o)[2][0]
            / len(o.observed_pairs)
            for i, (r, s, o, weight) in enumerate(groups)
        )

    delta = 1e-5
    gradient = np.array(
        [
            (objective(initial + d) - objective(initial - d)) / (2 * delta)
            for d in np.eye(len(initial)) * delta
        ]
    )
    state = fit(
        regions,
        observed,
        distance_model=WeightedEmbedding(jnp.zeros(2), species=("fox", "lynx")),
        config=TrainingConfig(
            objective="mlpe", steps=1, learning_rate=0.02, mlpe_initial_variances=(0.05, 0.1)
        ),
    )
    assert state.raw_variances is not None and state.selected_raw_variances is not None
    assert isinstance(state.distance_model, WeightedEmbedding)
    assert isinstance(state.optimizer_state, tuple)
    adam = state.optimizer_state[0]
    assert isinstance(adam, optax.ScaleByAdamState)
    assert isinstance(adam.mu, tuple) and isinstance(adam.nu, tuple)
    mu_model, mu_raw = adam.mu
    nu_model, nu_raw = adam.nu
    assert isinstance(mu_model, WeightedEmbedding) and isinstance(nu_model, WeightedEmbedding)
    assert isinstance(mu_raw, jax.Array) and isinstance(nu_raw, jax.Array)
    np.testing.assert_allclose(state.history[0].training_loss, objective(initial), rtol=1e-12)
    np.testing.assert_allclose(
        np.concatenate((mu_model.log_weights, mu_raw.reshape(-1))),
        0.1 * gradient,
        rtol=1e-6,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.concatenate((nu_model.log_weights, nu_raw.reshape(-1))),
        0.001 * gradient**2,
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.concatenate((state.distance_model.log_weights, state.raw_variances.reshape(-1))),
        initial - 0.02 * gradient / (np.abs(gradient) + 1e-8),
        rtol=1e-9,
        atol=1e-10,
    )
    np.testing.assert_array_equal(state.raw_variances, state.selected_raw_variances)
    assert isinstance(state.model.calibration, UnpooledMLPECalibrationModule)
    assert state.raw_variances.shape == (3, 2)
    for i, (prepared, species, observations, _) in enumerate(groups):
        distances, variances, (nll, beta, mean, covariance) = oracle_group(
            state.distance_model.log_weights,
            state.selected_raw_variances[i],
            prepared,
            species,
            observations,
        )
        head = state.model.calibration.for_group(prepared.name, species)
        assert head.region_name == prepared.name and head.species_name == species
        np.testing.assert_allclose([head.intercept, head.slope], beta, rtol=1e-11)
        np.testing.assert_allclose(
            [head.unit_variance, head.residual_variance], variances, rtol=1e-12
        )
        np.testing.assert_allclose(head.effect_mean, mean, atol=1e-12)
        np.testing.assert_allclose(head.effect_covariance, covariance, atol=1e-12)
        np.testing.assert_allclose(-head.ml_log_likelihood, nll, rtol=1e-12)
        np.testing.assert_allclose(
            [head.distance_center, head.distance_scale],
            [distances.mean(), distances.std(ddof=1)],
            rtol=1e-12,
        )
        predicted = state.model.predict(
            prepared, species=species, pairs={species: observations.observed_pairs}
        )[species]
        assert predicted.species_name == species
        np.testing.assert_allclose(
            predicted.values,
            beta[0] + beta[1] * (distances - distances.mean()) / distances.std(ddof=1),
            rtol=1e-11,
        )
    # Matching region alone must not select another species' calibration.
    valley = regions[0]
    new_group = replace(
        valley,
        sampling_units={"fox": valley.sampling_units["fox"], "lynx": valley.sampling_units["fox"]},
    )
    with pytest.raises(ValueError, match="valley.*lynx.*no MLPE calibration"):
        state.model.predict(new_group, species="lynx")
    with pytest.raises(ValueError, match="group identities"):
        UnpooledMLPECalibrationModule(
            state.model.target,
            {"valley": {"lynx": state.model.calibration.for_group("valley", "fox")}},
        )
    unseen = replace(regions[0], name="uncalibrated")
    assert np.isfinite(state.model.predict_raw_distances(unseen, species="fox")["fox"].values).all()
    with pytest.raises(ValueError, match="uncalibrated.*fox.*no MLPE calibration"):
        state.model.predict(unseen, species="fox")


def test_validation_uses_matching_training_group_and_selected_state_roundtrips(tmp_path):
    regions, observed, groups = irregular_problem()
    valley = regions[0]
    original = observed[valley.name]["fox"]
    held = original.observed_pairs[-1:]
    train_partition = ObservationPartition(valley.name, "fox", original.observed_pairs[:-1])
    partitions = {"alpine": {"fox": None, "lynx": None}, "valley": {"fox": train_partition}}
    validation_partition = {
        "valley": {"fox": ObservationPartition("valley", "fox", held, role="validation")}
    }
    validation_observed = PairwiseObservations.from_pairs(held, [50.0], target=original.target)
    validation = ([valley], {"valley": {"fox": validation_observed}})
    config = TrainingConfig(
        objective="mlpe", steps=3, learning_rate=0.03, seed=31, mlpe_initial_variances=(0.05, 0.1)
    )
    state = fit(
        regions,
        observed,
        partition=partitions,
        validation=validation,
        validation_partition=validation_partition,
        distance_model=WeightedEmbedding(jnp.zeros(2), species=("fox", "lynx")),
        config=config,
    )
    assert state.raw_variances is not None and state.selected_raw_variances is not None
    assert isinstance(state.distance_model, WeightedEmbedding)
    assert isinstance(state.model.distance_model, WeightedEmbedding)
    assert isinstance(state.model.calibration, UnpooledMLPECalibrationModule)
    assert state.selected_step < state.step
    assert not np.array_equal(state.selected_raw_variances, state.raw_variances)
    assert state.history[-1].validation_missing_groups == (("alpine", "fox"), ("alpine", "lynx"))
    assert state.history[-1].validation_missing_species == ("lynx",)
    assert set(state.history[-1].validation_by_species) == {"fox"}
    selected_observed = PairwiseObservations.from_pairs(
        original.observed_pairs[:-1], original.observed_values[:-1], target=original.target
    )
    for i, (prepared, species, observations, _) in enumerate(groups):
        if i == 2:
            observations = selected_observed
        distances, variances, (nll, beta, mean, covariance) = oracle_group(
            state.model.distance_model.log_weights,
            state.selected_raw_variances[i],
            prepared,
            species,
            observations,
        )
        head = state.model.calibration.for_group(prepared.name, species)
        np.testing.assert_allclose([head.intercept, head.slope], beta, rtol=1e-11)
        np.testing.assert_allclose(head.effect_mean, mean, atol=1e-12)
        np.testing.assert_allclose(head.effect_covariance, covariance, atol=1e-12)
        np.testing.assert_allclose(
            [head.unit_variance, head.residual_variance], variances, rtol=1e-12
        )
    train_distances, variances, (_, beta, _, _) = oracle_group(
        state.distance_model.log_weights, state.raw_variances[2], valley, "fox", selected_observed
    )
    raw_query = state.distance_model.predict_raw_distances(
        valley, species="fox", pairs={"fox": held}
    )["fox"].values[0]
    mean = beta[0] + beta[1] * (raw_query - train_distances.mean()) / train_distances.std(ddof=1)
    variance = 2 * variances[0] + variances[1]
    expected = 0.5 * (np.log(2 * np.pi * variance) + (50 - mean) ** 2 / variance)
    assert state.history[-1].validation_loss is not None
    np.testing.assert_allclose(state.history[-1].validation_loss, expected, rtol=1e-12)
    path = tmp_path / "group-state.eqx"
    eqx.tree_serialise_leaves(path, state)
    restored = eqx.tree_deserialise_leaves(path, state)
    assert restored.history == state.history
    resumed = fit(
        regions,
        observed,
        partition=partitions,
        validation=validation,
        validation_partition=validation_partition,
        state=restored,
        config=replace(config, steps=4),
    )
    uninterrupted = fit(
        regions,
        observed,
        partition=partitions,
        validation=validation,
        validation_partition=validation_partition,
        distance_model=WeightedEmbedding(jnp.zeros(2), species=("fox", "lynx")),
        config=replace(config, steps=4),
    )
    assert resumed.history == uninterrupted.history
    assert eqx.tree_equal(resumed, uninterrupted)
