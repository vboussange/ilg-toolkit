"""Scientific behavior of models.loglinear."""

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ilg_toolkit import (
    PairwiseObservations,
    RegionBatch,
    ResistanceSolverConfig,
    TargetSpec,
    TrainingConfig,
    fit,
)
from ilg_toolkit.models import LogLinearConductance


@pytest.mark.parametrize("patch_microbatch_size", [None, 3])
def test_patch_mean_predictor_obeys_conductance_bounds(patch_microbatch_size):
    features = np.array(
        [
            [[-4, 0], [-2, 2], [0, 2], [2, 0]],
            [[-2, 0], [0, -2], [2, -2], [4, 0]],
        ],
        dtype=float,
    )
    region = RegionBatch(
        "valley",
        cx.field(features, "row", "column", "feature"),
        ("a", "b"),
        np.array([[0, 0], [0, 3]]),
    )
    model = LogLinearConductance(
        2,
        patch_size=2,
        patch_microbatch_size=patch_microbatch_size,
        min_conductance=0.25,
        max_conductance=5.0,
    )
    np.testing.assert_allclose(model.conductance_surface(region), [[1.25, 1.25]])
    model = eqx.tree_at(
        lambda model: (model.coefficients, model.intercepts),
        model,
        (jnp.array([[0.5, 2.0]]), jnp.array([0.3])),
    )
    # Patch covariates are (-2, 0) and (2, 0), giving predictors -0.7 and 1.3.
    np.testing.assert_allclose(
        model.conductance_surface(region), [[0.7465853037914095, 3.9192966676192444]]
    )
    extreme = eqx.tree_at(lambda model: model.coefficients, model, jnp.array([[1e4, 2.0]]))
    np.testing.assert_allclose(extreme.conductance_surface(region), [[0.25, 5.0]])


@pytest.mark.parametrize(
    "settings",
    [
        {"in_channels": 0},
        {"in_channels": True},
        {"num_species": 0},
        {"num_species": 1.5},
        {"patch_size": 0},
        {"patch_size": True},
        {"patch_microbatch_size": 0},
        {"min_conductance": 0.0},
        {"min_conductance": True},
        {"max_conductance": 0.001},
        {"max_conductance": float("inf")},
        {"max_conductance": float("nan")},
        {"solver_config": "default"},
    ],
)
def test_invalid_model_configuration_is_rejected(settings):
    with pytest.raises((ValueError, TypeError)):
        LogLinearConductance(**({"in_channels": 2} | settings))


def test_training_gradients_and_predictions_match_dense_reference(dense_resistance):
    means = np.array([[[-0.5, 0.4], [0.3, -0.2]], [[-0.1, 0.7], [0.6, -0.9]]])
    regional_means = (means, means[::-1] + 0.2)
    regions = tuple(
        RegionBatch(
            name,
            cx.field(values.repeat(2, axis=0).repeat(2, axis=1), "row", "column", "feature"),
            ("a", "b", "c", "d"),
            np.array([[0, 0], [0, 2], [2, 0], [2, 2]]),
            feature_names=("elevation", "habitat"),
        )
        for name, values in zip(("valley", "ridge"), regional_means, strict=True)
    )
    pairs = (("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"), ("b", "d"), ("c", "d"))
    observations = tuple(
        PairwiseObservations.from_pairs(
            pairs,
            targets,
            target=TargetSpec("dissimilarity"),
        )
        for targets in ([0.4, 0.9, 1.4, 0.6, 1.1, 0.7], [0.5, 0.8, 1.2, 0.7, 1.0, 0.9])
    )
    model = LogLinearConductance(
        2, patch_size=2, patch_microbatch_size=3, solver_config=ResistanceSolverConfig(rtol=1e-10)
    )
    initial = np.array([0.2, -0.1, 0.3])

    def with_parameters(parameters):
        return eqx.tree_at(
            lambda value: (value.coefficients, value.intercepts),
            model,
            (parameters[None, :2], parameters[2:]),
        )

    def reference_distances(parameters, values):
        surface = 0.01 + np.exp(values @ parameters[:2] + parameters[2])
        return dense_resistance(surface, np.array([0, 1, 2, 3]))[np.triu_indices(4, 1)]

    def reference_loss(parameters):
        return np.mean(
            [
                np.mean(
                    (
                        np.log1p(reference_distances(parameters, values))
                        - np.log1p(observed.observed_values)
                    )
                    ** 2
                )
                for values, observed in zip(regional_means, observations, strict=True)
            ]
        )

    def public_loss(parameters):
        distance_model = with_parameters(parameters)
        return sum(
            jnp.mean(
                (
                    jnp.log1p(distance_model.predict(region).values)
                    - jnp.log1p(observed.observed_values)
                )
                ** 2
            )
            for region, observed in zip(regions, observations, strict=True)
        ) / len(regions)

    def reference_gradient(parameters):
        delta = np.eye(3) * 1e-5
        return np.array(
            [
                (reference_loss(parameters + offset) - reference_loss(parameters - offset)) / 2e-5
                for offset in delta
            ]
        )

    np.testing.assert_allclose(
        jax.grad(public_loss)(jnp.asarray(initial)),
        reference_gradient(initial),
        atol=1e-9,
        rtol=1e-6,
    )
    expected = initial.copy()
    first_moment, second_moment = np.zeros(3), np.zeros(3)
    for step in (1, 2):
        gradient = reference_gradient(expected)
        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * gradient**2
        expected -= (
            0.02
            * (first_moment / (1 - 0.9**step))
            / (np.sqrt(second_moment / (1 - 0.999**step)) + 1e-8)
        )
    state = fit(
        regions,
        observations,
        distance_model=with_parameters(jnp.asarray(initial)),
        config=TrainingConfig(objective="direct_log1p", steps=2, learning_rate=0.02),
    )
    np.testing.assert_allclose(
        state.history[-1].training_loss, reference_loss(expected), atol=1e-10
    )
    for region, values in zip(regions, regional_means, strict=True):
        np.testing.assert_allclose(
            state.model.predict(region).values,
            reference_distances(expected, values),
            atol=1e-8,
            rtol=1e-8,
        )


def test_surface_rejects_incompatible_feature_count(observation_problem):
    region, _, _ = observation_problem()
    model = LogLinearConductance(3, patch_size=1)
    with pytest.raises(ValueError):
        model.conductance_surface(region)
