"""Public bounded updates checked against an independent likelihood direction."""

from dataclasses import replace

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from test_mlpe_training import dense_profile

from ilg_toolkit import (
    PairwiseObservations,
    RegionBatch,
    SamplingUnits,
    TargetSpec,
    TrainingConfig,
    fit,
)
from ilg_toolkit.models import UNetEmbeddingDistance


@pytest.mark.parametrize("objective", ["direct_log1p", "mlpe"])
@pytest.mark.parametrize("species_count", [1, 2])
def test_bounded_neural_adam_matches_public_distance_likelihood_direction(objective, species_count):
    # Float64 isolates the chain rule from Adam amplification of float32
    # translation-null normalization biases, examined separately in the report.
    with jax.enable_x64():
        rng = np.random.default_rng(714)
        species = ("fox", "lynx")[:species_count]
        units = SamplingUnits(("a", "b", "c", "d"), [[0, 0], [0, 3], [3, 0], [3, 3]])
        region = RegionBatch(
            "forest",
            cx.field(rng.normal(size=(4, 4, 2)), "row", "column", "feature"),
            {name: units for name in species},
        )
        left, right = np.triu_indices(4, 1)
        targets = np.array([0.3, 0.8, 0.9, 0.2, 0.6, 0.4])
        observations = PairwiseObservations.from_pairs(
            [(units.ids[a], units.ids[b]) for a, b in zip(left, right, strict=True)],
            targets,
            target=TargetSpec("synthetic dissimilarity", units="index"),
        )
        model = UNetEmbeddingDistance(
            2,
            species=species,
            patch_size=1,
            base_channels=2,
            embedding_dim=2,
            dropout=0.0,
            key=jax.random.key(715),
        )
        model = jax.tree.map(
            lambda value: value.astype(jnp.float64) if eqx.is_inexact_array(value) else value,
            model,
        )
        config = TrainingConfig(
            objective=objective,
            steps=1,
            seed=716,
            learning_rate=1e-4,
            mlpe_initial_variances=(0.2, 0.4),
            species_chunk_size=4,
        )
        observed = {region.name: {name: observations for name in species}}
        state = fit([region], observed, distance_model=model, config=config)
        assert isinstance(state.optimizer_state, tuple)
        adam = state.optimizer_state[0]
        assert isinstance(adam, optax.ScaleByAdamState)
        initial_raw = jnp.asarray(np.log(np.expm1(np.tile([0.2, 0.4], (species_count, 1)) - 1e-10)))
        parameters = model if objective == "direct_log1p" else (model, initial_raw)
        latest = (
            state.latest_distance_model
            if objective == "direct_log1p"
            else (
                state.latest_distance_model,
                state.raw_variances,
            )
        )
        direction = jax.tree.map(
            lambda value: jnp.asarray(
                rng.normal(size=value.shape) * 0.1 / np.sqrt(max(value.size, 1)),
                dtype=value.dtype,
            ),
            eqx.filter(parameters, eqx.is_inexact_array),
        )

        def independent_objective(values):
            current, raw = (values, None) if objective == "direct_log1p" else values
            matrices = current.ecological_distances(region, species=species)
            losses = []
            for index, name in enumerate(species):
                distances = np.asarray(matrices[name])[left, right]
                if raw is None:
                    losses.append(np.mean((np.log1p(distances) - np.log1p(targets)) ** 2))
                else:
                    variances = np.logaddexp(0, np.asarray(raw)[index]) + 1e-10
                    losses.append(
                        dense_profile(distances, targets, left, right, 4, variances)[0]
                        / len(targets)
                    )
            return np.mean(losses)

        epsilon = 1e-4
        shifted = [
            eqx.apply_updates(
                parameters,
                jax.tree.map(
                    lambda value, multiplier=sign: multiplier * epsilon * value, direction
                ),
            )
            for sign in (1, -1)
        ]
        expected_derivative = (
            independent_objective(shifted[0]) - independent_objective(shifted[1])
        ) / (2 * epsilon)
        actual_derivative = sum(
            float(jnp.vdot(moment / 0.1, tangent))
            for moment, tangent in zip(
                jax.tree.leaves(adam.mu), jax.tree.leaves(direction), strict=True
            )
        )
        np.testing.assert_allclose(actual_derivative, expected_derivative, rtol=2e-5, atol=1e-7)
        for before, after, mean, variance in zip(
            jax.tree.leaves(eqx.filter(parameters, eqx.is_inexact_array)),
            jax.tree.leaves(eqx.filter(latest, eqx.is_inexact_array)),
            jax.tree.leaves(adam.mu),
            jax.tree.leaves(adam.nu),
            strict=True,
        ):
            np.testing.assert_allclose(variance, 0.001 * (mean / 0.1) ** 2, rtol=1e-10, atol=1e-16)
            expected = before - config.learning_rate * (mean / 0.1) / (
                jnp.sqrt(variance / 0.001) + 1e-8
            )
            np.testing.assert_allclose(after, expected, rtol=1e-9, atol=1e-12)
        resumed = fit([region], observed, state=state, config=replace(config, steps=2))
        uninterrupted = fit(
            [region], observed, distance_model=model, config=replace(config, steps=2)
        )
        assert eqx.tree_equal(resumed, uninterrupted)
        if species_count > 1:
            split = fit(
                [region],
                observed,
                distance_model=model,
                config=replace(config, species_chunk_size=1),
            )
            for actual, expected in zip(
                jax.tree.leaves(
                    (state.latest_distance_model, state.raw_variances, state.optimizer_state)
                ),
                jax.tree.leaves(
                    (split.latest_distance_model, split.raw_variances, split.optimizer_state)
                ),
                strict=True,
            ):
                np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=1e-5)
            for actual, expected in zip(
                jax.tree.leaves(state.optimizer_state),
                jax.tree.leaves(split.optimizer_state),
                strict=True,
            ):
                np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=3e-8)
