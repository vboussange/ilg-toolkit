"""Actual graph calculation checked against independent tiny-graph references."""

from typing import final

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
from ilg_toolkit.models import ConductanceModel, ResNet9Conductance
from ilg_toolkit.resistance import build_resistance_context, effective_resistance


def dense_resistance(surface, nodes):
    """NumPy full Laplacian oracle; shares no production graph/solve helpers."""
    rows, columns = surface.shape
    laplacian = np.zeros((surface.size, surface.size))
    for row in range(rows):
        for column in range(columns):
            first = row * columns + column
            for next_row, next_column in ((row + 1, column), (row, column + 1)):
                if next_row < rows and next_column < columns:
                    second = next_row * columns + next_column
                    weight = (surface[row, column] + surface[next_row, next_column]) / 2
                    incidence = np.zeros(surface.size)
                    incidence[first], incidence[second] = 1, -1
                    laplacian += weight * np.outer(incidence, incidence)
    inverse = np.linalg.pinv(laplacian, hermitian=True)
    diagonal = np.diag(inverse)
    matrix = diagonal[:, None] + diagonal[None, :] - 2 * inverse
    return matrix[nodes[:, None], nodes[None, :]]


@pytest.mark.parametrize("use_amg", [False, True])
def test_actual_resistance_and_gradients_match_independent_laplacian(use_amg):
    if use_amg:
        pytest.importorskip("amjax")
        pytest.importorskip("pyamg")
    with jax.enable_x64():
        surface = np.array([[1.0, 1.4], [0.8, 2.0]])
        nodes = np.array([3, 0, 1, 1])
        context = build_resistance_context(surface.shape, ResistanceSolverConfig(use_amg=use_amg))
        actual = effective_resistance(jnp.asarray(surface), nodes, context=context)
        expected = dense_resistance(surface, nodes)
        np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-8)
        gradient = jax.grad(
            lambda values: effective_resistance(values, nodes, context=context).sum()
        )(jnp.asarray(surface))
        finite_difference = np.zeros_like(surface)
        for index in np.ndindex(surface.shape):
            offset = np.zeros_like(surface)
            offset[index] = 1e-5
            finite_difference[index] = (
                dense_resistance(surface + offset, nodes).sum()
                - dense_resistance(surface - offset, nodes).sum()
            ) / 2e-5
        np.testing.assert_allclose(gradient, finite_difference, atol=1e-7, rtol=1e-6)


def test_resnet_fit_exposes_surface_and_label_free_target_predictions():
    with jax.enable_x64():
        features = np.random.default_rng(4).normal(size=(8, 8, 2)).astype(np.float32)
        region = RegionBatch(
            "small-valley",
            features,
            ("south", "north", "east"),
            np.array([[0, 0], [0, 7], [7, 7]]),
        )
        observations = PairwiseObservations.from_matrix(
            region.sampling_unit_ids,
            dense_resistance(np.full((2, 2), 2.0), np.array([0, 1, 3])),
            target=TargetSpec("synthetic dissimilarity", units="index"),
        )
        model = ResNet9Conductance(in_channels=2, patch_size=4, key=jax.random.key(6))
        result = fit(
            region,
            observations,
            model=model,
            config=TrainingConfig(
                epochs=4, learning_rate=0.001, solver=ResistanceSolverConfig(rtol=1e-8)
            ),
        )
        assert result.history[-1].training_loss < result.history[0].training_loss
        surface = result.model.conductance_surface(region)
        prediction = result.model.predict(region)
        assert surface.shape == (2, 2)
        assert np.all(np.isfinite(surface) & (surface > 0))
        expected = dense_resistance(surface, np.array([0, 1, 3]))
        np.testing.assert_allclose(prediction.values, expected, atol=2e-6, rtol=2e-6)
        np.testing.assert_allclose(result.model.landscape_scores(region), prediction.values)
        assert prediction.target.units == "index"


@final
class PointConductance(ConductanceModel):
    """Tiny learnable encoder keeps failure tests independent of CNN compilation."""

    log_scale: jax.Array
    patch_size: int = eqx.field(static=True, default=1)

    def conductance(self, features, *, patch_batch_size=None):
        return jnp.exp(self.log_scale) * features[..., 0]


def test_failed_solver_reports_region_and_epoch_without_changing_solver():
    with jax.enable_x64():
        region = RegionBatch(
            "bad-solve",
            np.linspace(0.1, 2, 12).reshape(3, 4, 1),
            ("first", "last"),
            np.array([[0, 0], [2, 3]]),
        )
        observations = PairwiseObservations.from_matrix(
            region.sampling_unit_ids, [[0, 1], [1, 0]], target=TargetSpec("dissimilarity")
        )
        with pytest.raises(RuntimeError, match="bad-solve.*epoch 0"):
            fit(
                region,
                observations,
                model=PointConductance(jnp.array(0.0)),
                config=TrainingConfig(
                    epochs=0, solver=ResistanceSolverConfig(rtol=1e-12, atol=1e-12, max_steps=1)
                ),
            )


def test_resnet_parameter_gradient_matches_independent_graph_reference():
    with jax.enable_x64():
        features = jnp.asarray(np.random.default_rng(8).normal(size=(8, 8, 2)), jnp.float32)
        nodes = np.array([0, 3])
        pixels = np.array([0, 63])
        encoder = ResNet9Conductance(2, patch_size=4, key=jax.random.key(9))

        def scaled_encoder(scale):
            return jax.tree.map(
                lambda value: value * scale if eqx.is_inexact_array(value) else value, encoder
            )

        actual_gradient = jax.grad(
            lambda scale: scaled_encoder(scale).predict_distances(features, pixels)[0, 1]
        )(jnp.array(1.0))
        step = 1e-3
        independent_values = [
            dense_resistance(np.asarray(scaled_encoder(scale).conductance(features)), nodes)[0, 1]
            for scale in (1 + step, 1 - step)
        ]
        expected_gradient = (independent_values[0] - independent_values[1]) / (2 * step)
        assert np.isfinite(actual_gradient) and abs(float(actual_gradient)) > 1e-5
        np.testing.assert_allclose(actual_gradient, expected_gradient, atol=1e-3, rtol=5e-3)


def test_solver_rejects_silent_float64_truncation_and_invalid_terminals():
    with jax.enable_x64(False), pytest.raises(RuntimeError, match="JAX_ENABLE_X64"):
        effective_resistance(jnp.ones((2, 2), dtype=jnp.float32), np.array([0, 3]))
    with jax.enable_x64():
        with pytest.raises(ValueError, match="integer"):
            effective_resistance(jnp.ones((2, 2)), np.array([0.0, 1.5]))
        with pytest.raises(RuntimeError, match="outside"):
            effective_resistance(jnp.ones((2, 2)), np.array([0, 4]))
        with pytest.raises(RuntimeError, match="strictly positive"):
            effective_resistance(jnp.array([[1.0, 0.0]]), np.array([0, 1]))


@pytest.mark.parametrize(
    "options",
    [
        dict(rtol=True),
        dict(atol=float("nan")),
        dict(rtol=0, atol=0),
        dict(max_steps=1.5),
        dict(max_steps=False),
        dict(use_amg=1),
    ],
)
def test_solver_settings_reject_invalid_options(options):
    with pytest.raises(ValueError, match="solver|tolerance"):
        ResistanceSolverConfig(**options)
