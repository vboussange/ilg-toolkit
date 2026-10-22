"""Scientific behavior of models.resnet."""

from typing import cast

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ilg_toolkit import (
    PairwiseObservations,
    RegionBatch,
    ResistanceSolverConfig,
    TargetSpec,
    TrainingConfig,
    fit,
)
from ilg_toolkit.models import ConductanceModel, ResNet9Conductance


def test_resnet_fit_exposes_surface_and_label_free_target_predictions(dense_resistance):
    with jax.enable_x64():
        features = np.random.default_rng(4).normal(size=(8, 8, 2)).astype(np.float32)
        region = RegionBatch(
            "small-valley",
            cx.field(features, "row", "column", "feature"),
            ("south", "north", "east"),
            np.array([[0, 0], [0, 7], [7, 7]]),
        )
        observations = PairwiseObservations.from_matrix(
            region.sampling_unit_ids,
            dense_resistance(np.full((2, 2), 2.0), np.array([0, 1, 3])),
            target=TargetSpec("synthetic dissimilarity", units="index"),
        )
        model = ResNet9Conductance(
            in_channels=2,
            patch_size=4,
            patch_microbatch_size=3,
            solver_config=ResistanceSolverConfig(rtol=1e-8),
            key=jax.random.key(6),
        )
        result = fit(
            region,
            observations,
            distance_model=model,
            config=TrainingConfig(
                objective="direct_log1p",
                epochs=4,
                learning_rate=0.001,
            ),
        )
        assert result.history[-1].training_loss < result.history[0].training_loss
        surface = cast(ConductanceModel, result.model.distance_model).conductance_surface(region)
        prediction = result.model.predict(region)
        assert surface.shape == (2, 2)
        assert np.all(np.isfinite(surface) & (surface > 0))
        expected = dense_resistance(surface, np.array([0, 1, 3]))
        np.fill_diagonal(expected, np.nan)
        np.testing.assert_allclose(prediction.as_matrix(), expected, atol=2e-6, rtol=2e-6)
        np.testing.assert_allclose(
            result.model.distance_model.predict(region).as_matrix(), prediction.as_matrix()
        )


def test_resnet_parameter_gradient_matches_independent_graph_reference(
    dense_resistance,
):
    with jax.enable_x64():
        features = jnp.asarray(np.random.default_rng(8).normal(size=(8, 8, 2)), jnp.float32)
        nodes = np.array([0, 3])
        region = RegionBatch(
            "reference",
            cx.field(features, "row", "column", "feature"),
            ("a", "b"),
            np.array([[0, 0], [7, 7]]),
        )
        encoder = ResNet9Conductance(
            2, patch_size=4, patch_microbatch_size=3, key=jax.random.key(9)
        )

        def scaled_encoder(scale):
            return jax.tree.map(
                lambda value: value * scale if eqx.is_inexact_array(value) else value, encoder
            )

        actual_gradient = jax.grad(lambda scale: scaled_encoder(scale).predict(region).values[0])(
            jnp.array(1.0)
        )
        step = 1e-3
        independent_values = [
            dense_resistance(np.asarray(scaled_encoder(scale).conductance_surface(region)), nodes)[
                0, 1
            ]
            for scale in (1 + step, 1 - step)
        ]
        expected_gradient = (independent_values[0] - independent_values[1]) / (2 * step)
        assert np.isfinite(actual_gradient) and abs(float(actual_gradient)) > 1e-5
        np.testing.assert_allclose(actual_gradient, expected_gradient, atol=1e-3, rtol=5e-3)
