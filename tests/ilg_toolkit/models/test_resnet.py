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
from ilg_toolkit.models import ResNet9Conductance


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
            patch_batch_size=3,
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
        assert cast(ResNet9Conductance, result.model.distance_model).patch_batch_size == 3
        surface = result.model.conductance_surface(region)
        prediction = result.model.predict(region)
        assert surface.shape == (2, 2)
        assert np.all(np.isfinite(surface) & (surface > 0))
        expected = dense_resistance(surface, np.array([0, 1, 3]))
        np.testing.assert_allclose(prediction.as_matrix(), expected, atol=2e-6, rtol=2e-6)
        np.testing.assert_allclose(
            result.model.ecological_distances(region), prediction.as_matrix()
        )
        assert prediction.target.units == "index"


def test_resnet_parameter_gradient_matches_independent_graph_reference(dense_resistance):
    with jax.enable_x64():
        features = jnp.asarray(np.random.default_rng(8).normal(size=(8, 8, 2)), jnp.float32)
        nodes = np.array([0, 3])
        pixels = np.array([0, 63])
        encoder = ResNet9Conductance(2, patch_size=4, patch_batch_size=3, key=jax.random.key(9))

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


def test_configured_resnet_chunks_match_unchunked_surfaces_and_parameter_gradients():
    features = jnp.asarray(np.random.default_rng(8).normal(size=(8, 12, 2)), jnp.float32)
    key = jax.random.key(19)
    unchunked = ResNet9Conductance(2, patch_size=4, key=key)
    chunked = ResNet9Conductance(2, patch_size=4, patch_batch_size=4, key=key)

    # Six patches exercise a ragged last chunk. GroupNorm operates within each patch.
    expected = unchunked.conductance(features)
    np.testing.assert_allclose(chunked.conductance(features), expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(
        chunked.conductance(features, patch_batch_size=2), expected, rtol=1e-6, atol=1e-7
    )
    value_and_grad = eqx.filter_jit(
        eqx.filter_value_and_grad(lambda encoder: jnp.sum(encoder.conductance(features) ** 2))
    )
    expected_value, expected_gradient = value_and_grad(unchunked)
    actual_value, actual_gradient = value_and_grad(chunked)
    np.testing.assert_allclose(actual_value, expected_value, rtol=1e-6, atol=1e-7)
    for actual, reference in zip(
        jax.tree.leaves(actual_gradient), jax.tree.leaves(expected_gradient), strict=True
    ):
        np.testing.assert_allclose(actual, reference, rtol=3e-5, atol=1e-5)
