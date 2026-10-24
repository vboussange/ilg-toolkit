"""Scientific behavior of shared distance-model calculations."""

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ilg_toolkit import RegionBatch
from ilg_toolkit.models import ResNet9Conductance, UNet


@pytest.mark.parametrize("family", ["conductance", "embedding"])
def test_patch_microbatch_size_preserves_predictions_and_gradients(family, dense_resistance):
    patch_size = 4 if family == "conductance" else 2
    shape = (2 * patch_size, 3 * patch_size, 2)
    features = np.random.default_rng(8).normal(size=shape).astype(np.float32)
    regions = {
        dtype: RegionBatch(
            "valley",
            cx.field(features.astype(dtype), "row", "column", "feature"),
            ("a", "b", "c"),
            np.array([[0, 0], [0, shape[1] - 1], [shape[0] - 1, shape[1] - 1]]),
        )
        for dtype in ("float32", "float64")
    }
    key = jax.random.key(19)
    if family == "conductance":
        reference = ResNet9Conductance(2, patch_size=4, key=key)
        microbatched = ResNet9Conductance(2, patch_size=4, key=key, patch_microbatch_size=4)
    else:
        reference = UNet(2, patch_size=patch_size, key=key, base_channels=2, embedding_dim=2)
        microbatched = UNet(
            2,
            patch_size=patch_size,
            key=key,
            base_channels=2,
            embedding_dim=2,
            patch_microbatch_size=4,
        )
    models = reference, microbatched

    def objective(model, multiplier, dtype):
        predictions = model.predict(regions[dtype]).values
        surface = model.conductance_surface(regions[dtype]) if family == "conductance" else None
        return jnp.sum(predictions**2) * multiplier, (predictions, surface)

    def gradient_vector(gradient):
        return np.concatenate(
            [np.asarray(leaf, dtype=np.float64).ravel() for leaf in jax.tree.leaves(gradient)]
        )

    evaluate = eqx.filter_jit(eqx.filter_value_and_grad(objective, has_aux=True))
    high_precision_model = jax.tree.map(
        lambda value: value.astype(jnp.float64) if eqx.is_inexact_array(value) else value,
        reference,
    )
    (precise_value, (precise_predictions, _)), precise_gradient = evaluate(
        high_precision_model, jnp.array(1.0, dtype=jnp.float64), "float64"
    )
    precise_gradient = gradient_vector(precise_gradient)
    if family == "conductance":
        direction = gradient_vector(eqx.filter(high_precision_model, eqx.is_inexact_array))
        actual_derivative = np.dot(precise_gradient, direction)
        step, reference_values = 1e-3, []
        for scale in (1 + step, 1 - step):
            scaled = jax.tree.map(
                lambda value, scale=scale: value * scale if eqx.is_inexact_array(value) else value,
                high_precision_model,
            )
            (_, (_, surface)), _ = evaluate(scaled, jnp.array(1.0, dtype=jnp.float64), "float64")
            assert surface is not None
            distances = dense_resistance(np.asarray(surface), np.array([0, 2, 5]))[
                np.triu_indices(3, 1)
            ]
            reference_values.append(np.sum(distances**2))
        expected_derivative = (reference_values[0] - reference_values[1]) / (2 * step)
        assert np.isfinite(actual_derivative) and abs(actual_derivative) > 1e-5
        np.testing.assert_allclose(actual_derivative, expected_derivative, atol=1e-3, rtol=5e-3)
    for dtype in ("float32", "float64"):
        reference, microbatched = (
            jax.tree.map(
                lambda value, dtype=dtype: (
                    value.astype(dtype) if eqx.is_inexact_array(value) else value
                ),
                model,
            )
            for model in models
        )
        # Reference scales give zero derivatives an error budget in the same units as nonzero ones.
        absolute_budget = 8 * np.finfo(dtype).eps
        reference_rtol = 3e-5 if dtype == "float32" else 1e-9
        value_rtol = 3e-6 if dtype == "float32" else 1e-9
        # Six raster patches exercise a final batch smaller than four.
        for multiplier in (1e-3, 1.0, 1e3):
            scale = jnp.asarray(multiplier, dtype=dtype)
            (expected_value, (expected_predictions, _)), expected_gradient = evaluate(
                reference, scale, dtype
            )
            (actual_value, (actual_predictions, _)), actual_gradient = evaluate(
                microbatched, scale, dtype
            )
            expected_gradient = gradient_vector(expected_gradient)
            actual_gradient = gradient_vector(actual_gradient)
            comparisons = (
                (actual_predictions, expected_predictions, precise_predictions, value_rtol),
                (actual_value, expected_value, precise_value * multiplier, value_rtol),
                (actual_gradient, expected_gradient, precise_gradient * multiplier, reference_rtol),
            )
            for actual, expected, precise, parity_rtol in comparisons:
                actual, expected, precise = (
                    np.asarray(x, dtype=np.float64) for x in (actual, expected, precise)
                )
                magnitude = np.max(np.abs(precise))
                assert magnitude > 0
                np.testing.assert_allclose(
                    actual / magnitude,
                    expected / magnitude,
                    rtol=parity_rtol,
                    atol=absolute_budget,
                )
                for computed in (actual, expected):
                    np.testing.assert_allclose(
                        computed / magnitude,
                        precise / magnitude,
                        rtol=reference_rtol,
                        atol=absolute_budget,
                    )
            for computed in (actual_gradient, expected_gradient):
                relative_error = np.linalg.norm(
                    computed / multiplier - precise_gradient
                ) / np.linalg.norm(precise_gradient)
                assert relative_error < reference_rtol
