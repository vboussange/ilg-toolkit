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
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_patch_microbatch_size_preserves_predictions_and_gradients(family, dtype):
    features = np.random.default_rng(8).normal(size=(8, 12, 2)).astype(np.float32).astype(dtype)
    region = RegionBatch(
        "valley",
        cx.field(features, "row", "column", "feature"),
        ("a", "b", "c"),
        np.array([[0, 0], [0, 11], [7, 11]]),
    )
    key = jax.random.key(19)
    if family == "conductance":
        reference = ResNet9Conductance(2, patch_size=4, key=key)
        microbatched = ResNet9Conductance(2, patch_size=4, key=key, patch_microbatch_size=4)
    else:
        reference = UNet(2, patch_size=4, key=key, base_channels=2, embedding_dim=2)
        microbatched = UNet(
            2, patch_size=4, key=key, base_channels=2, embedding_dim=2, patch_microbatch_size=4
        )
    reference, microbatched = (
        jax.tree.map(
            lambda value: value.astype(dtype) if eqx.is_inexact_array(value) else value, model
        )
        for model in (reference, microbatched)
    )

    def objective(model, multiplier):
        predictions = model.predict(region).values
        return jnp.sum(predictions**2) * multiplier, predictions

    def gradient_vector(gradient):
        return np.concatenate(
            [np.asarray(leaf, dtype=np.float64).ravel() for leaf in jax.tree.leaves(gradient)]
        )

    evaluate = eqx.filter_jit(eqx.filter_value_and_grad(objective, has_aux=True))
    high_precision_model = jax.tree.map(
        lambda value: value.astype(jnp.float64) if eqx.is_inexact_array(value) else value,
        reference,
    )
    (precise_value, precise_predictions), precise_gradient = evaluate(
        high_precision_model, jnp.array(1.0, dtype=jnp.float64)
    )
    precise_gradient = gradient_vector(precise_gradient)
    # Reference scales give zero derivatives an error budget in the same units as nonzero ones.
    absolute_budget = 8 * np.finfo(dtype).eps
    reference_rtol = 3e-5 if dtype == "float32" else 1e-9
    value_rtol = 3e-6 if dtype == "float32" else 1e-9
    # Six raster patches exercise a final batch smaller than four.
    for multiplier in (1e-3, 1.0, 1e3):
        scale = jnp.asarray(multiplier, dtype=dtype)
        (expected_value, expected_predictions), expected_gradient = evaluate(reference, scale)
        (actual_value, actual_predictions), actual_gradient = evaluate(microbatched, scale)
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
