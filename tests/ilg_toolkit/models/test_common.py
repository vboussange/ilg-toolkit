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
def test_patch_microbatch_size_preserves_predictions_and_gradients(family):
    features = np.random.default_rng(8).normal(size=(8, 12, 2)).astype(np.float32)
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
    # Six raster patches exercise a final batch smaller than four.
    np.testing.assert_allclose(
        microbatched.predict(region).values,
        reference.predict(region).values,
        rtol=1e-6,
        atol=1e-7,
    )
    value_and_grad = eqx.filter_jit(
        eqx.filter_value_and_grad(lambda model: jnp.sum(model.predict(region).values ** 2))
    )
    expected_value, expected_gradient = value_and_grad(reference)
    actual_value, actual_gradient = value_and_grad(microbatched)
    np.testing.assert_allclose(actual_value, expected_value, rtol=1e-6, atol=1e-7)
    for actual, expected in zip(
        jax.tree.leaves(actual_gradient), jax.tree.leaves(expected_gradient), strict=True
    ):
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=1e-5)
