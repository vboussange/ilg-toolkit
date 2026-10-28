"""Shared checks for neural distance models covered at tiny width."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ilg_toolkit import RegionBatch
from ilg_toolkit.models import AbstractDistanceModel, ConductanceModel


def _gradient_vector(gradient):
    leaves = jax.tree.leaves(eqx.filter(gradient, eqx.is_inexact_array))
    return np.concatenate([np.asarray(leaf, dtype=np.float64).ravel() for leaf in leaves])


def _assert_patch_microbatches_match(
    reference: AbstractDistanceModel,
    microbatched: AbstractDistanceModel,
    region: RegionBatch,
    *,
    rtol: float,
) -> jax.Array | None:
    """Assert equal predictions, gradients, and any conductance surface; return that surface."""

    @eqx.filter_jit
    @eqx.filter_value_and_grad(has_aux=True)
    def objective(model):
        predictions = model.predict(region).values
        surface = model.conductance_surface(region) if isinstance(model, ConductanceModel) else None
        return jnp.sum(predictions**2), (predictions, surface)

    (expected_value, (expected, expected_surface)), expected_gradient = objective(reference)
    (actual_value, (actual, actual_surface)), actual_gradient = objective(microbatched)
    assert expected.shape == (len(region.sampling_unit_ids),)
    assert np.all(np.isfinite(expected) & (expected > 0))
    np.testing.assert_allclose(actual, expected, rtol=rtol)
    np.testing.assert_allclose(actual_value, expected_value, rtol=rtol)
    if expected_surface is not None:
        assert actual_surface is not None
        assert np.all(np.isfinite(expected_surface) & (expected_surface > 0))
        np.testing.assert_allclose(actual_surface, expected_surface, rtol=rtol)
    expected_gradient = _gradient_vector(expected_gradient)
    actual_gradient = _gradient_vector(actual_gradient)
    assert np.all(np.isfinite(expected_gradient)) and np.any(expected_gradient != 0)
    # One scale for all leaves: parameters with vanishing gradients carry only roundoff.
    scale = np.max(np.abs(expected_gradient))
    np.testing.assert_allclose(actual_gradient / scale, expected_gradient / scale, atol=1e-5)
    return expected_surface


@pytest.fixture(name="assert_patch_microbatches_match", scope="session")
def assert_patch_microbatches_match_fixture():
    return _assert_patch_microbatches_match
