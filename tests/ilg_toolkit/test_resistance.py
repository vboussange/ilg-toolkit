"""Scientific behavior of resistance."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ilg_toolkit import ResistanceSolverConfig
from ilg_toolkit.models import LogLinearConductance
from ilg_toolkit.models.base import prepare_distance_context
from ilg_toolkit.resistance import effective_resistance


def _context(shape, config=None):
    """Resistance solver for a conductance grid graph of the given shape."""
    model = LogLinearConductance(1, patch_size=1, solver_config=config)
    return prepare_distance_context(model, jnp.zeros((*shape, 1)))


def test_amg_resistance_and_gradients_match_independent_laplacian(dense_resistance):
    # The default conjugate-gradient solver meets the same dense Laplacian reference
    # through the public log-linear and conductance training tests.
    pytest.importorskip("amjax")
    pytest.importorskip("pyamg")
    with jax.enable_x64():
        surface = np.array([[1.0, 1.4], [0.8, 2.0]])
        nodes = np.array([3, 0, 1, 1])
        context = _context(surface.shape, ResistanceSolverConfig(use_amg=True))
        actual = effective_resistance(surface, nodes, context=context)
        expected = dense_resistance(surface, nodes)
        np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(
            effective_resistance(surface.tolist(), nodes.tolist(), context=context),
            expected,
            atol=1e-8,
            rtol=1e-8,
        )
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


def test_solver_rejects_invalid_terminals_conductances_and_mismatched_contexts():
    with jax.enable_x64():
        square, row = _context((2, 2)), _context((1, 2))
        with pytest.raises(ValueError):
            effective_resistance(jnp.ones((2, 2)), np.array([0.0, 1.5]), context=square)
        with pytest.raises(RuntimeError):
            effective_resistance(jnp.ones((2, 2)), np.array([0, 4]), context=square)
        with pytest.raises(RuntimeError):
            effective_resistance(jnp.array([[1.0, 0.0]]), np.array([0, 1]), context=row)
        with pytest.raises(ValueError, match="does not match surface"):
            effective_resistance(jnp.ones((2, 2)), np.array([0, 1]), context=row)


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
    with pytest.raises(ValueError):
        ResistanceSolverConfig(**options)
