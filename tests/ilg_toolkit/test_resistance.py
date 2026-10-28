"""Tests of the effective-resistance layer and resistance solver settings."""

import coordax as cx
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from ilg_toolkit import RegionBatch, ResistanceSolverConfig


def _region(features, positions):
    """Returns a region batch with one elevation feature and one sampling unit per position."""
    return RegionBatch(
        "plot",
        cx.field(
            jnp.asarray(features)[..., None],
            "row",
            "column",
            cx.LabeledAxis("feature", np.array(["elevation"])),
        ),
        tuple(f"u{index}" for index in range(len(positions))),
        np.asarray(positions),
    )


@pytest.mark.filterwarnings("ignore:Sampling units share a vertex")
def test_amg_resistances_and_gradients_match_dense_laplacian(FeatureConductance, dense_resistance):
    pytest.importorskip("amjax")
    pytest.importorskip("pyamg")
    log_surface = np.log([[1.0, 1.4], [0.8, 2.0]])
    nodes = np.array([3, 0, 1, 1])
    region = _region(np.ones((2, 2)), [[1, 1], [0, 0], [0, 1], [0, 1]])
    model = FeatureConductance(
        jnp.asarray(log_surface), solver_config=ResistanceSolverConfig(use_amg=True)
    )
    upper = np.triu_indices(nodes.size, 1)
    prediction = model.predict(region)
    np.testing.assert_allclose(
        prediction.values,
        dense_resistance(np.exp(log_surface), nodes)[upper],
        atol=1e-8,
        rtol=1e-8,
    )
    assert prediction.values.dtype == jnp.float64

    gradient = eqx.filter_grad(lambda distance_model: distance_model.predict(region).values.sum())(
        model
    ).weight
    finite_difference = np.zeros_like(log_surface)
    for index in np.ndindex(log_surface.shape):
        offset = np.zeros_like(log_surface)
        offset[index] = 1e-6
        finite_difference[index] = (
            dense_resistance(np.exp(log_surface + offset), nodes)[upper].sum()
            - dense_resistance(np.exp(log_surface - offset), nodes)[upper].sum()
        ) / 2e-6
    np.testing.assert_allclose(gradient, finite_difference, atol=1e-7, rtol=1e-6)


def test_effective_resistance_rejects_nonpositive_conductances(PointConductance):
    region = _region([[1.0, 0.0]], [[0, 0], [0, 1]])
    with pytest.raises(RuntimeError, match="finite and strictly positive"):
        PointConductance(jnp.asarray(0.0)).predict(region)


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
