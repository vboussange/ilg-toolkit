"""Tests of the effective-resistance layer."""

import importlib.util

import coordax as cx
import equinox as eqx
import jax
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
        jnp.asarray(log_surface), solver_config=ResistanceSolverConfig(solver="amg")
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


def _dense_resistance_sum(log_surface, nodes):
    """Returns the summed effective resistances of vertex pairs from a dense grounded Laplacian."""
    surface = jnp.exp(log_surface).ravel()
    vertices = np.arange(surface.size).reshape(log_surface.shape)
    first = np.concatenate([vertices[:-1, :].ravel(), vertices[:, :-1].ravel()])
    second = np.concatenate([vertices[1:, :].ravel(), vertices[:, 1:].ravel()])
    weights = (surface[first] + surface[second]) / 2
    laplacian = (
        jnp.zeros((surface.size, surface.size))
        .at[first, first]
        .add(weights)
        .at[second, second]
        .add(weights)
        .at[first, second]
        .add(-weights)
        .at[second, first]
        .add(-weights)
    )
    inverse = jnp.pad(jnp.linalg.inv(laplacian[:-1, :-1]), (0, 1))[nodes[:, None], nodes[None, :]]
    diagonal = jnp.diag(inverse)
    return jnp.triu(diagonal[:, None] + diagonal[None, :] - 2 * inverse, 1).sum()


@pytest.fixture(params=["cudss_double", "cudss_gpu"])
def cudss_backend(request):
    """Runs a test on the CPU double and, with spineax and a GPU backend, on cuDSS."""
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize(
    "surface",
    [
        np.random.default_rng(3).uniform(0.5, 2.0, size=(4, 5)),
        10.0 ** np.random.default_rng(3).uniform(-3.0, 3.0, size=(4, 5)),
        np.where(np.random.default_rng(3).random((4, 5)) < 0.5, 1e-6, 1.0),
        np.where(np.arange(5) == 2, 1e-6, 1.0) * np.ones((4, 5)),
    ],
    ids=["well_conditioned", "six_orders_of_magnitude", "floor_patches", "floor_barrier"],
)
def test_cudss_resistances_and_gradients_match_dense_laplacian(
    cudss_backend, surface, FeatureConductance, dense_resistance
):
    log_surface = np.log(surface)
    nodes = np.array([0, 19, 4, 11])
    region = _region(np.ones(surface.shape), [[0, 0], [3, 4], [0, 4], [2, 1]])
    model = FeatureConductance(
        jnp.asarray(log_surface), solver_config=ResistanceSolverConfig(solver="cudss")
    )
    left, right = np.triu_indices(nodes.size, 1)
    expected = dense_resistance(surface, nodes)[left, right]
    prediction = model.predict(region)
    np.testing.assert_allclose(prediction.values, expected, rtol=1e-6)
    assert prediction.values.dtype == jnp.float64
    latent = model.backbone(region.feature_array)
    pairs = (jnp.asarray(nodes), (jnp.asarray(left), jnp.asarray(right)))
    np.testing.assert_allclose(model.ecological_distances(latent, pairs, 0), expected, rtol=1e-6)

    gradient = eqx.filter_grad(lambda distance_model: distance_model.predict(region).values.sum())(
        model
    ).weight
    expected_gradient = jax.grad(_dense_resistance_sum)(jnp.asarray(log_surface), nodes)
    np.testing.assert_allclose(
        gradient, expected_gradient, rtol=1e-5, atol=1e-5 * np.abs(expected_gradient).max()
    )


def test_cudss_matches_conjugate_gradient_on_a_well_conditioned_surface(
    cudss_double, FeatureConductance
):
    log_surface = np.log(np.random.default_rng(3).uniform(0.5, 2.0, size=(4, 5)))
    region = _region(np.ones((4, 5)), [[0, 0], [3, 4], [0, 4], [2, 1]])
    direct, iterative = (
        FeatureConductance(jnp.asarray(log_surface), solver_config=config).predict(region)
        for config in (
            ResistanceSolverConfig(solver="cudss"),
            ResistanceSolverConfig(solver="cg", rtol=1e-11, atol=1e-11),
        )
    )
    np.testing.assert_allclose(direct.values, iterative.values, rtol=1e-8)


def test_failed_cudss_factorization_raises(cudss_double, PointConductance):
    region = _region([[1.0, 1.0]], [[0, 0], [0, 1]])
    model = PointConductance(jnp.asarray(0.0), solver_config=ResistanceSolverConfig(solver="cudss"))
    cudss_double.failing = True
    with pytest.raises(RuntimeError):
        model.predict(region)


def test_cudss_without_its_backend_fails_naming_the_extra(PointConductance):
    if jax.default_backend() == "gpu" and importlib.util.find_spec("spineax") is not None:
        pytest.skip("cuDSS has its backend")
    region = _region([[1.0, 1.0]], [[0, 0], [0, 1]])
    model = PointConductance(jnp.asarray(0.0), solver_config=ResistanceSolverConfig(solver="cudss"))
    with pytest.raises((ImportError, RuntimeError), match=r"ilg-toolkit\[cudss\]") as caught:
        model.predict(region)
    assert caught.value.__cause__ is not None


@pytest.mark.parametrize("error", [ImportError("no spineax"), RuntimeError("no GPU backend")])
def test_cudss_backend_errors_keep_their_type_and_cause(PointConductance, monkeypatch, error):
    def unavailable():
        raise error

    monkeypatch.setattr("ilg_toolkit.resistance.CuDSSSolver", unavailable)
    region = _region([[1.0, 1.0]], [[0, 0], [0, 1]])
    model = PointConductance(jnp.asarray(0.0), solver_config=ResistanceSolverConfig(solver="cudss"))
    with pytest.raises(type(error), match=r"ilg-toolkit\[cudss\].*" + str(error)) as caught:
        model.predict(region)
    assert caught.value.__cause__ is error


def test_effective_resistance_rejects_nonpositive_conductances(PointConductance):
    region = _region([[1.0, 0.0]], [[0, 0], [0, 1]])
    with pytest.raises(RuntimeError, match="finite and strictly positive"):
        PointConductance(jnp.asarray(0.0)).predict(region)
