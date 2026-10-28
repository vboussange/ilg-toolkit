"""Scientific behavior of models.base."""

from dataclasses import replace
from typing import cast

import coordax as cx
import equinox as eqx
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from ilg_toolkit import (
    load_model,
    Log1pLoss,
    PairwiseObservations,
    RegionBatch,
    save_model,
    TargetSpec,
    train,
    TrainingConfig,
)
from ilg_toolkit.models import ConductanceModel, LogLinearConductance
from jaxscape import GridGraph, ResistanceDistance


def test_species_queries_follow_supplied_order_with_reused_sampling_unit_ids():
    region = RegionBatch(
        "river",
        cx.field(np.ones((1, 3, 1)), "row", "column", "feature"),
        ("a", "b", "a", "b"),
        np.array([[0, 0], [0, 2], [0, 0], [0, 1]]),
        species=("hare", "hare", "stoat", "stoat"),
    )
    model = LogLinearConductance(1, num_species=2, patch_size=1)
    model = eqx.tree_at(lambda value: value.intercepts, model, jnp.log(jnp.array([1.99, 0.99])))
    order = ("stoat", "hare")
    results = model.predict(region, species=("hare", "stoat"), species_order=order)
    np.testing.assert_allclose(results["hare"].values, [2.0], rtol=1e-6)
    np.testing.assert_allclose(results["stoat"].values, [0.5], rtol=1e-6)
    for name, result in results.items():
        assert result.species == name
        assert result.sampling_unit_ids == ("a", "b")
        np.testing.assert_array_equal(
            model.predict(region, species=name, species_order=order).values, result.values
        )
    surfaces = model.conductance_surface(region, species=("hare", "stoat"), species_order=order)
    np.testing.assert_allclose(surfaces["hare"], np.ones((1, 3)), rtol=1e-6)
    np.testing.assert_allclose(surfaces["stoat"], np.full((1, 3), 2), rtol=1e-6)
    unlabeled = replace(region, sampling_unit_ids=("a", "b", "c", "d"), species=None)
    np.testing.assert_allclose(
        model.conductance_surface(unlabeled, species="hare", species_order=order),
        surfaces["hare"],
    )
    partial = replace(unlabeled, species=("stoat",) * 4)
    np.testing.assert_allclose(
        model.conductance_surface(partial, species="hare", species_order=order),
        surfaces["hare"],
    )
    for options in (
        {},
        {"species": "hare"},
        {"species_order": order},
        {"species": "missing", "species_order": order},
        {"species": ("hare", "hare"), "species_order": order},
        {"species": "hare", "species_order": ("hare", "hare")},
    ):
        with pytest.raises(ValueError):
            model.predict(region, **options)


def test_requested_float32_resistances_and_gradients_match_jaxscape(CovariateConductance):
    # Float32 features exercise the float64 solve and cast back. Float64 resistances
    # meet the dense Laplacian reference in the log-linear and conductance training tests.
    dtype = jnp.float32
    region = RegionBatch(
        "river",
        cx.field(
            jnp.array([[[0.3], [0.8], [0.2]], [[0.5], [0.1], [0.7]]], dtype=dtype),
            "row",
            "column",
            "feature",
        ),
        ("a", "b", "c", "d", "e"),
        np.array([[1, 2], [0, 0], [0, 2], [0, 2], [1, 0]]),
    )
    model = CovariateConductance(jnp.array([[0.2, -0.1, 0.4], [0.3, 0.1, -0.2]], dtype=dtype))
    pairs = (("e", "a"), ("c", "b"), ("c", "d"))
    weights = jnp.array([0.4, -1.3, 2.1], dtype=dtype)

    def objective(distance_model, *, upstream):
        if upstream:
            surface = distance_model.conductance_surface(region)
            graph = GridGraph(surface.astype(jnp.float64), fun=lambda a, b: (a + b) / 2)
            config = distance_model.solver_config
            solver = lx.CG(rtol=config.rtol, atol=config.atol, max_steps=config.max_steps)
            matrix = ResistanceDistance(solver=solver).nodes_to_nodes_distance(
                graph, jnp.array([5, 0, 2, 2, 3])
            )
            values = matrix[jnp.array([4, 2, 2]), jnp.array([0, 1, 3])].astype(dtype)
        else:
            values = distance_model.predict(region, pairs=pairs).values
        return weights @ values, values

    evaluate = eqx.filter_jit(eqx.filter_value_and_grad(objective, has_aux=True))
    (_, expected), expected_gradient = evaluate(model, upstream=True)
    (_, actual), actual_gradient = evaluate(model, upstream=False)
    tolerance = 1e-6
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(
        actual_gradient.weight, expected_gradient.weight, rtol=tolerance, atol=tolerance
    )
    assert np.linalg.norm(expected_gradient.weight) > 0
    assert actual.dtype == dtype
    assert actual[2] == 0


def test_conductance_surface_uses_prediction_mode_for_custom_dropout(
    DropoutConductance, observation_problem
):
    region, _, _ = observation_problem()
    model = DropoutConductance()
    np.testing.assert_allclose(model.conductance_surface(region), [[1.0, np.e, np.e**2]])
    np.testing.assert_allclose(
        model.predict(region, pairs=[("unit/7", "unit/99")]).values,
        [2 / (1 + np.e) + 2 / (np.e + np.e**2)],
    )


def test_custom_conductance_hooks_fit_and_predict_requested_pairs(PointConductance):
    region = RegionBatch(
        "river",
        cx.field(np.ones((1, 3, 1)), "row", "column", "feature"),
        ("a", "b", "c"),
        np.array([[0, 0], [0, 1], [0, 2]]),
    )
    distance_model = PointConductance(jnp.log(jnp.array(2.0)))
    prediction = distance_model.predict(region, pairs=[("c", "a"), ("b", "a")])
    np.testing.assert_allclose(prediction.values, [1.0, 0.5], atol=1e-6)
    observations = PairwiseObservations.from_pairs(
        [("a", "b"), ("a", "c"), ("b", "c")],
        [1.0, 2.0, 1.0],
        target=TargetSpec("dissimilarity"),
    )
    state = train(
        region,
        observations,
        distance_model=distance_model,
        config=TrainingConfig(loss=Log1pLoss(), steps=3, learning_rate=0.1),
    )
    assert state.history[-1].training_loss < state.history[0].training_loss
    np.testing.assert_allclose(
        state.selected_model.predict(region).values,
        state.selected_model.distance_model.predict(region).values,
    )
    assert np.all(
        cast(ConductanceModel, state.selected_model.distance_model).conductance_surface(region) > 0
    )


def test_custom_common_distance_contract_fits_and_reloads_without_genetic_labels(
    CustomDistance, tmp_path
):
    region = RegionBatch(
        "custom",
        cx.field(np.array([[[0.0], [1.0], [3.0]]]), "row", "column", "feature"),
        ("west", "middle", "east"),
        np.array([[0, 0], [0, 1], [0, 2]]),
    )
    observations = PairwiseObservations(
        region.sampling_unit_ids,
        [[0, 4, 36], [4, 0, 16], [36, 16, 0]],
        target=TargetSpec("divergence"),
    )
    state = train(
        region,
        observations,
        distance_model=CustomDistance(jnp.array(1.0)),
        config=TrainingConfig(loss=Log1pLoss(), steps=3, learning_rate=0.1),
    )
    assert state.history[-1].training_loss < state.history[0].training_loss
    path = tmp_path / "custom.ilg"
    save_model(path, state.selected_model)
    restored = load_model(
        path,
        distance_model=CustomDistance(jnp.array(0.0)),
    )
    np.testing.assert_array_equal(
        restored.distance_model.predict(region).as_matrix(),
        state.selected_model.distance_model.predict(region).as_matrix(),
    )
    np.testing.assert_array_equal(
        restored.predict(region).values, state.selected_model.predict(region).values
    )


def test_prediction_and_conductance_surfaces_require_float64(observation_problem, float64_required):
    region, _, model = observation_problem()
    with float64_required():
        model.predict(region)
    with float64_required():
        LogLinearConductance(2, patch_size=1).conductance_surface(region)
