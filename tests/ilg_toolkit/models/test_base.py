"""Tests of distance models: species queries, conductance surfaces and custom distance models."""

import warnings
from dataclasses import replace
from typing import cast, final

import coordax as cx
import equinox as eqx
import jax.numpy as jnp
import lineax as lx
import numpy as np
import optax
import pytest
import xarray as xr
from ilg_toolkit import (
    load_model,
    Log1pLoss,
    PairwiseObservations,
    RegionBatch,
    ResistanceSolverConfig,
    save_model,
    TargetSpec,
    train,
    TrainingConfig,
)
from ilg_toolkit.models import (
    AbstractDistanceModel,
    ConductanceModel,
    IBDModel,
    LogLinearConductance,
)
from jaxscape import GridGraph, ResistanceDistance


def test_distance_models_carry_unique_species_names(PointConductance):
    region = RegionBatch(
        "river",
        cx.field(
            np.ones((1, 3, 1)), "row", "column", cx.LabeledAxis("feature", np.array(["elevation"]))
        ),
        ("a", "b", "c"),
        np.array([[0, 0], [0, 1], [0, 2]]),
    )
    unnamed = LogLinearConductance(1, patch_size=1)
    assert unnamed.species is None
    prediction = unnamed.predict(region)
    assert prediction.species is None
    assert prediction.pairs == (("a", "b"), ("a", "c"), ("b", "c"))
    named = PointConductance(jnp.zeros(2), species=("stoat", "hare"))
    assert named.species == ("stoat", "hare")
    for species in ((), ("hare", "hare"), "hare", ("hare", ""), ("hare", 1)):
        with pytest.raises(ValueError, match="[Ss]pecies"):
            LogLinearConductance(1, species=species)  # pyright: ignore[reportArgumentType]
        with pytest.raises(ValueError, match="[Ss]pecies"):
            PointConductance(jnp.zeros(2), species=species)


def test_species_queries_follow_the_model_species_with_reused_sampling_unit_ids():
    region = RegionBatch(
        "river",
        cx.field(
            np.ones((1, 3, 1)), "row", "column", cx.LabeledAxis("feature", np.array(["elevation"]))
        ),
        ("a", "b", "c", "a", "b"),
        np.array([[0, 0], [0, 2], [0, 1], [0, 0], [0, 1]]),
        species=("hare", "hare", "hare", "stoat", "stoat"),
    )
    model = LogLinearConductance(1, species=("stoat", "hare"), patch_size=1)
    model = eqx.tree_at(lambda value: value.intercepts, model, jnp.log(jnp.array([1.99, 0.99])))
    results = model.predict(region, {"hare": [("b", "a")], "stoat": None})
    assert tuple(results) == ("hare", "stoat")
    assert results["hare"].pairs == (("b", "a"),)
    assert results["stoat"].pairs == (("a", "b"),)
    np.testing.assert_allclose(results["hare"].values, [2.0], rtol=1e-6)
    np.testing.assert_allclose(results["stoat"].values, [0.5], rtol=1e-6)
    assert results["hare"].sampling_unit_ids == ("a", "b", "c")
    assert results["stoat"].sampling_unit_ids == ("a", "b")
    for name, result in results.items():
        assert result.species == name
    every_hare_pair = model.predict(region, {"hare": None})
    assert tuple(every_hare_pair) == ("hare",)
    assert every_hare_pair["hare"].pairs == (("a", "b"), ("a", "c"), ("b", "c"))
    np.testing.assert_allclose(every_hare_pair["hare"].values, [2.0, 1.0, 1.0], rtol=1e-6)
    np.testing.assert_allclose(model.conductance_surface(region, "hare").data, np.ones((1, 3)))
    np.testing.assert_allclose(
        model.conductance_surface(region, species="stoat").data, np.full((1, 3), 2), rtol=1e-6
    )
    unlabeled = replace(region, sampling_unit_ids=("a", "b", "c", "d", "e"), species=None)
    for query, message in (
        (None, "Multispecies models require a query mapping"),
        ([("a", "b")], "Multispecies models require a query mapping"),
        ({"missing": None}, "Unknown species"),
        ({"hare": [("a", "missing")]}, "absent from region 'river': missing"),
    ):
        with pytest.raises(ValueError, match=message):
            model.predict(region, query)
    for species in (None, "missing"):
        with pytest.raises(ValueError, match="species"):
            model.conductance_surface(region, species)
    with pytest.raises(ValueError, match="sampling-unit species labels"):
        model.predict(unlabeled, {"hare": None})


def test_query_pairs_are_checked_with_specific_messages(PointConductance):
    region = RegionBatch(
        "river",
        cx.field(
            np.ones((1, 3, 1)), "row", "column", cx.LabeledAxis("feature", np.array(["elevation"]))
        ),
        ("a", "b", "c"),
        np.array([[0, 0], [0, 1], [0, 2]]),
    )
    model = PointConductance(jnp.asarray(0.0))
    for query, message in (
        ("ab", "sequence of sampling-unit pairs"),
        ([], "at least one sampling-unit pair"),
        ([("a", "b"), ("b", "a")], "reversed duplicates"),
        ([("a", "a")], "two distinct"),
        ([("a", "z"), ("y", "b")], "absent from region 'river': z, y"),
    ):
        with pytest.raises(ValueError, match=message):
            model.predict(region, query)  # type: ignore[arg-type]
    pairs = np.array([["c", "a"], ["b", "a"]])
    prediction = model.predict(region, pairs)  # type: ignore[arg-type]
    assert prediction.pairs == (("c", "a"), ("b", "a"))
    labeled = replace(region, species=("frog",) * 3)
    with pytest.raises(ValueError, match="require a distance model with species"):
        model.predict(labeled)


def test_shared_vertex_warnings_group_sampling_units_by_species():
    region = RegionBatch(
        "river",
        cx.field(
            np.ones((2, 4, 1)), "row", "column", cx.LabeledAxis("feature", np.array(["elevation"]))
        ),
        ("a", "b", "c", "a", "d"),
        np.array([[0, 0], [0, 1], [1, 1], [0, 0], [1, 0]]),
        species=("frog", "frog", "frog", "newt", "newt"),
    )
    model = LogLinearConductance(1, species=("frog", "newt"), patch_size=1)
    # A frog and a newt sharing a cell are told apart by species.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.predict(region, {"frog": None, "newt": None})
    coarse = LogLinearConductance(1, species=("frog", "newt"), patch_size=2)
    with pytest.warns(UserWarning, match="share a vertex") as record:
        coarse.predict(region, {"frog": None, "newt": None})
    assert len(record) == 1
    assert str(record[0].message).endswith("region 'river': frog (a, b, c), newt (a, d)")


def test_regions_smaller_than_one_patch_are_named():
    region = RegionBatch(
        "pond",
        cx.field(
            np.ones((1, 3, 1)), "row", "column", cx.LabeledAxis("feature", np.array(["elevation"]))
        ),
        ("a", "b"),
        np.array([[0, 0], [0, 2]]),
    )
    with pytest.raises(
        ValueError, match=r"Region 'pond' \(1 x 3 cells\) is smaller than one 4 x 4"
    ):
        LogLinearConductance(1, patch_size=4).predict(region)


def test_single_species_queries_return_one_prediction(PointConductance):
    region = RegionBatch(
        "river",
        cx.field(
            np.ones((1, 3, 1)), "row", "column", cx.LabeledAxis("feature", np.array(["elevation"]))
        ),
        ("a", "b", "c"),
        np.array([[0, 0], [0, 1], [0, 2]]),
    )
    unnamed = PointConductance(jnp.asarray(0.0))
    every_pair = unnamed.predict(region)
    assert every_pair.pairs == (("a", "b"), ("a", "c"), ("b", "c"))
    np.testing.assert_allclose(every_pair.values, [1.0, 2.0, 1.0], atol=1e-6)
    selected = unnamed.predict(region, [("c", "a")])
    assert selected.pairs == (("c", "a"),)
    np.testing.assert_allclose(selected.values, [2.0], atol=1e-6)
    np.testing.assert_array_equal(
        unnamed.predict(region, query=[("c", "a")]).values, selected.values
    )
    with pytest.raises(ValueError, match="species"):
        unnamed.predict(region, {"frog": None})
    with pytest.raises(ValueError, match="species"):
        unnamed.conductance_surface(region, "frog")
    named = PointConductance(jnp.asarray(0.0), species=("frog",))
    assert named.predict(region, [("c", "a")]).species == "frog"
    by_species = named.predict(region, {"frog": [("c", "a")]})
    assert tuple(by_species) == ("frog",)
    np.testing.assert_array_equal(by_species["frog"].values, selected.values)
    np.testing.assert_array_equal(
        named.conductance_surface(region).data, named.conductance_surface(region, "frog").data
    )
    for query, message in (
        ({}, "at least one species"),
        ({"toad": None}, "Unknown species"),
        (3, "sequence of sampling-unit pairs"),
    ):
        with pytest.raises(ValueError, match=message):
            named.predict(region, query)  # pyright: ignore[reportArgumentType]


@pytest.mark.filterwarnings("ignore:Sampling units share a vertex")
def test_float32_patch_encoders_give_float64_resistances_matching_jaxscape(FeatureConductance):
    # Float32 features and weights exercise the float64 solve.
    # Sampling units c and d share a vertex, so their resistance is exactly zero.
    dtype = jnp.float32
    region = RegionBatch(
        "river",
        cx.field(
            jnp.array([[[0.3], [0.8], [0.2]], [[0.5], [0.1], [0.7]]], dtype=dtype),
            "row",
            "column",
            cx.LabeledAxis("feature", np.array(["elevation"])),
        ),
        ("a", "b", "c", "d", "e"),
        np.array([[1, 2], [0, 0], [0, 2], [0, 2], [1, 0]]),
    )
    model = FeatureConductance(jnp.array([[0.2, -0.1, 0.4], [0.3, 0.1, -0.2]], dtype=dtype))
    pairs = (("e", "a"), ("c", "b"), ("c", "d"))
    weights = jnp.array([0.4, -1.3, 2.1], dtype=dtype)

    def weighted_distances(distance_model, *, upstream):
        if upstream:
            surface = distance_model.conductance_surface(region).data
            graph = GridGraph(surface.astype(jnp.float64), fun=lambda a, b: (a + b) / 2)
            config = distance_model.solver_config
            solver = lx.CG(rtol=config.rtol, atol=config.atol, max_steps=config.max_steps)
            matrix = ResistanceDistance(solver=solver).nodes_to_nodes_distance(
                graph, jnp.array([5, 0, 2, 2, 3])
            )
            values = matrix[jnp.array([4, 2, 2]), jnp.array([0, 1, 3])]
        else:
            values = distance_model.predict(region, pairs).values
        return weights @ values, values

    evaluate = eqx.filter_jit(eqx.filter_value_and_grad(weighted_distances, has_aux=True))
    (_, expected), expected_gradient = evaluate(model, upstream=True)
    (_, actual), actual_gradient = evaluate(model, upstream=False)
    tolerance = 1e-6
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(
        actual_gradient.weight, expected_gradient.weight, rtol=tolerance, atol=tolerance
    )
    assert np.linalg.norm(expected_gradient.weight) > 0
    assert actual.dtype == jnp.float64
    assert actual_gradient.weight.dtype == dtype
    assert actual[2] == 0


def test_conductance_surface_uses_prediction_mode_for_custom_dropout(
    DropoutConductance, observation_problem
):
    region, _, _ = observation_problem()
    model = DropoutConductance()
    np.testing.assert_allclose(model.conductance_surface(region).data, [[1.0, np.e, np.e**2]])
    np.testing.assert_allclose(
        model.predict(region, [("unit/7", "unit/99")]).values,
        [2 / (1 + np.e) + 2 / (np.e + np.e**2)],
    )


@pytest.mark.parametrize("evaluate", ["predict", "conductance_surface"])
def test_prediction_and_conductance_surfaces_reject_stateful_layers(
    BatchNormConductance, observation_problem, evaluate
):
    region, _, _ = observation_problem()
    with pytest.raises(TypeError, match="stateful layers"):
        getattr(BatchNormConductance(), evaluate)(region)


def test_custom_conductance_hooks_train_and_predict_requested_pairs(PointConductance):
    region = RegionBatch(
        "river",
        cx.field(
            np.ones((1, 3, 1)), "row", "column", cx.LabeledAxis("feature", np.array(["elevation"]))
        ),
        ("a", "b", "c"),
        np.array([[0, 0], [0, 1], [0, 2]]),
    )
    distance_model = PointConductance(jnp.log(jnp.array(2.0)))
    prediction = distance_model.predict(region, [("c", "a"), ("b", "a")])
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
        config=TrainingConfig(loss=Log1pLoss(), steps=3, optimizer=optax.adam(0.1)),
    )
    assert state.history[-1].training_loss < state.history[0].training_loss
    np.testing.assert_allclose(
        state.selected_model.predict(region).values,
        state.selected_model.distance_model.predict(region).values,
    )
    assert np.all(
        np.asarray(
            cast(ConductanceModel, state.selected_model.distance_model)
            .conductance_surface(region)
            .data
        )
        > 0
    )


@final
class _ResistancePlusDistance(AbstractDistanceModel):
    """Effective resistance of a log-linear patch encoder plus IBD distance."""

    encoder: LogLinearConductance
    ibd: IBDModel
    species: tuple[str, ...] | None = eqx.field(static=True, default=None)
    patch_size: int = eqx.field(static=True, default=1)

    def backbone(self, features):
        return self.encoder.backbone(features)

    def ecological_distances(self, latent, pairs, species_index):
        return self.encoder.ecological_distances(
            latent, pairs, species_index
        ) + self.ibd.ecological_distances(latent, pairs, species_index)


def _assert_composed_patch_encoder_computes_its_own_resistances(solver):
    region = RegionBatch(
        "river",
        cx.field(
            np.arange(9.0).reshape(3, 3, 1),
            "row",
            "column",
            cx.LabeledAxis("feature", np.array(["elevation"])),
        ),
        ("a", "b", "c"),
        np.array([[0, 0], [2, 2], [0, 2]]),
    )
    encoder = LogLinearConductance(
        1, patch_size=1, solver_config=ResistanceSolverConfig(solver=solver)
    )
    encoder = eqx.tree_at(lambda model: model.coefficients, encoder, jnp.array([[0.3]]))
    ibd = IBDModel(patch_size=1)
    model = _ResistancePlusDistance(encoder=encoder, ibd=ibd)
    np.testing.assert_allclose(
        model.predict(region).values,
        encoder.predict(region).values + ibd.predict(region).values,
        rtol=1e-6,
    )
    observations = PairwiseObservations.from_pairs(
        [("a", "b"), ("a", "c"), ("b", "c")], [3.0, 1.0, 2.0], target=TargetSpec("divergence")
    )
    state = train(
        region,
        observations,
        distance_model=model,
        config=TrainingConfig(loss=Log1pLoss(), steps=2, optimizer=optax.adam(0.1)),
    )
    assert state.history[-1].training_loss < state.history[0].training_loss


def test_composed_patch_encoders_compute_their_own_resistances(solver_choice):
    _assert_composed_patch_encoder_computes_its_own_resistances(solver_choice)


def test_composed_patch_encoders_compute_their_own_resistances_with_cudss_on_a_gpu(cudss_gpu):
    _assert_composed_patch_encoder_computes_its_own_resistances("cudss")


def test_species_of_one_prediction_share_a_solver_state(
    cudss_double, training_problem, PointConductance
):
    species = ("frog", "newt", "toad")
    region, _, _ = training_problem("conductance", species=species)
    scales = jnp.array([0.0, 0.5, 1.0])
    direct, iterative = (
        PointConductance(scales, species=species, solver_config=config)
        for config in (
            ResistanceSolverConfig(solver="cudss"),
            ResistanceSolverConfig(rtol=1e-11, atol=1e-11),
        )
    )
    with cudss_double.one_analysis():
        predictions = direct.predict(region, dict.fromkeys(species))
    expected = iterative.predict(region, dict.fromkeys(species))
    for name in species:
        np.testing.assert_allclose(predictions[name].values, expected[name].values, rtol=1e-8)


def test_conductance_surfaces_need_no_solver_backend(training_problem, PointConductance):
    region, _, _ = training_problem("conductance")
    surface = PointConductance(
        jnp.asarray(0.0), solver_config=ResistanceSolverConfig(solver="cudss")
    ).conductance_surface(region)
    np.testing.assert_allclose(surface.data, region.feature_array[..., 0])


def test_custom_distance_model_trains_and_reloads_without_genetic_labels(CustomDistance, tmp_path):
    region = RegionBatch(
        "custom",
        cx.field(
            np.array([[[0.0], [1.0], [3.0]]]),
            "row",
            "column",
            cx.LabeledAxis("feature", np.array(["elevation"])),
        ),
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
        config=TrainingConfig(loss=Log1pLoss(), steps=3, optimizer=optax.adam(0.1)),
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


def test_conductance_surfaces_carry_patch_centers_from_xarray_features():
    # Descending y, as in most GeoTIFFs; each 2 x 2 patch is one vertex.
    raster = xr.DataArray(
        np.linspace(0, 1, 24).reshape(4, 6, 1),
        dims=("y", "x", "band"),
        coords={
            "y": [350.0, 250.0, 150.0, 50.0],
            "x": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0],
            "band": ["elevation"],
        },
    )
    features = cx.from_xarray(raster.rename(y="row", x="column", band="feature"))
    region = RegionBatch.from_points(
        "valley", features, ("a", "b"), np.array([[1010.0, 340.0], [1490.0, 60.0]])
    )
    model = eqx.tree_at(
        lambda value: value.coefficients, LogLinearConductance(1, patch_size=2), jnp.ones((1, 1))
    )
    surface = model.conductance_surface(region).to_xarray()
    assert surface.dims == ("row", "column")
    np.testing.assert_array_equal(surface["row"], [300.0, 100.0])
    np.testing.assert_array_equal(surface["column"], [1050.0, 1250.0, 1450.0])
    patch_means = raster.to_numpy()[..., 0].reshape(2, 2, 3, 2).mean(axis=(1, 3))
    np.testing.assert_allclose(surface.to_numpy(), 0.01 + np.exp(patch_means))

    # Synthetic features without coordinates give the same surface over plain axes.
    plain = RegionBatch(
        "valley",
        cx.field(raster.to_numpy(), "row", "column", features.axes["feature"]),
        ("a", "b"),
        region.grid_positions,
    )
    plain_surface = model.conductance_surface(plain).to_xarray()
    assert plain_surface.dims == ("row", "column") and not plain_surface.coords
    np.testing.assert_array_equal(plain_surface.to_numpy(), surface.to_numpy())


@pytest.mark.parametrize("evaluate", ["predict", "conductance_surface"])
def test_prediction_and_conductance_surfaces_check_the_patch_grid(patch_check_problem, evaluate):
    problem = patch_check_problem
    with problem.trim_required():
        getattr(problem.model, evaluate)(problem.misaligned)
    with problem.shared_vertices_warned():
        getattr(problem.model, evaluate)(problem.shared)
