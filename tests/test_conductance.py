"""Actual graph calculation checked against independent tiny-graph references."""

from dataclasses import replace
from typing import final

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from ilg_toolkit import (
    ArtifactError,
    PairwiseObservations,
    RegionBatch,
    ResistanceSolverConfig,
    SamplingUnits,
    TargetSpec,
    TrainingConfig,
    fit,
    load_checkpoint,
    load_model,
    save_checkpoint,
    save_model,
)
from ilg_toolkit.models import ConductanceModel, ResNet9Conductance
from ilg_toolkit.resistance import build_resistance_context, effective_resistance


def dense_resistance(surface, nodes):
    """NumPy full Laplacian oracle; shares no production graph/solve helpers."""
    rows, columns = surface.shape
    laplacian = np.zeros((surface.size, surface.size))
    for row in range(rows):
        for column in range(columns):
            first = row * columns + column
            for next_row, next_column in ((row + 1, column), (row, column + 1)):
                if next_row < rows and next_column < columns:
                    second = next_row * columns + next_column
                    weight = (surface[row, column] + surface[next_row, next_column]) / 2
                    incidence = np.zeros(surface.size)
                    incidence[first], incidence[second] = 1, -1
                    laplacian += weight * np.outer(incidence, incidence)
    inverse = np.linalg.pinv(laplacian, hermitian=True)
    diagonal = np.diag(inverse)
    matrix = diagonal[:, None] + diagonal[None, :] - 2 * inverse
    return matrix[nodes[:, None], nodes[None, :]]


@pytest.mark.parametrize("use_amg", [False, True])
def test_actual_resistance_and_gradients_match_independent_laplacian(use_amg):
    if use_amg:
        pytest.importorskip("amjax")
        pytest.importorskip("pyamg")
    with jax.enable_x64():
        surface = np.array([[1.0, 1.4], [0.8, 2.0]])
        nodes = np.array([3, 0, 1, 1])
        context = build_resistance_context(surface.shape, ResistanceSolverConfig(use_amg=use_amg))
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


def test_resnet_fit_exposes_surface_and_label_free_target_predictions():
    features = np.random.default_rng(4).normal(size=(8, 8, 2)).astype(np.float32)
    units = SamplingUnits(("south", "north", "east"), [[0, 0], [0, 7], [7, 7]])
    region = RegionBatch(
        "small-valley", cx.field(features, "row", "column", "feature"), {"fox": units}
    )
    observations = PairwiseObservations.from_matrix(
        units.ids,
        dense_resistance(np.full((2, 2), 2.0), np.array([0, 1, 3])),
        target=TargetSpec("synthetic dissimilarity", units="index"),
    )
    model = ResNet9Conductance(
        in_channels=2,
        species=("fox",),
        patch_size=4,
        patch_batch_size=3,
        solver_config=ResistanceSolverConfig(rtol=1e-8),
        key=jax.random.key(6),
    )
    result = fit(
        [region],
        {region.name: {"fox": observations}},
        distance_model=model,
        config=TrainingConfig(objective="direct_log1p", steps=4, learning_rate=0.001),
    )
    assert result.history[-1].training_loss < result.history[0].training_loss
    assert isinstance(result.model.distance_model, ResNet9Conductance)
    assert result.model.distance_model.patch_batch_size == 3
    surface = result.model.conductance_surface(region)["fox"]
    prediction = result.model.predict(region)["fox"]
    raw = result.model.predict_raw_distances(region)["fox"]
    matrix = result.model.ecological_distances(region)["fox"]
    assert isinstance(surface, jax.Array)
    assert isinstance(prediction.values, jax.Array)
    assert surface.shape == (2, 2)
    assert np.all(np.isfinite(surface) & (surface > 0))
    expected = dense_resistance(np.asarray(surface), np.array([0, 1, 3]))
    np.testing.assert_allclose(prediction.as_matrix(), expected, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(raw.as_matrix(), expected, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(matrix, prediction.as_matrix())
    assert prediction.target.units == "index"


@final
class PointConductance(ConductanceModel):
    """Tiny learnable encoder keeps failure tests independent of CNN compilation."""

    log_scale: jax.Array
    species: tuple[str, ...] = eqx.field(static=True, default=("fox",))
    patch_size: int = eqx.field(static=True, default=1)
    solver_config: ResistanceSolverConfig = eqx.field(
        static=True, kw_only=True, default=ResistanceSolverConfig()
    )

    def __init__(self, log_scale: jax.Array, *, solver_config=None):
        self.log_scale = log_scale
        self.solver_config = ResistanceSolverConfig() if solver_config is None else solver_config

    def _conductance(self, features, species_index, *, patch_batch_size=None):
        return jnp.exp(self.log_scale) * features[..., 0]


@pytest.mark.parametrize("jit", [False, True])
def test_sparse_resistance_fit_matches_dense_reference_with_repeated_and_grounded_terminals(jit):
    surface = np.array([[1.0, 1.4, 0.7], [0.8, 2.0, 1.1]])
    nodes = np.array([0, 0, 2, 5, 1])
    region = RegionBatch(
        "sparse-graph",
        cx.field(surface[..., None], "row", "column", "feature"),
        {
            "fox": SamplingUnits(
                ("a", "coincident", "c", "ground", "unrequested"),
                np.column_stack(np.unravel_index(nodes, surface.shape)),
            )
        },
    )
    pairs = (("ground", "a"), ("a", "coincident"), ("c", "a"))
    indices = (np.array([3, 0, 2]), np.array([0, 1, 0]))
    targets = dense_resistance(surface * 1.7, nodes)[indices]
    observations = PairwiseObservations.from_pairs(pairs, targets, target=TargetSpec("divergence"))
    solver = ResistanceSolverConfig(rtol=1e-10, atol=1e-12)
    result = fit(
        [region],
        {region.name: {"fox": observations}},
        distance_model=PointConductance(jnp.array(0.0), solver_config=solver),
        config=TrainingConfig(objective="direct_log1p", steps=1, learning_rate=0.01, jit=jit),
    )

    def objective(log_scale):
        distances = dense_resistance(np.exp(log_scale) * surface, nodes)[indices]
        return np.mean((np.log1p(distances) - np.log1p(targets)) ** 2)

    gradient = (objective(1e-5) - objective(-1e-5)) / 2e-5
    expected_scale = -0.01 * gradient / (abs(gradient) + 1e-8)
    assert isinstance(result.optimizer_state, tuple)
    adam = result.optimizer_state[0]
    assert isinstance(adam, optax.ScaleByAdamState)
    assert isinstance(adam.mu, PointConductance)
    assert float(adam.mu.log_scale) == pytest.approx(0.1 * gradient, abs=1e-10)
    assert isinstance(result.distance_model, PointConductance)
    assert float(result.distance_model.log_scale) == pytest.approx(expected_scale, abs=1e-10)
    expected = dense_resistance(np.exp(expected_scale) * surface, nodes)
    prediction = result.model.predict(region, pairs={"fox": pairs})["fox"]
    np.testing.assert_allclose(prediction.values, expected[indices], rtol=1e-8, atol=1e-10)
    assert prediction.pairs == pairs
    assert float(prediction.values[1]) == 0.0
    np.testing.assert_allclose(
        result.model.ecological_distances(region)["fox"], expected, atol=1e-10
    )


def test_failed_solver_reports_region_species_and_step_without_changing_solver():
    with jax.enable_x64():
        region = RegionBatch(
            "bad-solve",
            cx.field(np.linspace(0.1, 2, 12).reshape(3, 4, 1), "row", "column", "feature"),
            {"fox": SamplingUnits(("first", "last"), [[0, 0], [2, 3]])},
        )
        observations = PairwiseObservations.from_matrix(
            region.sampling_units["fox"].ids, [[0, 1], [1, 0]], target=TargetSpec("dissimilarity")
        )
        with pytest.raises(RuntimeError, match="bad-solve.*fox.*step 0.*max_steps=1"):
            fit(
                [region],
                {region.name: {"fox": observations}},
                distance_model=PointConductance(
                    jnp.array(0.0),
                    solver_config=ResistanceSolverConfig(rtol=1e-12, atol=1e-12, max_steps=1),
                ),
                config=TrainingConfig(
                    objective="direct_log1p",
                    steps=0,
                ),
            )


def test_resnet_species_graphs_and_joint_gradients_match_independent_reference():
    regions, observed, nodes, pairs, targets = [], {}, {}, {}, {}
    for seed, (name, shape, species) in enumerate(
        (("valley", (8, 8), ("fox", "lynx")), ("ridge", (8, 12), ("fox",)))
    ):
        features = np.random.default_rng(seed + 41).normal(size=(*shape, 2)).astype(np.float32)
        units = {}
        observed[name], nodes[name], pairs[name], targets[name] = {}, {}, {}, {}
        for index, species_name in enumerate(species):
            labels = tuple(f"{name}-{label}" for label in ("west", "north", "east"))
            positions = (
                [[0, 0], [0, shape[1] - 1], [shape[0] - 1, shape[1] - 1]]
                if index == 0
                else [[shape[0] - 1, 0], [1, 2], [1, shape[1] - 1]]
            )
            units[species_name] = SamplingUnits(labels, positions)
            patch_nodes = np.array(
                [row // 4 * (shape[1] // 4) + col // 4 for row, col in positions]
            )
            nodes[name][species_name] = patch_nodes
            requested = ((2, 0), (1, 2)) if index == 0 else ((0, 1), (2, 0), (1, 2))
            if name == "ridge":
                requested = ((2, 0),)
            pairs[name][species_name] = tuple(zip(*requested, strict=True))
            reference_surface = np.linspace(0.6 + index, 1.8 + index, shape[0] * shape[1] // 16)
            reference = dense_resistance(
                reference_surface.reshape(shape[0] // 4, shape[1] // 4), patch_nodes
            )
            targets[name][species_name] = reference[pairs[name][species_name]]
            observed[name][species_name] = PairwiseObservations.from_pairs(
                [(labels[a], labels[b]) for a, b in requested],
                targets[name][species_name],
                target=TargetSpec("divergence", units="index"),
            )
        regions.append(
            RegionBatch(
                name,
                cx.field(features, "row", "column", "feature"),
                units,
                feature_names=("elevation", "habitat"),
            )
        )
    encoder = ResNet9Conductance(
        2,
        species=("fox", "lynx"),
        patch_size=4,
        patch_batch_size=3,
        solver_config=ResistanceSolverConfig(rtol=1e-9),
        key=jax.random.key(9),
    )

    def independent_objective(model):
        losses = {name: [] for name in model.species}
        for region in regions:
            surfaces = model.conductance_surface(region, species=tuple(region.sampling_units))
            for name, surface in surfaces.items():
                distances = dense_resistance(np.asarray(surface), nodes[region.name][name])
                selected = distances[pairs[region.name][name]]
                expected_targets = targets[region.name][name]
                losses[name].append(np.mean((np.log1p(selected) - np.log1p(expected_targets)) ** 2))
        return np.mean([np.mean(values) for values in losses.values()])

    result = fit(
        regions,
        observed,
        distance_model=encoder,
        config=TrainingConfig(objective="direct_log1p", steps=1, learning_rate=0.0002),
    )
    np.testing.assert_allclose(
        result.history[0].training_loss, independent_objective(encoder), rtol=2e-6
    )
    assert isinstance(result.optimizer_state, tuple)
    adam = result.optimizer_state[0]
    assert isinstance(adam, optax.ScaleByAdamState) and isinstance(adam.mu, ResNet9Conductance)

    # Adam's first moment exposes the gradient of the actual public fit objective.
    # Compare independent finite differences separately for pooled parameters and
    # each species' projection, including the unequal regional coverage weights.
    def scale_backbone(scale):
        return eqx.tree_at(
            lambda model: model.backbone,
            encoder,
            jax.tree.map(lambda value: scale * value, encoder.backbone),
        )

    directions = [
        (
            scale_backbone,
            sum(
                float(jnp.vdot(gradient, value)) * 10
                for gradient, value in zip(
                    jax.tree.leaves(adam.mu.backbone),
                    jax.tree.leaves(encoder.backbone),
                    strict=True,
                )
            ),
        )
    ]
    for index in range(2):

        def scale_head(scale, index=index):
            return eqx.tree_at(
                lambda model: (model.projection_weight, model.projection_bias),
                encoder,
                (
                    encoder.projection_weight.at[index].multiply(scale),
                    encoder.projection_bias.at[index].multiply(scale),
                ),
            )

        actual = 10 * (
            float(jnp.vdot(adam.mu.projection_weight[index], encoder.projection_weight[index]))
            + float(adam.mu.projection_bias[index] * encoder.projection_bias[index])
        )
        directions.append((scale_head, actual))
    for scaled_model, actual in directions:
        delta = 1e-3
        expected = (
            independent_objective(scaled_model(1 + delta))
            - independent_objective(scaled_model(1 - delta))
        ) / (2 * delta)
        assert abs(actual) > 1e-6
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=8e-3)
    for region in regions:
        selected = tuple(region.sampling_units)
        surfaces = result.model.conductance_surface(region, species=selected)
        requested = {
            name: tuple(
                (region.sampling_units[name].ids[a], region.sampling_units[name].ids[b])
                for a, b in zip(*pairs[region.name][name], strict=True)
            )
            for name in selected
        }
        raw = result.model.predict_raw_distances(region, species=selected, pairs=requested)
        prediction = result.model.predict(region, species=selected, pairs=requested)
        matrices = result.model.ecological_distances(region, species=selected)
        for name in selected:
            expected = dense_resistance(np.asarray(surfaces[name]), nodes[region.name][name])
            np.testing.assert_allclose(
                raw[name].values, expected[pairs[region.name][name]], rtol=2e-6
            )
            np.testing.assert_allclose(prediction[name].values, raw[name].values, rtol=2e-6)
            np.testing.assert_allclose(matrices[name], expected, atol=2e-6, rtol=2e-6)
            assert raw[name].pairs == requested[name]
        if len(surfaces) == 2:
            assert not np.allclose(surfaces["fox"], surfaces["lynx"])


def test_configured_resnet_chunks_match_unchunked_surfaces_and_parameter_gradients():
    features = np.random.default_rng(8).normal(size=(8, 12, 2)).astype(np.float32)
    units = SamplingUnits(("a", "b", "c"), [[0, 0], [0, 11], [7, 11]])
    region = RegionBatch("patches", cx.field(features, "row", "column", "feature"), {"fox": units})
    observed = {
        region.name: {
            "fox": PairwiseObservations.from_matrix(
                units.ids,
                dense_resistance(np.full((2, 3), 2.0), np.array([0, 2, 5])),
                target=TargetSpec("divergence"),
            )
        }
    }
    config = TrainingConfig(objective="direct_log1p", steps=1, learning_rate=0.0001)
    reference = None
    # Six patches include ragged final chunks and a checkpointed whole-region
    # chunk. Compare real resistance-objective gradients and the one-step update.
    for batch_size in (None, 4, 2, 9):
        encoder = ResNet9Conductance(
            2, species=("fox",), patch_size=4, patch_batch_size=batch_size, key=jax.random.key(19)
        )
        state = fit([region], observed, distance_model=encoder, config=config)
        if reference is None:
            reference = state
            continue
        np.testing.assert_allclose(
            state.history[0].training_loss, reference.history[0].training_loss, rtol=1e-6
        )
        for actual, expected in zip(
            jax.tree.leaves((state.optimizer_state, state.distance_model)),
            jax.tree.leaves((reference.optimizer_state, reference.distance_model)),
            strict=True,
        ):
            if isinstance(actual, jax.Array):
                np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=2e-6)
        np.testing.assert_allclose(
            state.model.conductance_surface(region)["fox"],
            reference.model.conductance_surface(region)["fox"],
            rtol=2e-5,
            atol=2e-6,
        )


def test_selected_resnet_species_use_distinct_softplus_surfaces_and_ignore_other_heads():
    units = {
        "fox": SamplingUnits(("a", "b", "c"), [[0, 0], [0, 7], [7, 7]]),
        "lynx": SamplingUnits(("a", "b", "c"), [[7, 0], [0, 0], [0, 7]]),
    }
    region = RegionBatch(
        "selected", cx.field(np.ones((8, 8, 1), np.float32), "row", "column", "feature"), units
    )
    model = ResNet9Conductance(
        1, species=("fox", "lynx", "unrequested"), min_conductance=0.02, key=jax.random.key(31)
    )
    model = eqx.tree_at(
        lambda value: (value.projection_weight, value.projection_bias),
        model,
        (
            jnp.zeros_like(model.projection_weight).at[2].set(jnp.nan),
            jnp.array([-2.0, 1.0, jnp.nan], jnp.float32),
        ),
    )
    pairs = {"lynx": (("c", "a"),), "fox": (("c", "a"), ("b", "c"))}
    surfaces = model.conductance_surface(region, species=("lynx", "fox"))
    raw = model.predict_raw_distances(region, species=("lynx", "fox"), pairs=pairs)
    assert tuple(surfaces) == tuple(raw) == ("lynx", "fox")
    for name, logit, nodes, indices in (
        ("fox", -2.0, np.array([0, 1, 3]), (np.array([2, 1]), np.array([0, 2]))),
        ("lynx", 1.0, np.array([2, 0, 1]), (np.array([2]), np.array([0]))),
    ):
        surface = np.full((2, 2), np.logaddexp(0, logit) + 0.02)
        np.testing.assert_allclose(surfaces[name], surface, rtol=2e-7)
        np.testing.assert_allclose(
            raw[name].values, dense_resistance(surface, nodes)[indices], rtol=2e-6
        )
        assert raw[name].pairs == pairs[name]
    with pytest.raises(ValueError, match="explicit species"):
        model.conductance_surface(region)
    with pytest.raises(ValueError, match="no sampling units"):
        model.conductance_surface(region, species="unrequested")


@pytest.mark.parametrize("batch_size", [0, -1, True, False, 1.5, "2"])
def test_resnet_rejects_invalid_configured_patch_batches(batch_size):
    with pytest.raises(ValueError, match="patch_batch_size"):
        ResNet9Conductance(2, species=("fox",), patch_batch_size=batch_size, key=jax.random.key(0))


def test_solver_rejects_silent_float64_truncation_and_invalid_terminals():
    with jax.enable_x64(False), pytest.raises(RuntimeError, match="JAX_ENABLE_X64"):
        effective_resistance(jnp.ones((2, 2), dtype=jnp.float32), np.array([0, 3]))
    with jax.enable_x64():
        with pytest.raises(ValueError, match="integer"):
            effective_resistance(jnp.ones((2, 2)), np.array([0.0, 1.5]))
        with pytest.raises(RuntimeError, match="outside"):
            effective_resistance(jnp.ones((2, 2)), np.array([0, 4]))
        with pytest.raises(RuntimeError, match="strictly positive"):
            effective_resistance(jnp.array([[1.0, 0.0]]), np.array([0, 1]))


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
    with pytest.raises(ValueError, match="solver|tolerance"):
        ResistanceSolverConfig(**options)


@pytest.mark.parametrize("use_amg", [False, True])
def test_shared_conductance_fit_rebuilds_solver_on_native_reload_and_resume(tmp_path, use_amg):
    if use_amg:
        pytest.importorskip("amjax")
        pytest.importorskip("pyamg")
    solver = ResistanceSolverConfig(rtol=1e-9, atol=1e-11, max_steps=321, use_amg=use_amg)
    encoder = PointConductance(jnp.array(0.0), solver_config=solver)
    regions = []
    observations = {}
    for name, shape in (("square", (2, 2)), ("rectangle", (2, 3))):
        surface = np.linspace(0.8, 1.6, np.prod(shape)).reshape(shape)
        region = RegionBatch(
            name,
            cx.field(surface[..., None], "row", "column", "feature"),
            {
                "fox": SamplingUnits(
                    tuple(f"{name}-{label}" for label in ("west", "north", "east")),
                    [[0, 0], [0, shape[1] - 1], [shape[0] - 1, shape[1] - 1]],
                )
            },
            feature_names=("habitat",),
        )
        target = dense_resistance(2 * surface, np.array([0, shape[1] - 1, surface.size - 1]))
        regions.append(region)
        observations[name] = {
            "fox": PairwiseObservations.from_matrix(
                region.sampling_units["fox"].ids, target, target=TargetSpec("divergence")
            )
        }
    config = TrainingConfig(objective="direct_log1p", steps=2, learning_rate=0.02)
    uninterrupted = fit(regions, observations, distance_model=encoder, config=config)
    first = fit(regions, observations, distance_model=encoder, config=replace(config, steps=1))
    model_path, checkpoint_path = tmp_path / "model.ilg", tmp_path / "checkpoint.ilg"
    save_model(model_path, first.model)
    save_checkpoint(checkpoint_path, first)
    mismatched_template = PointConductance(jnp.array(9.0))
    with pytest.raises(ArtifactError, match="solver_config"):
        load_model(model_path, distance_model=mismatched_template)
    with pytest.raises(ArtifactError, match="solver_config"):
        load_checkpoint(checkpoint_path, distance_model=mismatched_template)
    assert mismatched_template.solver_config == ResistanceSolverConfig()
    template = PointConductance(jnp.array(9.0), solver_config=solver)
    restored_model = load_model(model_path, distance_model=template)
    restored_state = load_checkpoint(checkpoint_path, distance_model=template)
    assert isinstance(restored_model.distance_model, PointConductance)
    assert isinstance(restored_state.distance_model, PointConductance)
    assert restored_model.distance_model.solver_config == solver
    assert restored_state.distance_model.solver_config == solver
    assert template.solver_config == solver
    continued = fit(regions[::-1], observations, state=restored_state, config=config)
    assert continued.history[-1].training_loss < continued.history[0].training_loss
    for actual, expected in zip(
        jax.tree.leaves(continued), jax.tree.leaves(uninterrupted), strict=True
    ):
        if isinstance(actual, jax.Array):
            np.testing.assert_array_equal(actual, expected)
        else:
            assert actual == expected
    for region in regions:
        np.testing.assert_array_equal(
            restored_model.ecological_distances(region)["fox"],
            first.model.ecological_distances(region)["fox"],
        )
        np.testing.assert_allclose(
            continued.model.ecological_distances(region)["fox"],
            dense_resistance(
                np.asarray(continued.model.conductance_surface(region)["fox"]),
                np.array(
                    [0, region.feature_array.shape[1] - 1, region.feature_array[..., 0].size - 1]
                ),
            ),
            rtol=1e-8,
            atol=1e-10,
        )
