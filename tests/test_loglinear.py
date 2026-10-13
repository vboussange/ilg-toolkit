"""Generic log-linear conductance checked at native public scientific seams."""

from typing import Any, cast

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from test_conductance import dense_resistance

from ilg_toolkit import (
    PairwiseObservations,
    RegionBatch,
    ResistanceSolverConfig,
    SamplingUnits,
    TargetSpec,
    TrainingConfig,
    fit,
)
from ilg_toolkit.models import ConductanceModel, LogLinearConductance


def test_default_loglinear_is_native_deterministic_and_has_inspectable_species_rows():
    model = LogLinearConductance(3, species=("fox", "lynx"))
    region = RegionBatch(
        "valley",
        cx.field(jnp.arange(64 * 64 * 3).reshape(64, 64, 3), "row", "column", "feature"),
        {
            name: SamplingUnits(("a", "b", "c"), [[0, 0], [0, 63], [63, 63]])
            for name in ("fox", "lynx")
        },
        feature_names=("elevation", "wetland", "cropland"),
    )
    assert isinstance(model, ConductanceModel)
    assert model.species == ("fox", "lynx")
    assert model.coefficients.shape == (2, 3)
    np.testing.assert_array_equal(model.coefficients, 0)
    np.testing.assert_array_equal(model.intercepts, 0)
    assert model.patch_size == 32
    surfaces = model.conductance_surface(region, species=("fox", "lynx"))
    for name, surface in surfaces.items():
        np.testing.assert_allclose(surface, np.full((2, 2), 1.01), rtol=1e-12)
        raw = model.predict_raw_distances(region, species=name, pairs={name: (("a", "c"),)})
        # Two two-edge paths in parallel on the uniform 2x2 graph.
        np.testing.assert_allclose(raw[name].values, [1 / 1.01], rtol=1e-10)


@pytest.mark.parametrize(
    "options",
    [
        {"in_channels": 0},
        {"in_channels": -2},
        {"in_channels": True},
        {"in_channels": 1.5},
        {"patch_size": 0},
        {"patch_size": -1},
        {"patch_size": False},
        {"patch_size": 1.5},
        {"min_conductance": 0},
        {"min_conductance": -0.1},
        {"min_conductance": float("nan")},
        {"min_conductance": float("inf")},
        {"max_conductance": float("inf")},
        {"max_conductance": float("nan")},
        {"max_conductance": 0.01},
        {"min_conductance": 4.0, "max_conductance": 2.0},
        {"min_conductance": True},
        {"max_conductance": False},
    ],
)
def test_loglinear_rejects_invalid_dimensions_and_bounds(options):
    kwargs = {"in_channels": 3, "species": ("fox",), **options}
    with pytest.raises(ValueError, match="positive integer|finite.*bounds"):
        LogLinearConductance(**kwargs)


def test_loglinear_requires_explicit_solver_policy_and_native_feature_layout():
    with pytest.raises(TypeError, match="ResistanceSolverConfig"):
        LogLinearConductance(1, species=("fox",), solver_config=cast(Any, "auto"))
    model = LogLinearConductance(2, species=("fox",), patch_size=2)
    units = {"fox": SamplingUnits(("a", "b"), [[0, 0], [1, 1]])}
    wrong_channels = RegionBatch(
        "wrong", cx.field(np.ones((4, 4, 3)), "row", "column", "feature"), units
    )
    with pytest.raises(ValueError, match="2 features"):
        model.conductance_surface(wrong_channels)
    odd_shape = RegionBatch("odd", cx.field(np.ones((3, 4, 2)), "row", "column", "feature"), units)
    with pytest.raises(ValueError, match="not divisible"):
        model.conductance_surface(odd_shape)


def test_loglinear_averages_supplied_features_before_link_and_preserves_patch_permutations():
    rng = np.random.default_rng(57)
    continuous = rng.normal(size=(4, 6, 3))
    categories = np.arange(24).reshape(4, 6) % 3
    # Caller supplies two indicators with category 0 as the explicit reference.
    features = np.concatenate(
        (continuous, (categories == 1)[..., None], (categories == 2)[..., None]), axis=-1
    )
    weights = np.array([[0.6, -0.4, 0.3, 0.8, -0.2], [-0.2, 0.5, 0.1, -0.7, 0.6]])
    biases = np.array([-0.7, 0.4])
    model = LogLinearConductance(
        5, species=("fox", "lynx"), patch_size=2, min_conductance=0.2, max_conductance=2.4
    )
    model = eqx.tree_at(
        lambda value: (value.coefficients, value.intercepts),
        model,
        (jnp.asarray(weights), jnp.asarray(biases)),
    )
    units = {
        "fox": SamplingUnits(("a", "b", "c", "d"), [[0, 0], [0, 5], [3, 0], [3, 5]]),
        "lynx": SamplingUnits(("a", "b", "c"), [[1, 2], [3, 2], [1, 0]]),
    }
    field_names = ("elevation", "rain", "slope", "cropland", "wetland")
    region = RegionBatch(
        "reference-coded",
        cx.field(features, "row", "column", "feature"),
        units,
        feature_names=field_names,
    )
    shuffled = features.copy()
    means = np.empty((2, 3, 5))
    for row in range(2):
        for col in range(3):
            patch = features[2 * row : 2 * row + 2, 2 * col : 2 * col + 2]
            means[row, col] = np.mean(patch, axis=(0, 1))
            shuffled[2 * row : 2 * row + 2, 2 * col : 2 * col + 2] = patch[::-1, ::-1]
    permuted = RegionBatch(
        "permuted", cx.field(shuffled, "row", "column", "feature"), units, feature_names=field_names
    )
    requests = {"fox": (("d", "a"), ("b", "c")), "lynx": (("c", "b"),)}
    surfaces = model.conductance_surface(region, species=("fox", "lynx"))
    other_surfaces = model.conductance_surface(permuted, species=("fox", "lynx"))
    raw = model.predict_raw_distances(region, species=("fox", "lynx"), pairs=requests)
    matrices = model.ecological_distances(region, species=("fox", "lynx"))
    for index, (name, nodes, selected) in enumerate(
        (
            ("fox", np.array([0, 2, 3, 5]), (np.array([3, 1]), np.array([0, 2]))),
            ("lynx", np.array([1, 4, 0]), (np.array([2]), np.array([1]))),
        )
    ):
        eta = means @ weights[index] + biases[index]
        expected_surface = np.clip(np.exp(np.minimum(eta, np.log(2.2))) + 0.2, 0.2, 2.4)
        np.testing.assert_allclose(surfaces[name], expected_surface, rtol=1e-12)
        np.testing.assert_allclose(other_surfaces[name], expected_surface, rtol=1e-12)
        expected_matrix = dense_resistance(expected_surface, nodes)
        np.testing.assert_allclose(raw[name].values, expected_matrix[selected], rtol=1e-9)
        np.testing.assert_allclose(matrices[name], expected_matrix, atol=1e-10, rtol=1e-9)
        assert raw[name].pairs == requests[name]
        pixel_link = np.clip(
            np.exp(np.minimum(features @ weights[index] + biases[index], np.log(2.2))) + 0.2,
            0.2,
            2.4,
        )
        averaged_link = np.array(
            [
                [np.mean(pixel_link[2 * r : 2 * r + 2, 2 * c : 2 * c + 2]) for c in range(3)]
                for r in range(2)
            ]
        )
        assert not np.allclose(expected_surface, averaged_link)


def _two_species_problem():
    features = np.array(
        [
            [[0.0, 0.2], [0.2, -0.3], [0.8, 0.9]],
            [[-0.3, 0.6], [1.4, -0.2], [-0.7, 1.2]],
        ]
    )
    ids = tuple(f"unit-{index}" for index in range(6))
    units = SamplingUnits(ids, np.array(list(np.ndindex(2, 3))))
    region = RegionBatch(
        "graph",
        cx.field(features, "row", "column", "feature"),
        {"fox": units, "lynx": units},
        feature_names=("habitat", "rain"),
    )
    model = LogLinearConductance(
        2,
        species=("fox", "lynx"),
        patch_size=1,
        solver_config=ResistanceSolverConfig(rtol=1e-11, atol=1e-12),
    )
    left, right = np.triu_indices(6, 1)
    pairs = tuple((ids[a], ids[b]) for a, b in zip(left, right, strict=True))
    truth = np.array([[0.6, -0.4], [-0.2, 0.7]])
    targets = {}
    for index, name in enumerate(model.species):
        surface = np.exp(features @ truth[index] + (0.15 if index == 0 else -0.25)) + 0.01
        targets[name] = dense_resistance(surface, np.arange(6))[left, right]
    return region, model, pairs, (left, right), targets


@pytest.mark.parametrize("jit", [False, True])
def test_loglinear_direct_fit_gradients_match_dense_reference_and_species_are_independent(jit):
    region, model, pairs, indices, targets = _two_species_problem()
    observed = {
        region.name: {
            name: PairwiseObservations.from_pairs(pairs, values, target=TargetSpec("divergence"))
            for name, values in targets.items()
        }
    }
    config = TrainingConfig(objective="direct_log1p", steps=1, learning_rate=0.01, jit=jit)
    state = fit([region], observed, distance_model=model, config=config)

    def independent(parameters):
        losses = []
        for index, name in enumerate(model.species):
            eta = np.asarray(region.feature_array) @ parameters[index, :2] + parameters[index, 2]
            surface = np.clip(np.exp(np.minimum(eta, np.log(999.99))) + 0.01, 0.01, 1000.0)
            distances = dense_resistance(surface, np.arange(6))[indices]
            losses.append(np.mean((np.log1p(distances) - np.log1p(targets[name])) ** 2))
        return np.mean(losses)

    initial = np.zeros((2, 3))
    derivative = np.zeros_like(initial)
    delta = 1e-5
    for index in np.ndindex(initial.shape):
        offset = np.zeros_like(initial)
        offset[index] = delta
        derivative[index] = (independent(initial + offset) - independent(initial - offset)) / (
            2 * delta
        )
    np.testing.assert_allclose(state.history[0].training_loss, independent(initial), rtol=1e-10)
    assert isinstance(state.optimizer_state, tuple)
    adam = state.optimizer_state[0]
    assert isinstance(adam, optax.ScaleByAdamState) and isinstance(adam.mu, LogLinearConductance)
    moment = np.column_stack((adam.mu.coefficients, adam.mu.intercepts))
    np.testing.assert_allclose(moment, 0.1 * derivative, atol=1e-10, rtol=1e-7)
    trained = state.distance_model
    assert isinstance(trained, LogLinearConductance)
    expected = initial - config.learning_rate * derivative / (np.abs(derivative) + 1e-8)
    np.testing.assert_allclose(
        np.column_stack((trained.coefficients, trained.intercepts)), expected, atol=1e-10
    )
    assert state.model.feature_names == region.feature_names
    assert state.history[-1].training_loss < state.history[0].training_loss
    raw = state.model.predict_raw_distances(region, species=("fox", "lynx"))
    predicted = state.model.predict(region, species=("fox", "lynx"))
    for name in model.species:
        np.testing.assert_allclose(predicted[name].values, raw[name].values, rtol=1e-12)
    changed = {
        region.name: {
            **observed[region.name],
            "fox": PairwiseObservations.from_pairs(
                pairs,
                targets["fox"] * 1.8,
                target=TargetSpec("divergence"),
            ),
        }
    }
    other = fit([region], changed, distance_model=model, config=config)
    assert isinstance(other.distance_model, LogLinearConductance)
    np.testing.assert_array_equal(other.distance_model.coefficients[1], trained.coefficients[1])
    np.testing.assert_array_equal(other.distance_model.intercepts[1], trained.intercepts[1])
    np.testing.assert_array_equal(
        other.model.predict_raw_distances(region, species="lynx")["lynx"].values, raw["lynx"].values
    )
    assert not np.allclose(other.distance_model.coefficients[0], trained.coefficients[0])


def test_extreme_loglinear_predictors_keep_surfaces_and_fit_gradients_finite():
    features = np.array([[[-1e100, 1.0], [0.0, 1.0], [1e100, 1.0]]])
    region = RegionBatch(
        "extreme",
        cx.field(features, "row", "column", "feature"),
        {"fox": SamplingUnits(("a", "b", "c"), [[0, 0], [0, 1], [0, 2]])},
    )
    model = LogLinearConductance(
        2,
        species=("fox",),
        patch_size=1,
        solver_config=ResistanceSolverConfig(rtol=1e-11, atol=1e-12),
    )
    model = eqx.tree_at(lambda value: value.coefficients, model, jnp.array([[1e200, 0.0]]))
    surface = model.conductance_surface(region)["fox"]
    np.testing.assert_allclose(surface, [[0.01, 1.01, 1000.0]], rtol=1e-12)
    observed = PairwiseObservations.from_pairs(
        (("a", "b"), ("a", "c"), ("b", "c")),
        [1.0, 1.5, 0.5],
        target=TargetSpec("divergence"),
    )
    state = fit(
        [region],
        {region.name: {"fox": observed}},
        distance_model=model,
        config=TrainingConfig(objective="direct_log1p", steps=1),
    )
    assert isinstance(state.optimizer_state, tuple)
    adam = state.optimizer_state[0]
    assert isinstance(adam, optax.ScaleByAdamState) and isinstance(adam.mu, LogLinearConductance)
    for leaf in jax.tree.leaves((adam.mu, adam.nu, state.distance_model)):
        np.testing.assert_array_equal(np.isfinite(leaf), True)
    assert float(adam.mu.coefficients[0, 0]) == 0.0
    assert np.isfinite(state.model.predict(region)["fox"].values).all()


def test_loglinear_mlpe_gradients_and_group_predictions_match_independent_gaussian_reference():
    from test_mlpe_training import dense_profile

    from ilg_toolkit import UnpooledMLPECalibrationModule

    region, model, pairs, indices, truth = _two_species_problem()
    second_features = np.rot90(np.asarray(region.feature_array)).copy()
    second_ids = tuple(f"ridge-{index}" for index in range(6))
    second = RegionBatch(
        "ridge",
        cx.field(second_features, "row", "column", "feature"),
        {"fox": SamplingUnits(second_ids, np.array(list(np.ndindex(3, 2))))},
        feature_names=region.feature_names,
    )
    second_pairs = tuple((second_ids[a], second_ids[b]) for a, b in zip(*indices, strict=True))
    second_truth = dense_resistance(
        np.exp(second_features @ np.array([0.6, -0.4]) + 0.15) + 0.01, np.arange(6)
    )[indices]
    initial = np.array([[0.2, -0.1, 0.1], [-0.15, 0.25, -0.2]])
    model = eqx.tree_at(
        lambda value: (value.coefficients, value.intercepts),
        model,
        (jnp.asarray(initial[:, :2]), jnp.asarray(initial[:, 2])),
    )
    observed, groups = {region.name: {}, second.name: {}}, []
    for number, (prepared, name, distances, requested) in enumerate(
        (
            (region, "fox", truth["fox"], pairs),
            (region, "lynx", truth["lynx"], pairs),
            (second, "fox", second_truth, second_pairs),
        )
    ):
        z = (distances - distances.mean()) / distances.std(ddof=1)
        targets = 2 + number + (0.6 if name == "fox" else -0.5) * z
        targets += np.random.default_rng(number + 75).normal(0, 0.01, size=len(targets))
        observed[prepared.name][name] = PairwiseObservations.from_pairs(
            requested,
            targets,
            target=TargetSpec("divergence", units="index"),
        )
        groups.append((prepared, name, targets))
    variances = (0.05, 0.1)

    def independent(parameters):
        losses = {name: [] for name in model.species}
        for prepared, name, targets in groups:
            index = model.species.index(name)
            eta = np.asarray(prepared.feature_array) @ parameters[index, :2] + parameters[index, 2]
            surface = np.clip(np.exp(np.minimum(eta, np.log(999.99))) + 0.01, 0.01, 1000.0)
            distances = dense_resistance(surface, np.arange(6))[indices]
            nll = dense_profile(distances, targets, *indices, 6, variances)[0]
            losses[name].append(nll / len(distances))
        return np.mean([np.mean(values) for values in losses.values()])

    derivative = np.zeros_like(initial)
    delta = 1e-5
    for index in np.ndindex(initial.shape):
        offset = np.zeros_like(initial)
        offset[index] = delta
        derivative[index] = (independent(initial + offset) - independent(initial - offset)) / (
            2 * delta
        )
    state = fit(
        [second, region],
        observed,
        distance_model=model,
        config=TrainingConfig(
            objective="mlpe", steps=1, learning_rate=0.01, mlpe_initial_variances=variances
        ),
    )
    np.testing.assert_allclose(state.history[0].training_loss, independent(initial), atol=1e-10)
    assert isinstance(state.optimizer_state, tuple)
    adam = state.optimizer_state[0]
    assert isinstance(adam, optax.ScaleByAdamState) and isinstance(adam.mu, tuple)
    model_moment = adam.mu[0]
    assert isinstance(model_moment, LogLinearConductance)
    np.testing.assert_allclose(
        np.column_stack((model_moment.coefficients, model_moment.intercepts)),
        0.1 * derivative,
        atol=1e-9,
        rtol=1e-6,
    )
    assert state.history[-1].training_loss < state.history[0].training_loss
    assert isinstance(state.model.calibration, UnpooledMLPECalibrationModule)
    for prepared, name, targets in groups:
        matrix = state.model.ecological_distances(prepared, species=name)[name]
        surface = state.model.conductance_surface(prepared, species=name)[name]
        expected_distances = dense_resistance(np.asarray(surface), np.arange(6))[indices]
        np.testing.assert_allclose(np.asarray(matrix)[indices], expected_distances, rtol=1e-9)
        head = state.model.calibration.groups[prepared.name][name]
        expected_nll, beta, mean, posterior = dense_profile(
            expected_distances,
            targets,
            *indices,
            6,
            (head.unit_variance, head.residual_variance),
        )
        np.testing.assert_allclose([head.intercept, head.slope], beta, rtol=1e-9)
        np.testing.assert_allclose(-head.ml_log_likelihood, expected_nll, atol=1e-9)
        np.testing.assert_allclose(head.effect_mean, mean, atol=1e-9)
        np.testing.assert_allclose(head.effect_covariance, posterior, atol=1e-9)
        predicted = state.model.predict(prepared, species=name)[name]
        expected = beta[0] + beta[1] * (
            (expected_distances - expected_distances.mean()) / expected_distances.std(ddof=1)
        )
        np.testing.assert_allclose(predicted.values, expected, rtol=1e-9)
        assert predicted.region_name == prepared.name and predicted.species_name == name


@pytest.mark.parametrize(("floor", "cap", "expected"), [(2.0, 2.5, 2.5), (0.25, 0.4, 0.4)])
def test_zero_initialization_respects_a_narrow_configured_conductance_interval(
    floor, cap, expected
):
    model = LogLinearConductance(
        1, species=("fox",), patch_size=1, min_conductance=floor, max_conductance=cap
    )
    region = RegionBatch(
        "narrow",
        cx.field(np.zeros((1, 2, 1)), "row", "column", "feature"),
        {"fox": SamplingUnits(("a", "b"), [[0, 0], [0, 1]])},
    )
    np.testing.assert_allclose(model.conductance_surface(region)["fox"], expected, rtol=1e-12)
