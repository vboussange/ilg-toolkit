"""Tests of the log-linear patch encoder and its uniform-conductance initialization."""

import itertools

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from ilg_toolkit import (
    Log1pLoss,
    MLPELoss,
    PairwiseObservations,
    RegionBatch,
    ResistanceSolverConfig,
    TargetSpec,
    train,
    TrainingConfig,
)
from ilg_toolkit.models import initialize_uniform_conductance, LogLinearConductance


@pytest.mark.parametrize("patch_microbatch_size", [None, 3])
def test_patch_mean_predictor_obeys_conductance_bounds(patch_microbatch_size):
    features = np.array(
        [
            [[-4, 0], [-2, 2], [0, 2], [2, 0]],
            [[-2, 0], [0, -2], [2, -2], [4, 0]],
        ],
        dtype=float,
    )
    region = RegionBatch(
        "valley",
        cx.field(
            features, "row", "column", cx.LabeledAxis("feature", np.array(["elevation", "canopy"]))
        ),
        ("a", "b"),
        np.array([[0, 0], [0, 3]]),
    )
    model = LogLinearConductance(
        2,
        patch_size=2,
        patch_microbatch_size=patch_microbatch_size,
        min_conductance=0.25,
        max_conductance=5.0,
    )
    np.testing.assert_allclose(np.asarray(model.conductance_surface(region).data), [[1.25, 1.25]])
    model = eqx.tree_at(
        lambda model: (model.coefficients, model.intercepts),
        model,
        (jnp.array([[0.5, 2.0]]), jnp.array([0.3])),
    )
    # Patch means of the features are (-2, 0) and (2, 0), giving predictors -0.7 and 1.3.
    np.testing.assert_allclose(
        np.asarray(model.conductance_surface(region).data),
        [[0.7465853037914095, 3.9192966676192444]],
    )
    extreme = eqx.tree_at(lambda model: model.coefficients, model, jnp.array([[1e4, 2.0]]))
    np.testing.assert_allclose(np.asarray(extreme.conductance_surface(region).data), [[0.25, 5.0]])


@pytest.mark.parametrize(
    "settings",
    [
        {"in_channels": 0},
        {"in_channels": True},
        {"species": ()},
        {"species": "frog"},
        {"species": ("frog", "frog")},
        {"patch_size": 0},
        {"patch_size": True},
        {"patch_microbatch_size": 0},
        {"min_conductance": 0.0},
        {"min_conductance": True},
        {"max_conductance": 0.001},
        {"max_conductance": float("inf")},
        {"max_conductance": float("nan")},
        {"solver_config": "default"},
    ],
)
def test_invalid_model_configuration_is_rejected(settings):
    with pytest.raises((ValueError, TypeError)):
        LogLinearConductance(**({"in_channels": 2} | settings))


def test_training_gradients_and_predictions_match_dense_reference(dense_resistance):
    """Log-linear updates over two regions of different sizes match a dense-Laplacian reference.

    The regions have different patch grids and pair counts. Each keeps its own
    resistance solver, and the two region losses carry equal weight.
    """
    valley = np.array([[[-0.5, 0.4], [0.3, -0.2]], [[-0.1, 0.7], [0.6, -0.9]]])
    ridge = np.array([[[0.1, -0.6], [0.8, 0.2], [-0.4, 0.5]]])
    regional_means = (valley, ridge)
    regional_ids = (("a", "b", "c", "d"), ("a", "b", "c"))
    regions = (
        RegionBatch(
            "valley",
            cx.field(
                valley.repeat(2, axis=0).repeat(2, axis=1),
                "row",
                "column",
                cx.LabeledAxis("feature", np.array(("elevation", "habitat"))),
            ),
            regional_ids[0],
            np.array([[0, 0], [0, 2], [2, 0], [2, 2]]),
        ),
        RegionBatch(
            "ridge",
            cx.field(
                ridge.repeat(2, axis=0).repeat(2, axis=1),
                "row",
                "column",
                cx.LabeledAxis("feature", np.array(("elevation", "habitat"))),
            ),
            regional_ids[1],
            np.array([[0, 0], [0, 2], [1, 5]]),
        ),
    )
    observations = tuple(
        PairwiseObservations.from_pairs(
            tuple(itertools.combinations(ids, 2)),
            targets,
            target=TargetSpec("dissimilarity"),
        )
        for ids, targets in zip(
            regional_ids, ([0.4, 0.9, 1.4, 0.6, 1.1, 0.7], [0.5, 1.2, 0.8]), strict=True
        )
    )
    model = LogLinearConductance(
        2, patch_size=2, patch_microbatch_size=3, solver_config=ResistanceSolverConfig(rtol=1e-10)
    )
    initial = np.array([0.2, -0.1, 0.3])

    def with_parameters(parameters):
        return eqx.tree_at(
            lambda value: (value.coefficients, value.intercepts),
            model,
            (parameters[None, :2], parameters[2:]),
        )

    def reference_distances(parameters, values):
        surface = 0.01 + np.exp(values @ parameters[:2] + parameters[2])
        count = surface.size
        return dense_resistance(surface, np.arange(count))[np.triu_indices(count, 1)]

    def reference_loss(parameters):
        return np.mean(
            [
                np.mean(
                    (
                        np.log1p(reference_distances(parameters, values))
                        - np.log1p(observed.observed_values)
                    )
                    ** 2
                )
                for values, observed in zip(regional_means, observations, strict=True)
            ]
        )

    def public_loss(parameters):
        distance_model = with_parameters(parameters)
        return jnp.mean(
            jnp.stack(
                [
                    jnp.mean(
                        (
                            jnp.log1p(distance_model.predict(region).values)
                            - jnp.log1p(observed.observed_values)
                        )
                        ** 2
                    )
                    for region, observed in zip(regions, observations, strict=True)
                ]
            )
        )

    def reference_gradient(parameters):
        delta = np.eye(3) * 1e-5
        return np.array(
            [
                (reference_loss(parameters + offset) - reference_loss(parameters - offset)) / 2e-5
                for offset in delta
            ]
        )

    np.testing.assert_allclose(
        jax.grad(public_loss)(jnp.asarray(initial)),
        reference_gradient(initial),
        atol=1e-9,
        rtol=1e-6,
    )
    expected = initial.copy()
    first_moment, second_moment = np.zeros(3), np.zeros(3)
    for step in (1, 2):
        gradient = reference_gradient(expected)
        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * gradient**2
        expected -= (
            0.02
            * (first_moment / (1 - 0.9**step))
            / (np.sqrt(second_moment / (1 - 0.999**step)) + 1e-8)
        )
    state = train(
        regions,
        observations,
        distance_model=with_parameters(jnp.asarray(initial)),
        config=TrainingConfig(loss=Log1pLoss(), steps=2, optimizer=optax.adam(0.02)),
    )
    np.testing.assert_allclose(state.history[0].training_loss, reference_loss(initial), atol=1e-10)
    np.testing.assert_allclose(
        state.history[-1].training_loss, reference_loss(expected), atol=1e-10
    )
    for region, values in zip(regions, regional_means, strict=True):
        np.testing.assert_allclose(
            state.selected_model.predict(region).values,
            reference_distances(expected, values),
            atol=1e-8,
            rtol=1e-8,
        )


def test_surface_rejects_incompatible_landscape_features(observation_problem):
    region, _, _ = observation_problem()
    model = LogLinearConductance(3, patch_size=1)
    with pytest.raises(ValueError):
        model.conductance_surface(region)


_UNIFORM_POSITIONS = np.array([[0, 0], [0, 7], [3, 4], [7, 0], [7, 6], [5, 2]])


def _uniform_problem(dense_resistance, conductance, *, name="valley", seed=3, noise=0.0):
    """Returns a region on a 4x4 conductance grid graph and resistances at a uniform conductance."""
    rng = np.random.default_rng(seed)
    region = RegionBatch(
        name,
        cx.field(
            rng.normal(size=(8, 8, 2)),
            "row",
            "column",
            cx.LabeledAxis("feature", np.array(("grassland", "water"))),
        ),
        tuple(f"population-{index}" for index in range(len(_UNIFORM_POSITIONS))),
        _UNIFORM_POSITIONS,
    )
    vertices = (_UNIFORM_POSITIONS[:, 0] // 2) * 4 + _UNIFORM_POSITIONS[:, 1] // 2
    resistance = dense_resistance(np.full((4, 4), conductance), vertices)
    left, right = np.triu_indices(len(vertices), 1)
    values = resistance[left, right] * np.exp(rng.normal(scale=noise, size=len(left)))
    observations = PairwiseObservations.from_pairs(
        [
            (region.sampling_unit_ids[i], region.sampling_unit_ids[j])
            for i, j in zip(left, right, strict=True)
        ],
        values,
        target=TargetSpec("synthetic divergence", units="index"),
    )
    return region, observations


def _training_loss(model, region, observations, loss):
    state = train(
        region,
        observations,
        distance_model=model,
        config=TrainingConfig(loss=loss, steps=0),
    )
    return state.history[0].training_loss


def test_uniform_initialization_recovers_the_generating_conductance(dense_resistance):
    region, observations = _uniform_problem(dense_resistance, 3.7)
    model = eqx.tree_at(
        lambda model: model.coefficients,
        LogLinearConductance(2, patch_size=2),
        jnp.array([[0.4, -0.2]]),
    )
    initialized = initialize_uniform_conductance(model, region, observations, Log1pLoss())
    np.testing.assert_array_equal(initialized.coefficients, np.zeros((1, 2)))
    np.testing.assert_allclose(
        np.asarray(initialized.conductance_surface(region).data), 3.7, rtol=1e-6
    )
    np.testing.assert_allclose(
        _training_loss(initialized, region, observations, Log1pLoss()), 0, atol=1e-12
    )


def test_uniform_initialization_minimizes_the_region_weighted_loss(dense_resistance):
    ridge, ridge_observations = _uniform_problem(
        dense_resistance, 2.0, name="ridge", seed=5, noise=0.4
    )
    valley, valley_observations = _uniform_problem(
        dense_resistance, 9.0, name="valley", seed=6, noise=0.4
    )
    regions = [ridge, valley]
    # The valley region keeps two pairs; both regions receive equal weight.
    observations = [
        ridge_observations,
        valley_observations.select_pairs(
            [("population-0", "population-1"), ("population-2", "population-4")]
        ),
    ]
    model = initialize_uniform_conductance(
        LogLinearConductance(2, patch_size=2), regions, observations, Log1pLoss()
    )
    conductance = float(np.asarray(model.conductance_surface(regions[0]).data)[0, 0])
    assert 2.0 < conductance < 9.0
    initialized_loss = _training_loss(model, regions, observations, Log1pLoss())
    for factor in (0.999, 1.001, 0.9, 1.1):
        neighbor = eqx.tree_at(
            lambda model: model.intercepts,
            model,
            jnp.log(jnp.array([conductance * factor - model.min_conductance])),
        )
        assert initialized_loss <= _training_loss(neighbor, regions, observations, Log1pLoss())


@pytest.mark.parametrize("generating", [1e5, 1e-5])
def test_uniform_initialization_stays_inside_the_conductance_bounds(dense_resistance, generating):
    region, observations = _uniform_problem(dense_resistance, generating)
    model = LogLinearConductance(2, patch_size=2, min_conductance=0.01, max_conductance=1000.0)
    initialized = initialize_uniform_conductance(model, region, observations, Log1pLoss())
    surface = np.asarray(initialized.conductance_surface(region).data)
    assert np.all(np.isfinite(initialized.intercepts))
    assert np.all((surface > 0.01) & (surface < 1000.0))
    np.testing.assert_allclose(surface, 1000.0 if generating > 1 else 0.01, rtol=1e-3)


def test_uniform_initialization_sets_one_intercept_per_species(dense_resistance):
    from dataclasses import replace

    region, frog = _uniform_problem(dense_resistance, 0.5)
    newt = replace(frog, values=frog.values / 6.0, species="newt")
    frog = replace(frog, species="frog")
    count = len(region.sampling_unit_ids)
    region = replace(
        region,
        sampling_unit_ids=region.sampling_unit_ids * 2,
        grid_positions=np.tile(region.grid_positions, (2, 1)),
        species=("frog",) * count + ("newt",) * count,
    )
    model = initialize_uniform_conductance(
        LogLinearConductance(2, species=("newt", "frog"), patch_size=2),
        region,
        (frog, newt),
        Log1pLoss(),
    )
    np.testing.assert_allclose(
        np.asarray(model.conductance_surface(region, "frog").data), 0.5, rtol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(model.conductance_surface(region, "newt").data), 3.0, rtol=1e-6
    )


def test_standardized_mlpe_does_not_move_the_uniform_conductance(dense_resistance):
    region, observations = _uniform_problem(dense_resistance, 3.7, noise=0.3)
    model = LogLinearConductance(2, patch_size=2)
    initialized = initialize_uniform_conductance(model, region, observations, MLPELoss())
    np.testing.assert_array_equal(initialized.intercepts, model.intercepts)


def test_uniform_initialization_requires_a_log_linear_patch_encoder(
    dense_resistance, PointConductance
):
    region, observations = _uniform_problem(dense_resistance, 1.0)
    with pytest.raises(TypeError):
        initialize_uniform_conductance(
            PointConductance(jnp.array(0.0)), region, observations, Log1pLoss()
        )


def test_uniform_conductance_initialization_requires_float64(observation_problem, float64_required):
    region, target, _ = observation_problem()
    observations = PairwiseObservations.from_pairs(
        [("unit/7", "unit/20"), ("unit/20", "unit/99")], [1.0, 2.0], target=target
    )
    with float64_required():
        initialize_uniform_conductance(
            LogLinearConductance(2, patch_size=1), region, observations, Log1pLoss()
        )


def test_uniform_initialization_checks_the_patch_grid(patch_check_problem):
    problem = patch_check_problem
    misaligned, shared = problem.misaligned, problem.shared
    with problem.trim_required():
        initialize_uniform_conductance(
            problem.model, misaligned, problem.observations(misaligned), Log1pLoss()
        )
    with problem.shared_vertices_warned():
        initialize_uniform_conductance(
            problem.model, shared, problem.observations(shared), Log1pLoss()
        )


def test_feature_effects_map_feature_names_to_per_species_coefficients():
    model = eqx.tree_at(
        lambda model: (model.coefficients, model.intercepts),
        LogLinearConductance(3, species=("frog", "newt")),
        (jnp.array([[-0.25, 1.5, 0.0], [2.0, -3.0, 0.125]]), jnp.array([0.4, -0.2])),
    )

    effects = model.feature_effects(("grassland", "permanent water", "other"))

    assert effects == {
        "grassland": (-0.25, 2.0),
        "permanent water": (1.5, -3.0),
        "other": (0.0, 0.125),
    }
    assert list(effects) == ["grassland", "permanent water", "other"]
    assert all(type(value) is float for values in effects.values() for value in values)
    single = eqx.tree_at(
        lambda model: model.coefficients, LogLinearConductance(2), jnp.array([[0.75, -1.25]])
    )
    assert single.feature_effects(["elevation", "canopy"]) == {
        "elevation": (0.75,),
        "canopy": (-1.25,),
    }


@pytest.mark.parametrize(
    "feature_names",
    [("elevation",), ("elevation", "canopy", "water"), ("canopy", "canopy"), ("", "canopy")],
)
def test_feature_effects_require_one_unique_name_per_feature(feature_names):
    with pytest.raises(ValueError):
        LogLinearConductance(2).feature_effects(feature_names)


def test_uniform_initialization_shares_one_solver_across_regions_of_one_shape(cudss_double):
    features = np.random.default_rng(5).uniform(size=(3, 4, 1))
    regions = [
        RegionBatch(
            name,
            cx.field(
                features,
                "row",
                "column",
                cx.LabeledAxis("feature", np.array(["elevation"])),
            ),
            ("a", "b", "c"),
            np.array([[0, 0], [2, 3], [0, 3]]),
        )
        for name in ("river", "ridge")
    ]
    observations = [
        PairwiseObservations.from_pairs(
            [("a", "b"), ("a", "c"), ("b", "c")], values, target=TargetSpec("divergence")
        )
        for values in ([3.0, 1.0, 2.0], [2.0, 1.0, 3.0])
    ]
    direct, iterative = (
        LogLinearConductance(1, patch_size=1, solver_config=config)
        for config in (
            ResistanceSolverConfig(solver="cudss"),
            ResistanceSolverConfig(rtol=1e-11, atol=1e-11),
        )
    )
    rebuilds, analyses = cudss_double.rebuild_count(), cudss_double.analysis_count()
    initialized = initialize_uniform_conductance(direct, regions, observations, Log1pLoss())
    assert cudss_double.rebuild_count() == rebuilds
    assert cudss_double.analysis_count() == analyses + 1
    np.testing.assert_allclose(
        initialized.intercepts,
        initialize_uniform_conductance(iterative, regions, observations, Log1pLoss()).intercepts,
        rtol=1e-6,
    )
