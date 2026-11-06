"""Tests of the log-linear patch encoder and its uniform-conductance initialization."""

import itertools
import math
from typing import cast

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
    RidgePenalty,
    TargetSpec,
    train,
    TrainingConfig,
    TrainingState,
)
from ilg_toolkit.models import initialize_uniform_conductance, LogLinearConductance
from jax.flatten_util import ravel_pytree


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


@pytest.mark.parametrize("cap", ["hard", "smooth"])
@pytest.mark.parametrize(
    ("species", "coefficients", "species_index", "expected"),
    [
        (None, [[1.0]], 0, [True, False, False, False, True]),
        (("frog", "newt"), [[1.0], [-2.0]], 0, [True, False, False, False, True]),
        (("frog", "newt"), [[1.0], [-2.0]], jnp.asarray(1), [True, True, False, True, True]),
    ],
)
def test_saturation_flags_feature_vectors_below_the_floor_and_at_the_cap(
    species, coefficients, species_index, expected, cap
):
    model = eqx.tree_at(
        lambda model: model.coefficients,
        LogLinearConductance(
            1, species=species, min_conductance=0.25, max_conductance=5.0, cap=cap
        ),
        jnp.array(coefficients),
    )
    # The floor holds below the predictor log 0.25 = -1.39, the cap from log 4.75 = 1.56.
    # The frog's predictors are (-3, -1, 0, 1, 3), the newt's (6, 2, 0, -2, -6).
    features = jnp.array([[-3.0], [-1.0], [0.0], [1.0], [3.0]])

    np.testing.assert_array_equal(model.is_saturated(features, species_index), expected)


def _predictors_around_the_ceiling(model):
    """Returns predictors from 0 to where exp overflows, dense within 64 ulps of log(max - min)."""
    ceiling = float(jnp.log(model.max_conductance - model.min_conductance))
    return jnp.asarray(
        np.concatenate(
            [ceiling + np.spacing(ceiling) * np.arange(-64, 65), [0.0, ceiling + 1.0, 1e3]]
        )
    )


@pytest.mark.parametrize("settings", [{}, {"cap": "hard"}], ids=["default", "hard"])
def test_conductance_stops_the_exponential_at_the_ceiling_then_clips(settings):
    model = eqx.tree_at(
        lambda model: model.coefficients, LogLinearConductance(1, **settings), jnp.array([[1.0]])
    )
    features = jnp.concatenate(
        [jnp.linspace(-10.0, 10.0, 41), _predictors_around_the_ceiling(model)]
    )[:, None]
    ceiling = jnp.log(model.max_conductance - model.min_conductance)

    def conductance(coefficients):
        return eqx.tree_at(lambda model: model.coefficients, model, coefficients).conductance(
            features, 0
        )

    def capped_exponential(coefficients):
        predictor = features @ coefficients[0] + model.intercepts[0]
        return jnp.clip(
            jnp.exp(jnp.minimum(predictor, ceiling)) + model.min_conductance,
            model.min_conductance,
            model.max_conductance,
        )

    np.testing.assert_array_equal(
        conductance(model.coefficients), capped_exponential(model.coefficients)
    )
    np.testing.assert_array_equal(
        jax.jacobian(conductance)(model.coefficients),
        jax.jacobian(capped_exponential)(model.coefficients),
    )


def test_saturation_at_the_cap_is_where_the_conductance_is_capped():
    model = eqx.tree_at(
        lambda model: model.coefficients, LogLinearConductance(1), jnp.array([[1.0]])
    )
    features = _predictors_around_the_ceiling(model)[:, None]
    # With the default bounds, the capped conductance rounds below max_conductance.
    capped = model.conductance(jnp.array([[1e3]]), 0)[0]

    at_cap = np.asarray(model.conductance(features, 0)) == capped

    assert at_cap.any() and not at_cap.all()
    np.testing.assert_array_equal(model.is_saturated(features, 0), at_cap)


@pytest.mark.parametrize("leading_shape", [(), (6,), (2, 3), (2, 3, 4)])
def test_saturation_keeps_the_leading_dimensions(leading_shape):
    model = eqx.tree_at(
        lambda model: model.coefficients, LogLinearConductance(2), jnp.array([[8.0, -6.0]])
    )
    features = jnp.asarray(np.random.default_rng(3).normal(size=(*leading_shape, 2)))

    saturated = model.is_saturated(features, 0)

    assert saturated.shape == leading_shape
    assert saturated.dtype == jnp.bool_
    np.testing.assert_array_equal(
        saturated.reshape(-1), model.is_saturated(features.reshape(-1, 2), 0)
    )


# The scale of the smooth cap, in log conductance, that `LogLinearConductance` documents.
_SMOOTH_CAP_SCALE = 0.1


def _predictor_model(cap):
    """Returns a one-feature model with the default bounds whose linear predictor is the feature."""
    return eqx.tree_at(
        lambda model: model.coefficients, LogLinearConductance(1, cap=cap), jnp.array([[1.0]])
    )


def _ceiling(model):
    """Returns the linear predictor at which min_conductance + exp(predictor) reaches the cap."""
    return math.log(model.max_conductance - model.min_conductance)


def _scalar_conductance(model):
    """Returns the conductance of a one-feature model as a function of one linear predictor."""
    return lambda predictor: model.conductance(jnp.reshape(predictor, (1, 1)), 0)[0]


def test_the_hard_cap_is_the_default():
    assert LogLinearConductance(1).cap == "hard"
    assert eqx.tree_equal(LogLinearConductance(1), LogLinearConductance(1, cap="hard"))
    assert not eqx.tree_equal(LogLinearConductance(1), LogLinearConductance(1, cap="smooth"))


def test_the_smooth_cap_matches_the_hard_cap_below_the_ceiling_within_the_documented_bound():
    hard, smooth = (_predictor_model(cap) for cap in ("hard", "smooth"))
    # Depths of the linear predictor below the ceiling, in log units, down to below the floor.
    depths = np.concatenate([np.linspace(1.0, 4.0, 31), np.linspace(5.0, 20.0, 16)])
    features = jnp.asarray(_ceiling(hard) - depths)[:, None]

    deviation = np.abs(
        np.asarray(smooth.conductance(features, 0)) / np.asarray(hard.conductance(features, 0)) - 1
    )

    bound = _SMOOTH_CAP_SCALE * np.exp(-depths / _SMOOTH_CAP_SCALE)
    np.testing.assert_array_less(deviation, bound + 1e-14)
    # One log unit below the ceiling, the bound is tight: the cap there has this scale.
    assert 0.9 * bound[0] < deviation[0] < 5e-6


def test_the_smooth_cap_rises_monotonically_toward_max_conductance():
    model = _predictor_model("smooth")
    ceiling = _ceiling(model)
    conductance = _scalar_conductance(model)
    predictors = jnp.linspace(ceiling - 3.0, ceiling + 6.0, 901)

    values = np.asarray(jax.vmap(conductance)(predictors))
    slopes = np.asarray(jax.vmap(jax.grad(conductance))(predictors))

    assert np.all(np.diff(values) >= 0)
    assert np.all(np.diff(values)[predictors[1:] <= ceiling + 1.0] > 0)
    assert np.all(values <= model.max_conductance)
    assert np.all(slopes[predictors <= ceiling + 3.0] > 0)
    # The cap lowers the slope of the exponential, and the conductance reaches its cap only
    # in rounding.
    np.testing.assert_array_less(slopes, np.exp(np.asarray(predictors)))
    np.testing.assert_allclose(values[-1], model.max_conductance, rtol=1e-15)
    # At the ceiling, the conductance in excess of min_conductance is 2**-scale of its cap.
    np.testing.assert_allclose(
        conductance(ceiling) - model.min_conductance,
        (model.max_conductance - model.min_conductance) * 2**-_SMOOTH_CAP_SCALE,
        rtol=1e-14,
    )


@pytest.mark.parametrize("cap", ["hard", "smooth"])
def test_only_the_smooth_cap_is_continuous_in_gradient_across_the_ceiling(cap):
    model = _predictor_model(cap)
    ceiling = _ceiling(model)
    conductance = _scalar_conductance(model)
    slope = jax.grad(conductance)
    excess = model.max_conductance - model.min_conductance
    for gap in (1e-3, 1e-5, 1e-7):
        below, above = jnp.asarray(ceiling - gap), jnp.asarray(ceiling + gap)
        assert abs(conductance(above) - conductance(below)) <= 2 * gap * excess
        jump = abs(float(slope(above) - slope(below)))
        if cap == "hard":
            # The slope drops from the exponential's to 0.
            np.testing.assert_allclose(jump, excess, rtol=1e-3)
        else:
            # The second derivative of the smooth cap is at most excess * (1 + 1 / (4 * scale)).
            assert jump <= 2 * gap * excess * (1 + 1 / (4 * _SMOOTH_CAP_SCALE))


def _finite_difference_hessians(model, limit, steps):
    """Returns central-difference Hessians of a weighted conductance sum near a conductance limit.

    The parameters are the two coefficients and the intercept. The first feature vector puts
    its linear predictor 1e-8 below the ceiling, or below log(min_conductance); the others put
    theirs within 2 log units of it.
    """
    location = _ceiling(model) if limit == "ceiling" else math.log(model.min_conductance)
    coefficients, intercept = np.array([0.5, -0.25]), location - 0.2
    features = np.array([[0.0, 0.3], [1.2, -0.4], [-2.0, 1.5], [1.5, 2.0]])
    features[0, 0] = (location - 1e-8 - intercept - coefficients[1] * features[0, 1]) / 0.5
    weights = jnp.array([1.0, -0.7, 0.4, 0.2])

    def objective(parameters):
        configured = eqx.tree_at(
            lambda model: (model.coefficients, model.intercepts),
            model,
            (parameters[None, :2], parameters[2:]),
        )
        return weights @ configured.conductance(jnp.asarray(features), 0)

    gradient = jax.jit(jax.grad(objective))
    point = jnp.array([*coefficients, intercept])
    return [
        np.stack(
            [
                (gradient(point + step * unit) - gradient(point - step * unit)) / (2 * step)
                for unit in jnp.eye(point.size)
            ],
            axis=1,
        )
        for step in steps
    ]


@pytest.mark.parametrize(
    ("cap", "limit"), [("hard", "floor"), ("smooth", "floor"), ("smooth", "ceiling")]
)
def test_the_finite_difference_hessian_at_a_smooth_limit_is_symmetric_and_stable_in_the_step(
    cap, limit
):
    hessians = _finite_difference_hessians(_predictor_model(cap), limit, (1e-3, 1e-4, 1e-5))

    scale = np.abs(hessians[-1]).max()
    assert scale > 0
    for hessian in hessians:
        assert np.abs(hessian - hessian.T).max() <= 1e-3 * scale
        np.testing.assert_allclose(hessian, hessians[-1], rtol=0, atol=1e-3 * scale)


def test_the_finite_difference_hessian_at_the_hard_cap_grows_as_the_step_shrinks():
    coarse, fine = _finite_difference_hessians(_predictor_model("hard"), "ceiling", (1e-3, 1e-5))

    assert np.abs(fine).max() > 50 * np.abs(coarse).max()
    assert np.abs(fine - fine.T).max() > 0.1 * np.abs(fine).max()


def test_under_the_smooth_cap_saturation_flags_where_the_predictor_acts_at_most_half_as_much():
    model = _predictor_model("smooth")
    conductance = _scalar_conductance(model)
    # Grids around log(min_conductance) and the ceiling that skip both points.
    offsets = np.linspace(-2.0, 2.0, 400)
    floor_predictors = jnp.asarray(math.log(model.min_conductance) + offsets)
    cap_predictors = jnp.asarray(_ceiling(model) + offsets)

    # Below the floor, the predictor moves log conductance by less than half.
    log_slope = jax.vmap(jax.grad(lambda predictor: jnp.log(conductance(predictor))))
    np.testing.assert_array_equal(
        model.is_saturated(floor_predictors[:, None], 0),
        np.asarray(log_slope(floor_predictors)) < 0.5,
    )
    # At the cap, it moves log conductance in excess of min_conductance by at most half.
    excess_slope = jax.vmap(jax.grad(conductance))(cap_predictors) / (
        jax.vmap(conductance)(cap_predictors) - model.min_conductance
    )
    np.testing.assert_array_equal(
        model.is_saturated(cap_predictors[:, None], 0), np.asarray(excess_slope) <= 0.5
    )


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
        {"cap": "soft"},
        {"cap": None},
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


@pytest.mark.parametrize(
    ("cap", "max_conductance"), [("hard", 1000.0), ("smooth", 1000.0), ("smooth", 4.0)]
)
def test_uniform_initialization_recovers_the_generating_conductance(
    dense_resistance, cap, max_conductance
):
    # Under a cap of 4, the smooth cap lowers the conductance at intercept log(3.69) by about 4%.
    region, observations = _uniform_problem(dense_resistance, 3.7)
    model = eqx.tree_at(
        lambda model: model.coefficients,
        LogLinearConductance(2, patch_size=2, max_conductance=max_conductance, cap=cap),
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


@pytest.mark.parametrize("cap", ["hard", "smooth"])
@pytest.mark.parametrize("generating", [1e5, 1e-5])
def test_uniform_initialization_stays_inside_the_conductance_bounds(
    dense_resistance, generating, cap
):
    region, observations = _uniform_problem(dense_resistance, generating)
    model = LogLinearConductance(
        2, patch_size=2, min_conductance=0.01, max_conductance=1000.0, cap=cap
    )
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
    with cudss_double.one_analysis():
        initialized = initialize_uniform_conductance(direct, regions, observations, Log1pLoss())
    np.testing.assert_allclose(
        initialized.intercepts,
        initialize_uniform_conductance(iterative, regions, observations, Log1pLoss()).intercepts,
        rtol=1e-6,
    )


def _corridor_problem(dense_resistance) -> tuple[RegionBatch, PairwiseObservations]:
    """Returns sampling units around a corridor that conducts beyond a cap at 5.

    The genetic observations are effective resistances on the surface
    ``0.01 + exp(5 * corridor + 0.3 * noise)``, whose corridor conducts about 150 times more
    than its background. Under a cap at 5, the corridor cannot follow, and a ridge penalty
    on the coefficients pulls its coefficient down: under the hard cap, the optimum of the
    training objective puts a corridor vertex at the ceiling.
    """
    rng = np.random.default_rng(0)
    corridor = np.zeros((6, 8))
    corridor[2:4] = 1.0
    features = np.stack([corridor, rng.normal(size=corridor.shape)], axis=-1)
    positions = np.array([[2, 0], [3, 7], [0, 1], [5, 6], [0, 6], [5, 1], [2, 4]])
    units = tuple(f"unit-{index}" for index in range(len(positions)))
    region = RegionBatch(
        "valley",
        cx.field(
            features, "row", "column", cx.LabeledAxis("feature", np.array(["corridor", "noise"]))
        ),
        units,
        positions,
    )
    surface = 0.01 + np.exp(features @ np.array([5.0, 0.3]))
    resistance = dense_resistance(surface, positions[:, 0] * corridor.shape[1] + positions[:, 1])
    left, right = np.triu_indices(len(units), 1)
    observations = PairwiseObservations.from_pairs(
        [(units[i], units[j]) for i, j in zip(left, right, strict=True)],
        resistance[left, right],
        target=TargetSpec("synthetic divergence", units="index"),
    )
    return region, observations


# One optimizer and one penalty object share the compiled update across the corridor tests.
_CORRIDOR_OPTIMIZER = optax.lbfgs()
_CORRIDOR_PENALTY = RidgePenalty(prior_sd=3.0, where=lambda model: model.coefficients)


def _train_on_the_corridor(dense_resistance, cap, **settings) -> TrainingState:
    """Trains a log-linear patch encoder capped at 5 on the corridor problem with L-BFGS."""
    region, observations = _corridor_problem(dense_resistance)
    return train(
        region,
        observations,
        distance_model=LogLinearConductance(
            2,
            patch_size=1,
            max_conductance=5.0,
            cap=cap,
            solver_config=ResistanceSolverConfig(rtol=1e-12, atol=1e-12, max_steps=5000),
        ),
        config=TrainingConfig(
            loss=MLPELoss(),
            steps=40,
            optimizer=_CORRIDOR_OPTIMIZER,
            gradient_tolerance=1e-8,
            penalty=_CORRIDOR_PENALTY,
            **settings,
        ),
    )


def _ceiling_gaps(state: TrainingState, dense_resistance) -> np.ndarray:
    """Returns how far below the ceiling the latest model puts each vertex's linear predictor."""
    region, _ = _corridor_problem(dense_resistance)
    model = cast(LogLinearConductance, state.latest_model.distance_model)
    predictors = model.backbone(region.feature_array) @ model.coefficients[0] + model.intercepts[0]
    return _ceiling(model) - np.asarray(predictors)


def _summed_objective_hessian(dense_resistance, state: TrainingState, step: float) -> np.ndarray:
    """Returns the central-difference Hessian of the summed training objective at a trained model.

    The parameters are those training updates: the coefficients, the intercept and the
    statistical parameters of the maximum-likelihood population effects (MLPE) loss.
    """
    region, observations = _corridor_problem(dense_resistance)
    loss, penalty = state.config.loss, state.config.penalty
    assert penalty is not None
    trainable, fixed = eqx.partition(
        (state.latest_model.distance_model, state.loss_state), eqx.is_inexact_array
    )
    point, unravel = ravel_pytree(trainable)

    def objective(flat):
        model, loss_state = eqx.combine(unravel(flat), fixed)
        return loss.sum_over_pairs(region, observations, model, loss_state) + penalty(model)

    gradient = eqx.filter_jit(jax.grad(objective))
    return np.stack(
        [
            (gradient(point + step * unit) - gradient(point - step * unit)) / (2 * step)
            for unit in jnp.eye(point.size)
        ],
        axis=1,
    )


def _eigenvalues(hessian: np.ndarray) -> np.ndarray:
    """Returns the eigenvalues of a finite-difference Hessian's symmetric part, ascending."""
    return np.linalg.eigvalsh((hessian + hessian.T) / 2)


@pytest.mark.parametrize(
    ("objective_tolerance", "stop_reason"), [(None, "steps"), (1e-8, "objective_tolerance")]
)
def test_a_hard_cap_at_the_optimum_keeps_the_gradient_above_the_tolerance(
    dense_resistance, objective_tolerance, stop_reason
):
    state = _train_on_the_corridor(
        dense_resistance, "hard", objective_tolerance=objective_tolerance
    )

    # No update reached the gradient tolerance: the gradient jumps across the ceiling.
    assert not state.converged
    assert state.stop_reason == stop_reason
    assert (state.step == state.config.steps) == (objective_tolerance is None)
    assert np.abs(_ceiling_gaps(state, dense_resistance)).min() < 1e-6
    # A negative eigenvalue of the finite-difference Hessian grows as the step shrinks.
    coarse, fine = (
        _eigenvalues(_summed_objective_hessian(dense_resistance, state, step)).min()
        for step in (1e-4, 1e-5)
    )
    assert fine < 5 * coarse < 0


def test_a_smooth_cap_trains_to_the_gradient_tolerance_with_a_positive_definite_hessian(
    dense_resistance,
):
    state = _train_on_the_corridor(dense_resistance, "smooth")

    assert state.converged
    assert state.stop_reason == "gradient_tolerance"
    assert state.step < state.config.steps
    # The corridor still sits beyond the ceiling.
    assert (_ceiling_gaps(state, dense_resistance) < 0).any()
    hessians = [_summed_objective_hessian(dense_resistance, state, step) for step in (1e-4, 1e-5)]
    for hessian in hessians:
        assert np.abs(hessian - hessian.T).max() <= 1e-6 * np.abs(hessian).max()
        assert _eigenvalues(hessian).min() > 0
    np.testing.assert_allclose(*map(_eigenvalues, hessians), rtol=1e-5)
