"""Tests of the isolation-by-distance (IBD) model."""

import itertools
import math
import string

import coordax as cx
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from ilg_toolkit import (
    DirectCalibrationHead,
    Ensemble,
    fit_calibration_head,
    LandscapeGeneticsModel,
    load_model,
    Log1pLoss,
    MLPECalibrationHead,
    MLPELoss,
    PairwiseObservations,
    RegionBatch,
    save_model,
    TargetSpec,
    train,
    TrainingConfig,
)
from ilg_toolkit.models import IBDModel
from jaxscape import EuclideanDistance, GridGraph


_DIVERGENCE = TargetSpec("divergence", units="index")
# Slope and intercept estimated by SciPy's limited-memory Broyden–Fletcher–Goldfarb–Shanno
# method with bounds (L-BFGS-B), with non-negative bounds on the mean region log1p squared
# loss, for ``_reference_problem`` from the vertex distances of its sampling units at
# patch_size=2.
_SCIPY_SLOPE = 0.10436956986904002
_SCIPY_INTERCEPT = 0.032285430139206664


def _region(name, shape, positions, *, channels=1, seed=0):
    features = np.random.default_rng(seed).normal(size=(*shape, channels))
    names = cx.LabeledAxis("feature", np.array([f"feature-{index}" for index in range(channels)]))
    ids = tuple(string.ascii_letters[: len(positions)])
    return RegionBatch(name, cx.field(features, "row", "column", names), ids, np.asarray(positions))


def _pairs(region):
    return tuple(itertools.combinations(region.sampling_unit_ids, 2))


def _observations(region, values, target=_DIVERGENCE, *, pairs=None, species=None):
    return PairwiseObservations.from_pairs(
        _pairs(region) if pairs is None else pairs,
        jnp.asarray(values, dtype=jnp.float64),
        target=target,
        sampling_unit_ids=region.sampling_unit_ids,
        species=species,
    )


def _vertex_distances(region, patch_size, pairs=None):
    """Straight-line distances between the pairs' vertices, from the grid positions."""
    vertices = dict(
        zip(region.sampling_unit_ids, np.asarray(region.grid_positions) // patch_size, strict=True)
    )
    return np.array([np.hypot(*(vertices[a] - vertices[b])) for a, b in (pairs or _pairs(region))])


def _selected_ibd_model(state) -> IBDModel:
    model = state.selected_model.distance_model
    assert isinstance(model, IBDModel)
    return model


def _lbfgs(steps=200, loss=None):
    return TrainingConfig(
        loss=Log1pLoss() if loss is None else loss,
        optimizer=optax.lbfgs(),
        steps=steps,
        eval_every=steps,
        gradient_tolerance=1e-10,
    )


def _reference_problem():
    """Returns two regions with unequal pair counts and targets that no affine map fits exactly."""
    alpine = _region("alpine", (8, 8), [[0, 0], [1, 5], [4, 1], [7, 7], [5, 4]])
    coastal = _region("coastal", (6, 10), [[0, 0], [1, 9], [5, 2]], seed=1)
    alpine_targets = [0.21, 0.27, 0.43, 0.30, 0.18, 0.36, 0.22, 0.38, 0.26, 0.12]
    coastal_targets = [0.47, 0.33, 0.41]
    return [alpine, coastal], [
        _observations(alpine, alpine_targets),
        _observations(coastal, coastal_targets),
    ]


def test_lbfgs_training_reproduces_the_scipy_reference():
    regions, observations = _reference_problem()
    # The SciPy reference used these vertex distances.
    np.testing.assert_allclose(
        _vertex_distances(regions[0], 2),
        [2, 2, math.sqrt(18), math.sqrt(8), math.sqrt(8), math.sqrt(10), 2, math.sqrt(10), 2]
        + [math.sqrt(2)],
    )
    np.testing.assert_allclose(_vertex_distances(regions[1], 2), [4, math.sqrt(5), math.sqrt(13)])

    state = train(regions, observations, distance_model=IBDModel(patch_size=2), config=_lbfgs())

    model = _selected_ibd_model(state)
    assert state.converged
    np.testing.assert_allclose(model.slopes, [_SCIPY_SLOPE], rtol=1e-7)
    np.testing.assert_allclose(model.intercepts, [_SCIPY_INTERCEPT], rtol=1e-7)


def test_regions_have_equal_weight_despite_unequal_pair_counts():
    # Sampling units on a line give distances 1, 3, 6, 2, 5 and 3. "large" repeats the line
    # in ten rows and observes pairs within rows only. At each distance the
    # regional targets plus one differ by a factor of four; their geometric mean
    # is 2 * distance + 4, so the log1p squared loss has optimum 2 * distance + 3
    # and value log(2)**2.
    line = [0, 1, 3, 6]
    small = _region("small", (1, 7), [[0, column] for column in line])
    large = _region(
        "large", (20, 7), [[2 * row, column] for row in range(10) for column in line], seed=1
    )
    large_ids = np.array(large.sampling_unit_ids).reshape(10, 4)
    large_pairs = tuple(pair for row in large_ids for pair in itertools.combinations(row, 2))
    small_distances = _vertex_distances(small, 1)
    large_distances = _vertex_distances(large, 1, large_pairs)
    observations = [
        _observations(small, 4 * small_distances + 7),
        _observations(large, large_distances + 1, pairs=large_pairs),
    ]

    state = train(
        [small, large], observations, distance_model=IBDModel(patch_size=1), config=_lbfgs()
    )

    model = _selected_ibd_model(state)
    np.testing.assert_allclose(model.slopes, [2.0], rtol=1e-9)
    np.testing.assert_allclose(model.intercepts, [3.0], rtol=1e-9)
    assert state.history[-1].training_loss == pytest.approx(math.log(2.0) ** 2, abs=1e-12)


def test_affine_targets_are_recovered_through_softplus():
    regions, _ = _reference_problem()
    observations = [
        _observations(region, 0.7 * _vertex_distances(region, 2) + 0.2) for region in regions
    ]

    state = train(regions, observations, distance_model=IBDModel(patch_size=2), config=_lbfgs())

    model = _selected_ibd_model(state)
    np.testing.assert_allclose(model.slopes, [0.7], rtol=1e-9)
    np.testing.assert_allclose(model.intercepts, [0.2], rtol=1e-9)


@pytest.mark.parametrize(
    ("targets", "slope", "intercept"),
    [
        # Targets 9, 0 and 3 at distances 1, 3 and 2 put the optimum on the slope's
        # zero bound, with the intercept at the geometric mean of the targets plus one.
        ([9.0, 0.0, 3.0], 0.0, 40.0 ** (1 / 3) - 1.0),
        # Zero targets put both coefficients on their zero bound.
        ([0.0, 0.0, 0.0], 0.0, 0.0),
    ],
)
def test_softplus_approaches_optima_on_the_zero_bounds(targets, slope, intercept):
    region = _region("ridge", (1, 4), [[0, 0], [0, 1], [0, 3]])
    np.testing.assert_allclose(_vertex_distances(region, 1), [1.0, 3.0, 2.0])

    state = train(
        region,
        _observations(region, targets),
        distance_model=IBDModel(patch_size=1),
        config=_lbfgs(steps=500),
    )

    model = _selected_ibd_model(state)
    np.testing.assert_allclose(model.slopes, [slope], atol=1e-5)
    np.testing.assert_allclose(model.intercepts, [intercept], atol=1e-5)


def test_ecological_distances_are_affine_jaxscape_euclidean_vertex_distances():
    region = _region("valley", (6, 8), [[0, 0], [1, 7], [5, 2], [3, 3], [4, 6]])
    model = IBDModel(patch_size=2, slope=0.3, intercept=0.05)
    # Vertex (row, column) coordinates; 64-bit integers give float64 distances.
    nodes = jnp.asarray(region.grid_positions, dtype=jnp.int64) // 2
    vertex_distances = EuclideanDistance()(GridGraph(jnp.ones((3, 4))), nodes=nodes)
    pairs = (("a", "b"), ("d", "c"), ("e", "a"), ("c", "e"))
    index = {label: position for position, label in enumerate(region.sampling_unit_ids)}
    expected = [0.3 * vertex_distances[index[a], index[b]] + 0.05 for a, b in pairs]

    prediction = model.predict(region, pairs)

    np.testing.assert_allclose(model.slopes, [0.3], rtol=1e-12)
    np.testing.assert_allclose(model.intercepts, [0.05], rtol=1e-12)
    np.testing.assert_allclose(prediction.values, expected, rtol=1e-12)
    np.testing.assert_allclose(prediction.values, 0.3 * _vertex_distances(region, 2, pairs) + 0.05)
    # Landscape features are ignored: their values and count leave distances unchanged.
    other = _region("valley", (6, 8), np.asarray(region.grid_positions), channels=3, seed=7)
    assert eqx.tree_equal(model.predict(other, pairs), prediction)
    # Patch size 1 measures distances between raster cells.
    fine = IBDModel(patch_size=1, slope=0.3, intercept=0.05).predict(region, pairs)
    np.testing.assert_allclose(fine.values, 0.3 * _vertex_distances(region, 1, pairs) + 0.05)
    with pytest.raises(ValueError, match="drop 1 row"):
        IBDModel(patch_size=4).predict(_region("odd", (5, 8), [[0, 0], [4, 7]]))


def test_patch_size_sets_the_vertex_spacing_and_the_slope_absorbs_it():
    regions, observations = _reference_problem()
    doubled = [
        RegionBatch(
            region.name,
            cx.field(
                np.repeat(np.repeat(region.feature_array, 2, axis=0), 2, axis=1),
                "row",
                "column",
                region.features.axes["feature"],
            ),
            region.sampling_unit_ids,
            np.asarray(region.grid_positions) * 2 + 1,
        )
        for region in regions
    ]

    # Each cell becomes a 2 x 2 block: at patch_size=2 the vertices are the original
    # cells, and at patch_size=1 every vertex distance doubles.
    coarse, fine = (
        _selected_ibd_model(
            train(
                doubled,
                observations,
                distance_model=IBDModel(patch_size=patch_size),
                config=_lbfgs(),
            )
        )
        for patch_size in (2, 1)
    )

    np.testing.assert_allclose(fine.slopes, coarse.slopes / 2, rtol=1e-7)
    np.testing.assert_allclose(fine.intercepts, coarse.intercepts, rtol=1e-7)
    for region in doubled:
        np.testing.assert_allclose(
            fine.predict(region).values, coarse.predict(region).values, rtol=1e-7
        )


def test_each_species_has_its_own_slope_and_intercept():
    # Sampling units of each species reuse labels; one frog and one newt share a vertex.
    frog = _region("marsh", (8, 8), [[0, 0], [1, 5], [4, 1], [7, 7]])
    newt = _region("marsh", (8, 8), [[0, 1], [3, 6], [6, 2]])
    region = RegionBatch(
        "marsh",
        frog.features,
        frog.sampling_unit_ids + newt.sampling_unit_ids,
        np.concatenate([frog.grid_positions, newt.grid_positions]),
        ("frog",) * 4 + ("newt",) * 3,
    )
    frog_pairs, newt_pairs = (("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")), _pairs(newt)
    observations = [
        _observations(
            frog,
            0.7 * _vertex_distances(frog, 2, frog_pairs) + 0.2,
            pairs=frog_pairs,
            species="frog",
        ),
        _observations(
            newt, 0.3 * _vertex_distances(newt, 2) + 0.05, pairs=newt_pairs, species="newt"
        ),
    ]
    model = IBDModel(species=("frog", "newt"), patch_size=2)

    state = train(region, observations, distance_model=model, config=_lbfgs())

    trained = _selected_ibd_model(state)
    assert trained.species == ("frog", "newt")
    np.testing.assert_allclose(trained.slopes, [0.7, 0.3], rtol=1e-9)
    np.testing.assert_allclose(trained.intercepts, [0.2, 0.05], rtol=1e-9)
    predictions = trained.predict(region, {"newt": newt_pairs, "frog": None})
    assert list(predictions) == ["newt", "frog"]
    np.testing.assert_allclose(
        predictions["newt"].values, 0.3 * _vertex_distances(newt, 2) + 0.05, rtol=1e-6
    )
    np.testing.assert_allclose(
        predictions["frog"].values, 0.7 * _vertex_distances(frog, 2) + 0.2, rtol=1e-6
    )


def _mlpe_problem():
    region = _region(
        "alpine", (8, 8), [[0, 0], [1, 5], [4, 1], [7, 7], [5, 4], [2, 3], [6, 0]], seed=3
    )
    distances = _vertex_distances(region, 2)
    noise = np.random.default_rng(5).normal(scale=0.02, size=distances.size)
    return region, _observations(region, 0.05 * distances + 0.1 + noise)


def test_mlpe_calibration_heads_fit_regardless_of_the_affine_parameters():
    region, observations = _mlpe_problem()
    steep = IBDModel(patch_size=2, slope=5.0, intercept=2.0)

    heads = [
        fit_calibration_head(region, observations, model, loss=MLPELoss())
        for model in (IBDModel(patch_size=2), steep)
    ]

    calibrations = []
    for head in heads:
        assert isinstance(head, MLPECalibrationHead)
        calibrations.append(head.for_region("alpine"))
    first, second = calibrations
    assert first.converged and second.converged
    # MLPE standardizes distances, so positive affine maps give the same fit.
    np.testing.assert_allclose(first.ml_log_likelihood, second.ml_log_likelihood, rtol=1e-9)
    np.testing.assert_allclose(first.slope, second.slope, rtol=1e-6)
    np.testing.assert_allclose(first.intercept, second.intercept, rtol=1e-6)
    assert first.slope > 0
    predictions = [
        LandscapeGeneticsModel(model, head, region.feature_names).predict(region)
        for model, head in zip((IBDModel(patch_size=2), steep), heads, strict=True)
    ]
    np.testing.assert_allclose(predictions[0].values, predictions[1].values, rtol=1e-6)


def test_odds_targets_round_trip_through_training_and_prediction():
    regions, _ = _reference_problem()
    target = TargetSpec("Fst", units="index", transform="odds")
    odds = [0.7 * _vertex_distances(region, 2) + 0.2 for region in regions]
    fst = [values / (1 + values) for values in odds]
    observations = [
        _observations(region, values, target) for region, values in zip(regions, fst, strict=True)
    ]

    state = train(regions, observations, distance_model=IBDModel(patch_size=2), config=_lbfgs())

    model = state.selected_model
    assert model.target == target
    assert isinstance(model.calibration_head, DirectCalibrationHead)
    np.testing.assert_allclose(_selected_ibd_model(state).slopes, [0.7], rtol=1e-9)
    np.testing.assert_allclose(_selected_ibd_model(state).intercepts, [0.2], rtol=1e-9)
    for region, expected_fst, expected_odds in zip(regions, fst, odds, strict=True):
        prediction = model.predict(region)
        np.testing.assert_allclose(prediction.values, expected_fst, rtol=1e-6)
        np.testing.assert_allclose(target.forward(prediction.values), expected_odds, rtol=1e-6)
        np.testing.assert_allclose(
            model.distance_model.predict(region).values, expected_odds, rtol=1e-6
        )


def test_ensembles_average_members_and_saved_models_reload(tmp_path):
    region, observations = _mlpe_problem()
    members = [
        LandscapeGeneticsModel(
            model,
            fit_calibration_head(region, observations, model, loss=Log1pLoss()),
            region.feature_names,
        )
        for model in (
            IBDModel(patch_size=2, slope=0.04, intercept=0.1),
            IBDModel(patch_size=2, slope=0.06, intercept=0.12),
        )
    ]
    pairs = (("a", "b"), ("c", "g"))
    distances = _vertex_distances(region, 2, pairs)

    result = Ensemble(members).predict(region, pairs)

    np.testing.assert_allclose(result.members[0].values, 0.04 * distances + 0.1, rtol=1e-12)
    np.testing.assert_allclose(result.members[1].values, 0.06 * distances + 0.12, rtol=1e-12)
    np.testing.assert_allclose(result.mean.values, 0.05 * distances + 0.11, rtol=1e-12)

    head = fit_calibration_head(region, observations, members[0].distance_model, loss=MLPELoss())
    saved = LandscapeGeneticsModel(members[0].distance_model, head, region.feature_names)
    path = tmp_path / "ibd.eqx"
    save_model(path, saved)
    restored = load_model(path, distance_model=IBDModel(patch_size=2))

    assert eqx.tree_equal(restored.distance_model, saved.distance_model)
    assert eqx.tree_equal(restored.calibration_head, saved.calibration_head)
    assert eqx.tree_equal(restored.predict(region, pairs), saved.predict(region, pairs))


@pytest.mark.parametrize(
    "settings",
    [
        {"patch_size": 0},
        {"patch_size": 1.5},
        {"patch_size": True},
        {"slope": 0.0},
        {"slope": -1.0},
        {"intercept": 0.0},
        {"intercept": float("nan")},
        {"slope": float("inf")},
        {"slope": True},
        {"species": ()},
        {"species": ("frog", "frog")},
        {"species": "frog"},
    ],
)
def test_constructor_rejects_invalid_settings(settings):
    with pytest.raises(ValueError):
        IBDModel(**settings)
