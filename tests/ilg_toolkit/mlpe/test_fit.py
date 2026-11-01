"""Tests of maximum-likelihood population effects (MLPE) calibrations against R references."""

import itertools
from dataclasses import replace
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from ilg_toolkit import (
    fit_calibration_head,
    MLPECalibrationHead,
    MLPELoss,
    PairwiseObservations,
    TargetSpec,
)
from ilg_toolkit.mlpe import fit, MLPECalibration, MLPEError
from scipy.optimize import minimize


def _fit(mlpe_reference, observations, distances, **options) -> MLPECalibration:
    region, distance_model = mlpe_reference.problem(observations.observed_pairs, distances)
    head = fit_calibration_head(region, observations, distance_model, **options)
    assert isinstance(head, MLPECalibrationHead)
    return head.for_region(region.name)


def test_calibration_matches_frozen_r_full_ml_and_predicts_without_query_targets(
    reference_observations, mlpe_reference
):
    observations = reference_observations()
    shift = mlpe_reference.shift
    calibration = _fit(
        mlpe_reference, observations, mlpe_reference.distances + shift, loss=MLPELoss()
    )
    np.testing.assert_allclose(
        [calibration.intercept, calibration.slope], mlpe_reference.fixed, rtol=3e-5, atol=3e-6
    )
    np.testing.assert_allclose(
        [calibration.unit_variance, calibration.residual_variance],
        mlpe_reference.variances,
        rtol=2e-4,
        atol=2e-6,
    )
    assert calibration.ml_log_likelihood == pytest.approx(mlpe_reference.log_likelihood, abs=2e-5)
    np.testing.assert_allclose(calibration.effect_mean, mlpe_reference.blups, rtol=2e-4, atol=2e-5)
    assert calibration.distance_center == pytest.approx(-0.14328369260805038 + shift, abs=1e-14)
    assert calibration.distance_scale == pytest.approx(1.012896471531856, abs=1e-14)
    assert calibration.calibration_pairs == observations.observed_pairs
    head = MLPECalibrationHead(observations.target, {("alpine", None): calibration})
    query = head.predict(
        np.array([2.5, -2.0]) + shift, [("new", "p01"), ("p02", "new")], region_name="alpine"
    )
    # Independent frozen predictions from the R coefficients and distance moments.
    np.testing.assert_allclose(query.values, [-0.49407802616878604, 2.4619456646531583], atol=1e-5)
    assert query.values[0] < 0  # Signed calibration is not clipped or repaired.


def test_calibration_honors_the_declared_distance_scale_threshold(
    reference_observations, mlpe_reference
):
    distances = (mlpe_reference.distances + mlpe_reference.shift) * 1e-14
    observations = reference_observations()
    with pytest.raises(MLPEError):
        _fit(mlpe_reference, observations, distances, loss=MLPELoss())
    calibration = _fit(
        mlpe_reference, observations, distances, loss=MLPELoss(min_distance_scale=1e-16)
    )
    np.testing.assert_allclose(
        [calibration.intercept, calibration.slope], mlpe_reference.fixed, rtol=3e-5, atol=3e-6
    )
    np.testing.assert_allclose(
        [calibration.unit_variance, calibration.residual_variance],
        mlpe_reference.variances,
        rtol=2e-4,
        atol=2e-6,
    )


def test_calibration_rejects_unidentified_variances(reference_observations, mlpe_reference):
    observations = reference_observations()
    disjoint = PairwiseObservations.from_pairs(
        [("a", "b"), ("c", "d"), ("e", "f")], [1, 3, 2], target=observations.target
    )
    with pytest.raises(ValueError, match="unidentifiable"):
        _fit(mlpe_reference, disjoint, [1, 2, 3], loss=MLPELoss())
    with pytest.raises(MLPEError):
        _fit(mlpe_reference, observations, np.ones(len(mlpe_reference.distances)), loss=MLPELoss())


def test_explicit_numerical_constraints_and_optimization_failure_are_visible(
    reference_observations, mlpe_reference
):
    invalid_options: tuple[dict[str, Any], ...] = (
        {"variance_floor": 0},
        {"min_distance_scale": float("nan")},
        {"jitter": 1e-3},
    )
    for options in invalid_options:
        with pytest.raises(ValueError):
            MLPELoss(**options)
    distances = mlpe_reference.distances + mlpe_reference.shift
    for max_iterations in (1, 0, 2.5):
        with pytest.raises(MLPEError):
            _fit(
                mlpe_reference,
                reference_observations(),
                distances,
                loss=MLPELoss(),
                max_iterations=max_iterations,
            )


def _additive_distance_problem(*, additive_sd, unit, residual, slope, seed, count=30, keep=1.0):
    """Returns observations and distances with an additive term the targets do not share.

    The targets of ``count`` populations on a 40 x 40 grid follow Euclidean distance.
    The distances add a per-population term, as a terminal's own resistance does on a
    conductance surface. Each pair is observed with probability ``keep``.
    """
    side = 40
    rng = np.random.default_rng(seed)
    labels = [f"p{index:02d}" for index in range(count)]
    grid = np.column_stack(
        np.unravel_index(rng.choice(side * side, count, replace=False), (side, side))
    )
    euclidean = np.linalg.norm(grid[:, None].astype(float) - grid[None].astype(float), axis=-1)
    left, right = np.triu_indices(count, 1)
    if keep < 1.0:
        observed = rng.random(left.size) < keep
        left, right = left[observed], right[observed]
    pair_distances = euclidean[left, right]
    effects = rng.normal(0.0, np.sqrt(unit), count)
    targets = (
        0.6
        + slope * (pair_distances - pair_distances.mean()) / pair_distances.std()
        + effects[left]
        + effects[right]
        + rng.normal(0.0, np.sqrt(residual), left.size)
    )
    additive = rng.normal(0.0, additive_sd, count)
    distances = pair_distances + additive[left] + additive[right]
    observations = PairwiseObservations.from_pairs(
        [(labels[a], labels[b]) for a, b in zip(left, right, strict=True)],
        targets,
        target=TargetSpec("synthetic dissimilarity", units="index"),
    )
    return observations, distances - min(distances.min(), 0.0), targets, left, right


@pytest.mark.parametrize(
    ("count", "additive_sd", "unit", "residual", "slope", "seed", "dense_optimum"),
    [
        pytest.param(30, 40.0, 1e-3, 2e-4, 0.02, 0, 1121.234, id="stalled-at-a-start"),
        pytest.param(30, 20.0, 1e-3, 2e-5, 0.08, 0, 1568.009, id="stalled-away-from-a-start"),
        pytest.param(12, 40.0, 1e-3, 2e-4, 0.5, 1, 119.389, id="line-search-ends-abnormally"),
        pytest.param(6, 20.0, 1e-3, 2e-4, 200.0, 1, -21.622, id="mode-reached-by-one-start"),
        pytest.param(30, 1000.0, 1e-3, 2e-4, 0.02, 0, 1027.199, id="mode-at-a-large-ratio"),
        pytest.param(12, 100.0, 1.0, 1e-3, 200.0, 2, -12.551, id="starts-stranded-at-the-limit"),
        pytest.param(30, 100.0, 1e-3, 1e-3, 0.02, 1, 805.219, id="start-stranded-below-the-best"),
    ],
)
def test_converged_calibration_is_a_local_optimum_of_the_dense_likelihood(
    mlpe_reference,
    dense_jax_profile,
    count,
    additive_sd,
    unit,
    residual,
    slope,
    seed,
    dense_optimum,
):
    observations, distances, targets, left, right = _additive_distance_problem(
        additive_sd=additive_sd, unit=unit, residual=residual, slope=slope, seed=seed, count=count
    )
    calibration = _fit(mlpe_reference, observations, distances, loss=MLPELoss())

    def log_likelihood(log_variances):
        positive = jnp.exp(jnp.asarray(log_variances)) - 1e-10
        raw = positive + jnp.log(-jnp.expm1(-positive))
        return -float(
            dense_jax_profile(
                jnp.asarray(distances), jnp.asarray(targets), left, right, count, raw
            )[0]
        )

    fitted = np.log([calibration.unit_variance, calibration.residual_variance])
    assert calibration.converged
    assert calibration.ml_log_likelihood == pytest.approx(log_likelihood(fitted), abs=1e-5)
    assert calibration.ml_log_likelihood == pytest.approx(dense_optimum, abs=1e-3)
    for step in itertools.product((-1e-2, 0.0, 1e-2), repeat=2):
        assert log_likelihood(fitted + np.array(step)) <= calibration.ml_log_likelihood + 1e-5


@pytest.mark.parametrize(
    ("additive_sd", "unit", "residual", "jitter", "count", "keep", "component"),
    [
        pytest.param(40.0, 1e-3, 0.0, 0.0, 30, 1.0, "residual_variance", id="no-residual-noise"),
        pytest.param(0.0, 0.0, 2e-4, 0.0, 30, 1.0, "unit_variance", id="no-population-effects"),
        pytest.param(3.0, 1.0, 0.0, 0.0, 30, 1.0, "residual_variance", id="large-unit-variance"),
        pytest.param(3.0, 1e3, 0.0, 1e-6, 30, 1.0, "residual_variance", id="ratio-held-by-jitter"),
        pytest.param(0.0, 1.0, 0.2, 0.0, 12, 0.2, "residual_variance", id="few-pairs"),
    ],
)
def test_calibration_keeps_an_optimum_on_the_variance_floor(
    mlpe_reference, additive_sd, unit, residual, jitter, count, keep, component
):
    observations, distances, *_ = _additive_distance_problem(
        additive_sd=additive_sd,
        unit=unit,
        residual=residual,
        slope=0.02,
        seed=0,
        count=count,
        keep=keep,
    )
    loss = MLPELoss(jitter=jitter)
    calibration = _fit(mlpe_reference, observations, distances, loss=loss)
    assert getattr(calibration, component) == pytest.approx(loss.variance_floor)


def test_calibration_restarts_a_start_that_stops_away_from_a_local_optimum(
    monkeypatch, mlpe_reference
):
    # Every run stops after three iterations, as a collapsed line search does.
    observations, distances, *_ = _additive_distance_problem(
        additive_sd=40.0, unit=1e-3, residual=2e-4, slope=0.02, seed=0
    )

    def stopping_early(objective, start, **options):
        return minimize(
            objective, start, **options | {"options": options["options"] | {"maxiter": 3}}
        )

    monkeypatch.setattr(fit, "minimize", stopping_early)
    calibration = _fit(mlpe_reference, observations, distances, loss=MLPELoss())
    assert calibration.ml_log_likelihood == pytest.approx(1121.234, abs=1e-3)


@pytest.mark.parametrize(
    ("count", "additive_sd", "unit", "slope"),
    [
        # The unit variance of 1e5 over the variance floor of 1e-10 is a ratio of 1e15.
        pytest.param(30, 3.0, 1e5, 200.0, id="every-start-at-the-limit"),
        pytest.param(6, 40.0, 1e-3, 200.0, id="other-starts-at-a-lower-optimum"),
    ],
)
def test_calibration_raises_when_the_optimum_exceeds_the_supported_variance_ratio(
    mlpe_reference, count, additive_sd, unit, slope
):
    observations, distances, *_ = _additive_distance_problem(
        additive_sd=additive_sd, unit=unit, residual=0.0, slope=slope, seed=0, count=count
    )
    with pytest.raises(MLPEError, match="alpine, species None: .* at the variance ratio limit"):
        _fit(mlpe_reference, observations, distances, loss=MLPELoss())


def test_calibration_raises_when_rejected_trials_keep_every_start_from_the_optimum(
    monkeypatch, mlpe_reference
):
    # The dense optimum has a log variance ratio of 3.36, so no trial reaches it.
    observations, distances, *_ = _additive_distance_problem(
        additive_sd=40.0, unit=1e-3, residual=2e-4, slope=0.02, seed=0
    )
    evaluate = fit._optimizer_value_and_gradient

    def rejecting(parameters, *args, max_log_ratio, jitter, **kwargs):
        settings = dict(max_log_ratio=max_log_ratio, jitter=jitter)
        value, gradient = evaluate(parameters, *args, **settings, **kwargs)
        log_unit, log_residual = fit._log_variances(parameters, **settings)
        return jnp.where(log_unit - log_residual > 2.0, jnp.nan, value), gradient

    monkeypatch.setattr(fit, "_optimizer_value_and_gradient", rejecting)
    with pytest.raises(MLPEError, match="stopped away from a local optimum"):
        _fit(mlpe_reference, observations, distances, loss=MLPELoss())


def test_calibration_raises_without_a_finite_likelihood(monkeypatch, mlpe_reference):
    observations, distances, *_ = _additive_distance_problem(
        additive_sd=40.0, unit=1e-3, residual=2e-4, slope=0.02, seed=0
    )
    evaluate = fit._optimizer_value_and_gradient

    def rejecting(parameters, *args, **kwargs):
        _, gradient = evaluate(parameters, *args, **kwargs)
        return jnp.nan, gradient

    monkeypatch.setattr(fit, "_optimizer_value_and_gradient", rejecting)
    with pytest.raises(MLPEError, match="reached a finite likelihood"):
        _fit(mlpe_reference, observations, distances, loss=MLPELoss())


def test_mlpe_calibrations_reject_invalid_fitted_values(calibrated):
    head, _ = calibrated
    calibration = head.for_region("alpine")
    count = len(calibration.population_ids)
    for changes in (
        {"slope": float("nan")},
        {"distance_scale": 0.0},
        {"unit_variance": 0.0},
        {"effect_mean": np.zeros(count + 1)},
        {"effect_covariance": np.full((count, count), np.inf)},
        {"population_ids": ("p01",) * count},
    ):
        with pytest.raises(MLPEError):
            replace(calibration, **changes)
