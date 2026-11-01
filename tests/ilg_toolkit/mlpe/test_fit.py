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


def _additive_distance_problem(*, additive_sd, unit, residual, slope, seed):
    """Returns observations and distances with an additive term the targets do not share.

    The targets of 30 populations on a 40 x 40 grid follow Euclidean distance. The
    distances add a per-population term, as a terminal's own resistance does on a
    conductance surface.
    """
    count, side = 30, 40
    rng = np.random.default_rng(seed)
    labels = [f"p{index:02d}" for index in range(count)]
    grid = np.column_stack(
        np.unravel_index(rng.choice(side * side, count, replace=False), (side, side))
    )
    euclidean = np.linalg.norm(grid[:, None].astype(float) - grid[None].astype(float), axis=-1)
    left, right = np.triu_indices(count, 1)
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
    ("additive_sd", "unit", "residual", "slope", "dense_optimum"),
    [
        pytest.param(40.0, 1e-3, 2e-4, 0.02, 1121.234, id="stalled-at-a-start"),
        pytest.param(20.0, 1e-3, 2e-5, 0.08, 1568.009, id="stalled-away-from-a-start"),
    ],
)
def test_converged_calibration_is_a_local_optimum_of_the_dense_likelihood(
    mlpe_reference, dense_jax_profile, additive_sd, unit, residual, slope, dense_optimum
):
    observations, distances, targets, left, right = _additive_distance_problem(
        additive_sd=additive_sd, unit=unit, residual=residual, slope=slope, seed=0
    )
    calibration = _fit(mlpe_reference, observations, distances, loss=MLPELoss())

    def log_likelihood(log_variances):
        raw = jnp.log(jnp.expm1(jnp.exp(jnp.asarray(log_variances)) - 1e-10))
        return -float(
            dense_jax_profile(jnp.asarray(distances), jnp.asarray(targets), left, right, 30, raw)[0]
        )

    fitted = np.log([calibration.unit_variance, calibration.residual_variance])
    assert calibration.converged
    assert calibration.ml_log_likelihood == pytest.approx(log_likelihood(fitted), abs=1e-6)
    assert calibration.ml_log_likelihood == pytest.approx(dense_optimum, abs=1e-3)
    for step in itertools.product((-1e-2, 0.0, 1e-2), repeat=2):
        assert log_likelihood(fitted + np.array(step)) <= calibration.ml_log_likelihood + 1e-6


def test_calibration_keeps_an_optimum_on_the_variance_floor(mlpe_reference):
    # Targets without residual noise have their optimum at the smallest residual variance.
    observations, distances, *_ = _additive_distance_problem(
        additive_sd=40.0, unit=1e-3, residual=0.0, slope=0.02, seed=0
    )
    loss = MLPELoss()
    calibration = _fit(mlpe_reference, observations, distances, loss=loss)
    assert calibration.converged
    assert calibration.residual_variance == pytest.approx(loss.variance_floor)


def test_calibration_raises_when_the_optimum_exceeds_the_supported_variance_ratio(mlpe_reference):
    # The unit variance of 1e5 over the variance floor of 1e-10 is a ratio of 1e15.
    observations, distances, *_ = _additive_distance_problem(
        additive_sd=3.0, unit=1e5, residual=0.0, slope=200.0, seed=0
    )
    with pytest.raises(MLPEError, match="No deterministic MLPE optimization start converged"):
        _fit(mlpe_reference, observations, distances, loss=MLPELoss())


def test_calibration_raises_when_rejected_trials_stop_every_start(monkeypatch, mlpe_reference):
    observations, distances, *_ = _additive_distance_problem(
        additive_sd=40.0, unit=1e-3, residual=2e-4, slope=0.02, seed=0
    )
    evaluate = fit._optimizer_value_and_gradient

    def rejecting(log_variances, *args, **kwargs):
        value, gradient = evaluate(log_variances, *args, **kwargs)
        return jnp.where(log_variances[0] - log_variances[1] > 5.0, jnp.nan, value), gradient

    monkeypatch.setattr(fit, "_optimizer_value_and_gradient", rejecting)
    with pytest.raises(MLPEError, match="No deterministic MLPE optimization start converged"):
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
