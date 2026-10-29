"""Maximum-likelihood population effects (MLPE) prediction on the original scale."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve

from ..data import _validate_pairs, PairwiseObservations, TargetSpec
from ..prediction import Prediction
from .fit import MLPECalibration, MLPEError


def _query(distances, pairs):
    """Validates query distances and pairs, and returns them with the canonical pairs."""
    distances = jnp.asarray(distances, dtype=jnp.float64)
    try:
        pairs = _validate_pairs(pairs)
    except ValueError as error:
        raise MLPEError(str(error)) from error
    if distances.shape != (len(pairs),) or not bool(jnp.isfinite(distances).all()):
        raise MLPEError("Query distances must be finite and aligned with labeled pairs")
    canonical = tuple(tuple(sorted(pair)) for pair in pairs)
    return distances, pairs, canonical


def _extend(calibration, population_ids, mean, covariance, pairs):
    """Extends the population-effect posterior with the prior of unseen sampling units."""
    extras = tuple(
        dict.fromkeys(label for pair in pairs for label in pair if label not in population_ids)
    )
    n = len(population_ids)
    extended_mean = jnp.zeros(n + len(extras), dtype=jnp.float64).at[:n].set(mean)
    extended_covariance = jnp.eye(n + len(extras), dtype=jnp.float64) * calibration.unit_variance
    extended_covariance = extended_covariance.at[:n, :n].set(covariance)
    return population_ids + extras, extended_mean, extended_covariance


def _endpoints(pairs, population_ids):
    """Returns the population indices of each pair's two endpoints."""
    lookup = {label: index for index, label in enumerate(population_ids)}
    return (
        jnp.asarray([lookup[left] for left, _ in pairs], dtype=jnp.int32),
        jnp.asarray([lookup[right] for _, right in pairs], dtype=jnp.int32),
    )


@eqx.filter_jit
def _effect_prediction(mean, covariance, left, right):
    """Returns pair population-effect means and variances, and whether the factor is finite."""
    factor = jnp.linalg.cholesky(covariance)
    effect_variance = jnp.sum(jnp.square(factor[left] + factor[right]), axis=1)
    return mean[left] + mean[right], effect_variance, jnp.isfinite(factor).all()


def _predict_mlpe(
    calibration: MLPECalibration,
    distances,
    pairs,
    *,
    target: TargetSpec,
    region_name: str,
    known_effects: bool = False,
    support: PairwiseObservations | None = None,
    support_distances=None,
) -> Prediction:
    """Predicts genetic values from an MLPE calibration and ecological distances.

    ``target`` is the calibration head's target contract. Conditional modes add
    ``model_variance`` and ``effect_variance`` to the ``model_values`` extra, all
    on the model scale. Support distances align with the support observations,
    whose target must be ``target`` and whose pairs are disjoint from calibration
    and query pairs. Residual variance and jitter remain on the calibration.
    """
    if known_effects and support is not None:
        raise MLPEError("known_effects and support are mutually exclusive")
    if (support is None) != (support_distances is None):
        raise MLPEError("Support observations and their distances must be supplied together")
    distances, pairs, canonical = _query(distances, pairs)
    model_values = (
        calibration.intercept
        + calibration.slope * (distances - calibration.distance_center) / calibration.distance_scale
    )
    extras = {}
    if known_effects or support is not None:
        population_ids = calibration.population_ids
        mean, covariance = calibration.effect_mean, calibration.effect_covariance
        if support is not None:
            if set(canonical) & {tuple(sorted(pair)) for pair in support.observed_pairs}:
                raise MLPEError(
                    "Support/query overlap: an unordered support pair cannot be a query"
                )
            if support.target != target:
                raise MLPEError("Support target metadata must match calibration")
            population_ids, mean, covariance = _condition_on_support(
                calibration, support_distances, support
            )
        population_ids, mean, covariance = _extend(
            calibration, population_ids, mean, covariance, pairs
        )
        left, right = _endpoints(pairs, population_ids)
        effects, effect_variance, valid = _effect_prediction(mean, covariance, left, right)
        if not bool(valid):
            raise MLPEError("Population-effect prediction covariance is not positive definite")
        model_values += effects
        variance = effect_variance + calibration.residual_variance + calibration.loss.jitter
        if not bool(jnp.isfinite(model_values).all() & jnp.isfinite(variance).all()):
            raise MLPEError("Conditional prediction produced nonfinite means or variances")
        extras.update(model_variance=variance, effect_variance=effect_variance)
    extras["model_values"] = model_values
    return Prediction(
        target.inverse(model_values),
        pairs,
        region_name,
        tuple(dict.fromkeys(label for pair in pairs for label in pair)),
        extras=extras,
    )


@eqx.filter_jit
def _condition_posterior(mean, covariance, residual, left, right, noise):
    """Conditions the population-effect posterior on support residuals with Gaussian noise."""
    prior_factor = jnp.linalg.cholesky(covariance)
    whitened = prior_factor[left] + prior_factor[right]
    precision = jnp.eye(len(mean)) + (whitened.T @ whitened) / noise
    factor = jnp.linalg.cholesky((precision + precision.T) / 2)
    update = cho_solve((factor, True), whitened.T @ (residual - mean[left] - mean[right]) / noise)
    updated_mean = mean + prior_factor @ update
    updated_covariance = prior_factor @ cho_solve((factor, True), prior_factor.T)
    return updated_mean, (updated_covariance + updated_covariance.T) / 2


def _condition_on_support(
    calibration: MLPECalibration,
    support_distances,
    support_observations: PairwiseObservations,
) -> tuple[tuple[str, ...], jax.Array, jax.Array]:
    """Conditions population effects on support observations with the calibration fixed."""
    distances, pairs, canonical = _query(support_distances, support_observations.observed_pairs)
    calibration_pairs = tuple((min(pair), max(pair)) for pair in calibration.calibration_pairs)
    if set(canonical) & set(calibration_pairs):
        raise MLPEError("Support pairs must be disjoint from the calibration pairs")
    labels, mean, covariance = _extend(
        calibration,
        calibration.population_ids,
        calibration.effect_mean,
        calibration.effect_covariance,
        pairs,
    )
    left, right = _endpoints(pairs, labels)
    residual = support_observations.target.forward(support_observations.observed_values)
    residual -= (
        calibration.intercept
        + calibration.slope * (distances - calibration.distance_center) / calibration.distance_scale
    )
    d = calibration.residual_variance + calibration.loss.jitter
    updated_mean, updated_covariance = _condition_posterior(
        mean, covariance, residual, left, right, jnp.asarray(d)
    )
    if not bool(jnp.isfinite(updated_mean).all() & jnp.isfinite(updated_covariance).all()):
        raise MLPEError("Support update produced a nonfinite population-effect posterior")
    return labels, updated_mean, updated_covariance
