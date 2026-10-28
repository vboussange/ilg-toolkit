"""Maximum-likelihood population effects (MLPE) prediction on the original scale."""

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve

from ..data import _validate_pairs, PairwiseObservations
from ..prediction import Prediction
from .fit import MLPEError, RegionalMLPECalibration


def _query(distances, pairs):
    distances = jnp.asarray(distances, dtype=jnp.float64)
    try:
        pairs = _validate_pairs(pairs)
    except ValueError as error:
        raise MLPEError(str(error)) from error
    if distances.shape != (len(pairs),) or not bool(jnp.isfinite(distances).all()):
        raise MLPEError("Query distances must be finite and aligned with labeled pairs")
    canonical = tuple(tuple(sorted(pair)) for pair in pairs)
    return distances, pairs, canonical


def _extend(head, population_ids, mean, covariance, pairs):
    extras = tuple(
        dict.fromkeys(label for pair in pairs for label in pair if label not in population_ids)
    )
    n = len(population_ids)
    extended_mean = jnp.zeros(n + len(extras), dtype=jnp.float64).at[:n].set(mean)
    extended_covariance = jnp.eye(n + len(extras), dtype=jnp.float64) * head.unit_variance
    extended_covariance = extended_covariance.at[:n, :n].set(covariance)
    return population_ids + extras, extended_mean, extended_covariance


def _endpoints(pairs, population_ids):
    lookup = {label: index for index, label in enumerate(population_ids)}
    return (
        jnp.asarray([lookup[left] for left, _ in pairs], dtype=jnp.int32),
        jnp.asarray([lookup[right] for _, right in pairs], dtype=jnp.int32),
    )


@jax.jit
def _effect_prediction(mean, covariance, left, right):
    factor = jnp.linalg.cholesky(covariance)
    effect_variance = jnp.sum(jnp.square(factor[left] + factor[right]), axis=1)
    return mean[left] + mean[right], effect_variance, jnp.isfinite(factor).all()


def _predict_mlpe(
    calibration: RegionalMLPECalibration,
    distances,
    pairs,
    *,
    known_effects: bool = False,
    support: PairwiseObservations | None = None,
    support_distances=None,
) -> Prediction:
    """Genetic predictions from a fitted MLPE calibration and ecological distances.

    Conditional modes add ``model_variance`` and ``effect_variance`` to the
    ``model_values`` extra, all on the model scale. Support distances align with
    the support observations, whose pairs are disjoint from calibration and
    query pairs. Residual variance and jitter remain on the calibration.
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
        calibration.target.inverse(model_values),
        pairs,
        calibration.region_name,
        tuple(dict.fromkeys(label for pair in pairs for label in pair)),
        extras=extras,
    )


@jax.jit
def _condition_posterior(mean, covariance, residual, left, right, noise):
    # Population-effect precision includes every sampling unit observed in support.
    prior_factor = jnp.linalg.cholesky(covariance)
    whitened = prior_factor[left] + prior_factor[right]
    precision = jnp.eye(len(mean)) + (whitened.T @ whitened) / noise
    factor = jnp.linalg.cholesky((precision + precision.T) / 2)
    update = cho_solve((factor, True), whitened.T @ (residual - mean[left] - mean[right]) / noise)
    updated_mean = mean + prior_factor @ update
    updated_covariance = prior_factor @ cho_solve((factor, True), prior_factor.T)
    return updated_mean, (updated_covariance + updated_covariance.T) / 2


def _condition_on_support(
    head: RegionalMLPECalibration,
    support_distances,
    support_observations: PairwiseObservations,
) -> tuple[tuple[str, ...], jax.Array, jax.Array]:
    """Population effects conditioned on support observations with calibration fixed."""
    if support_observations.target != head.target:
        raise MLPEError("Support target metadata must match calibration")
    distances, pairs, canonical = _query(support_distances, support_observations.observed_pairs)
    calibration_pairs = tuple((min(pair), max(pair)) for pair in head.calibration_pairs)
    if set(canonical) & set(calibration_pairs):
        raise MLPEError("Support reuses a calibration pair; targets cannot be conditioned on twice")
    labels, mean, covariance = _extend(
        head,
        head.population_ids,
        head.effect_mean,
        head.effect_covariance,
        pairs,
    )
    left, right = _endpoints(pairs, labels)
    residual = support_observations.target.forward(support_observations.observed_values)
    residual -= (
        head.intercept + head.slope * (distances - head.distance_center) / head.distance_scale
    )
    d = head.residual_variance + head.loss.jitter
    updated_mean, updated_covariance = _condition_posterior(
        mean, covariance, residual, left, right, d
    )
    if not bool(jnp.isfinite(updated_mean).all() & jnp.isfinite(updated_covariance).all()):
        raise MLPEError("Support update produced a nonfinite population-effect posterior")
    return labels, updated_mean, updated_covariance
