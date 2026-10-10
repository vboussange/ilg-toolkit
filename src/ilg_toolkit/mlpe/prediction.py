"""Explicit population-effect prediction without query genetic observations."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve

from ..data import ObservationPartition, PairwiseObservations, TargetSpec
from .fit import MLPEError, MLPEHead


@dataclass(frozen=True)
class MLPEPredictionProvenance:
    """Canonical target-access identities for prediction eligibility checks."""

    mode: str
    region_name: str
    calibration_pairs: tuple[tuple[str, str], ...]
    calibration_roles: tuple[str, ...]
    support_pairs: tuple[tuple[str, str], ...] = ()
    support_roles: tuple[str, ...] = ()
    unseen_effect_policy: str = "independent_prior"


@dataclass(frozen=True)
class MLPEConditionalPrediction:
    """Inverse-transformed means and model-scale variance for new observations.

    ``model_variance`` includes population-effect posterior uncertainty and
    independent residual variance plus configured jitter. It treats the encoder,
    fixed coefficients, and variance parameters as fixed. ``values`` is the
    inverse transform of ``model_values``; a nonlinear transform does not make
    these original-scale expectations or turn the variance into original units.
    """

    values: jax.Array
    model_values: jax.Array
    model_variance: jax.Array
    effect_variance: jax.Array
    residual_variance: float
    jitter: float
    pairs: tuple[tuple[str, str], ...]
    target: TargetSpec
    region_name: str
    provenance: MLPEPredictionProvenance
    scale: str = "original"
    variance_scale: str = "model"
    variance_components: tuple[str, ...] = (
        "population_effect_posterior",
        "residual_variance",
        "configured_jitter",
    )
    excluded_uncertainty: tuple[str, ...] = (
        "encoder",
        "fixed_effect_coefficients",
        "variance_parameters",
    )


def _query(scores, pairs):
    try:
        scores = jnp.asarray(scores, dtype=jnp.float64)
        raw_pairs = tuple(pairs)
        if any(isinstance(pair, str) for pair in raw_pairs):
            raise ValueError("Each pair must contain two labels")
        pairs = tuple(tuple(pair) for pair in raw_pairs)
    except (TypeError, ValueError) as error:
        raise MLPEError(
            "Query scores and pairs must be numeric scores and labelled pairs"
        ) from error
    if scores.shape != (len(pairs),) or not bool(jnp.isfinite(scores).all()):
        raise MLPEError("Query scores must be finite and aligned with labelled pairs")
    if any(
        len(pair) != 2
        or any(not isinstance(label, str) or not label for label in pair)
        or pair[0] == pair[1]
        for pair in pairs
    ):
        raise MLPEError("Query pairs require distinct nonempty sampling-unit labels")
    canonical = tuple(tuple(sorted(pair)) for pair in pairs)
    if len(set(canonical)) != len(canonical):
        raise MLPEError("Query pairs contain duplicate unordered identities, including reversals")
    return scores, pairs, canonical


def _extend(head, population_ids, mean, covariance, pairs):
    extras = tuple(
        dict.fromkeys(label for pair in pairs for label in pair if label not in population_ids)
    )
    n = len(population_ids)
    extended_mean = jnp.zeros(n + len(extras), dtype=jnp.float64).at[:n].set(jnp.asarray(mean))
    extended_covariance = jnp.eye(n + len(extras), dtype=jnp.float64) * head.unit_variance
    extended_covariance = extended_covariance.at[:n, :n].set(jnp.asarray(covariance))
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


@jax.enable_x64()
def _predict(head, scores, pairs, population_ids, mean, covariance, provenance):
    scores, pairs, canonical = _query(scores, pairs)
    if set(canonical) & set(provenance.support_pairs):
        raise MLPEError("Support/query overlap: an unordered support pair cannot be a query")
    population_ids, mean, covariance = _extend(head, population_ids, mean, covariance, pairs)
    left, right = _endpoints(pairs, population_ids)
    model_values = head.intercept + head.slope * (scores - head.score_center) / head.score_scale
    effects, effect_variance, valid = _effect_prediction(mean, covariance, left, right)
    if not bool(valid):
        raise MLPEError("Population-effect prediction covariance is not positive definite")
    model_values += effects
    variance = effect_variance + head.residual_variance + head.config.jitter
    if not bool(jnp.isfinite(model_values).all() & jnp.isfinite(variance).all()):
        raise MLPEError("Conditional prediction produced nonfinite means or variances")
    values = jnp.asarray(head.target.inverse(model_values))
    return MLPEConditionalPrediction(
        values,
        model_values,
        variance,
        effect_variance,
        head.residual_variance,
        head.config.jitter,
        pairs,
        head.target,
        head.region_name,
        provenance,
    )


def predict_known_effects(head: MLPEHead, scores, pairs) -> MLPEConditionalPrediction:
    """Use saved effects for known populations and independent priors for unseen ones.

    This operation is explicit; ordinary ``head.predict_marginal`` retains zero
    population-effect means. Query genetic observations are never accepted.
    """
    provenance = MLPEPredictionProvenance(
        "known_effects",
        head.region_name,
        tuple((min(pair), max(pair)) for pair in head.calibration_pairs),
        head.calibration_roles,
    )
    return _predict(
        head,
        scores,
        pairs,
        head.population_ids,
        head.effect_mean,
        head.effect_covariance,
        provenance,
    )


@dataclass(frozen=True)
class MLPESupportConditioner:
    """A regional posterior updated only from declared support observations.

    The original calibration and encoder remain fixed. Query pairs are supplied
    later, without targets, and must be disjoint from all support pairs.
    """

    head: MLPEHead
    population_ids: tuple[str, ...]
    effect_mean: jax.Array
    effect_covariance: jax.Array
    provenance: MLPEPredictionProvenance

    def predict(self, scores, pairs) -> MLPEConditionalPrediction:
        """Predict disjoint queries, extending unseen effects with independent priors."""
        return _predict(
            self.head,
            scores,
            pairs,
            self.population_ids,
            self.effect_mean,
            self.effect_covariance,
            self.provenance,
        )


@jax.jit
def _condition_posterior(mean, covariance, residual, left, right, noise):
    # Whitened precision has one row/column per sampling unit, including unseen
    # support endpoints. Endpoint gathers replace the explicit incidence matrix.
    prior_factor = jnp.linalg.cholesky(covariance)
    whitened = prior_factor[left] + prior_factor[right]
    precision = jnp.eye(len(mean)) + (whitened.T @ whitened) / noise
    factor = jnp.linalg.cholesky((precision + precision.T) / 2)
    update = cho_solve((factor, True), whitened.T @ (residual - mean[left] - mean[right]) / noise)
    updated_mean = mean + prior_factor @ update
    updated_covariance = prior_factor @ cho_solve((factor, True), prior_factor.T)
    return updated_mean, (updated_covariance + updated_covariance.T) / 2


@jax.enable_x64()
def condition_on_support(
    head: MLPEHead,
    support_scores,
    support_observations: PairwiseObservations,
    *,
    partition: ObservationPartition,
) -> MLPESupportConditioner:
    """Update effects from exactly the explicitly declared support pairs.

    Scores align with ``support_observations.observed_pairs``. The observations
    must contain exactly the declared support set, with the calibrated target
    metadata. Previously consumed calibration pairs cannot be supplied again,
    even in reversed order or under a different input row ordering. This is a
    Gaussian posterior update with fixed encoder, coefficients and variances.
    It factors a population-sized system, never a support-pair covariance.
    """
    if not isinstance(partition, ObservationPartition) or partition.role != "support":
        raise MLPEError("Conditioning requires an explicit support-role partition")
    if partition.region_name != head.region_name:
        raise MLPEError("Support partition region_name does not match calibration")
    if support_observations.target != head.target:
        raise MLPEError("Support target metadata must match calibration")
    scores, pairs, canonical = _query(support_scores, support_observations.observed_pairs)
    if set(canonical) != set(partition.pairs):
        raise MLPEError("Support observations must contain exactly the declared support pairs")
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
    residual = jnp.asarray(
        support_observations.target.forward(support_observations.observed_values)
    )
    residual -= head.intercept + head.slope * (scores - head.score_center) / head.score_scale
    d = head.residual_variance + head.config.jitter
    updated_mean, updated_covariance = _condition_posterior(
        mean, covariance, residual, left, right, d
    )
    if not bool(jnp.isfinite(updated_mean).all() & jnp.isfinite(updated_covariance).all()):
        raise MLPEError("Support update produced a nonfinite population-effect posterior")
    provenance = MLPEPredictionProvenance(
        "support",
        head.region_name,
        calibration_pairs,
        head.calibration_roles,
        canonical,
        ("support",) * len(canonical),
    )
    return MLPESupportConditioner(
        head,
        labels,
        updated_mean,
        updated_covariance,
        provenance,
    )
