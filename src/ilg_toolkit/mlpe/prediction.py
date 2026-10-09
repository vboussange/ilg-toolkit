"""Explicit population-effect prediction without query genetic observations."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_solve

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

    values: np.ndarray
    model_values: np.ndarray
    model_variance: np.ndarray
    effect_variance: np.ndarray
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
        scores = np.asarray(scores, dtype=np.float64)
        raw_pairs = tuple(pairs)
        if any(isinstance(pair, str) for pair in raw_pairs):
            raise ValueError("Each pair must contain two labels")
        pairs = tuple(tuple(pair) for pair in raw_pairs)
    except (TypeError, ValueError) as error:
        raise MLPEError(
            "Query scores and pairs must be numeric scores and labelled pairs"
        ) from error
    if scores.shape != (len(pairs),) or not np.isfinite(scores).all():
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
    extended_mean = np.zeros(n + len(extras))
    extended_covariance = np.eye(n + len(extras)) * head.unit_variance
    extended_mean[:n] = mean
    extended_covariance[:n, :n] = covariance
    return population_ids + extras, extended_mean, extended_covariance


def _incidence(pairs, population_ids):
    lookup = {label: index for index, label in enumerate(population_ids)}
    incidence = np.zeros((len(pairs), len(population_ids)))
    for row, (left, right) in enumerate(pairs):
        incidence[row, [lookup[left], lookup[right]]] = 1.0
    return incidence


def _predict(head, scores, pairs, population_ids, mean, covariance, provenance):
    scores, pairs, canonical = _query(scores, pairs)
    if set(canonical) & set(provenance.support_pairs):
        raise MLPEError("Support/query overlap: an unordered support pair cannot be a query")
    population_ids, mean, covariance = _extend(head, population_ids, mean, covariance, pairs)
    incidence = _incidence(pairs, population_ids)
    model_values = head.intercept + head.slope * (scores - head.score_center) / head.score_scale
    model_values += incidence @ mean
    try:
        factor = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise MLPEError(
            "Population-effect prediction covariance is not positive definite"
        ) from error
    effect_variance = np.sum(np.square(incidence @ factor), axis=1)
    variance = effect_variance + head.residual_variance + head.config.jitter
    if not (np.isfinite(model_values).all() and np.isfinite(variance).all()):
        raise MLPEError("Conditional prediction produced nonfinite means or variances")
    values = head.target.inverse(model_values)
    for array in (values, model_values, variance, effect_variance):
        array.setflags(write=False)
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
        tuple(tuple(sorted(pair)) for pair in head.calibration_pairs),
        head.calibration_roles,
    )
    return _predict(
        head,
        scores,
        pairs,
        head.population_ids,
        np.asarray(head.effect_mean),
        np.asarray(head.effect_covariance),
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
    effect_mean: tuple[float, ...]
    effect_covariance: tuple[tuple[float, ...], ...]
    provenance: MLPEPredictionProvenance

    def predict(self, scores, pairs) -> MLPEConditionalPrediction:
        """Predict disjoint queries, extending unseen effects with independent priors."""
        return _predict(
            self.head,
            scores,
            pairs,
            self.population_ids,
            np.asarray(self.effect_mean),
            np.asarray(self.effect_covariance),
            self.provenance,
        )


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
    calibration_pairs = tuple(tuple(sorted(pair)) for pair in head.calibration_pairs)
    if set(canonical) & set(calibration_pairs):
        raise MLPEError("Support reuses a calibration pair; targets cannot be conditioned on twice")
    labels, mean, covariance = _extend(
        head,
        head.population_ids,
        np.asarray(head.effect_mean),
        np.asarray(head.effect_covariance),
        pairs,
    )
    incidence = _incidence(pairs, labels)
    residual = support_observations.target.forward(support_observations.observed_values)
    residual -= head.intercept + head.slope * (scores - head.score_center) / head.score_scale
    d = head.residual_variance + head.config.jitter
    try:
        # In whitened effect coordinates the posterior precision is I + L' Z' Z L/d.
        # Its dimension is the number of populations, including support endpoints.
        prior_factor = np.linalg.cholesky(covariance)
        whitened = incidence @ prior_factor
        precision = np.eye(len(labels)) + (whitened.T @ whitened) / d
        factor = np.linalg.cholesky((precision + precision.T) / 2)
        update = cho_solve((factor, True), whitened.T @ (residual - incidence @ mean) / d)
        updated_mean = mean + prior_factor @ update
        updated_covariance = prior_factor @ cho_solve((factor, True), prior_factor.T)
        updated_covariance = (updated_covariance + updated_covariance.T) / 2
    except (np.linalg.LinAlgError, ValueError) as error:
        raise MLPEError("Support population-effect posterior factorization failed") from error
    if not (np.isfinite(updated_mean).all() and np.isfinite(updated_covariance).all()):
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
        tuple(updated_mean),
        tuple(map(tuple, updated_covariance)),
        provenance,
    )
