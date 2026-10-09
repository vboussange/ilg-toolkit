"""Float64 standalone calibration, distinct from encoder optimization."""

import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_solve
from scipy.optimize import minimize

from ..data import ObservationPartition, PairwiseObservations, TargetSpec


class MLPEError(ValueError):
    """Invalid or numerically unusable full-ML calibration."""


@dataclass(frozen=True)
class MLPEConfig:
    """Explicit constraints; fitting always uses NumPy float64.

    Jitter is an explicitly declared addition to residual observation variance,
    consistently included in likelihood and population-effect posterior. It is
    never increased automatically. Both fixed coefficients remain signed.
    """

    variance_floor: float = 1e-10
    min_score_scale: float = 1e-12
    jitter: float = 0.0
    max_iterations: int = 2000

    def __post_init__(self):
        for name in ("variance_floor", "min_score_scale"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise MLPEError(f"{name} must be finite and positive")
        if not math.isfinite(self.jitter) or not 0 <= self.jitter <= 1e-6:
            raise MLPEError("jitter must be finite and in [0, 1e-6]; no automatic increases")
        if not isinstance(self.max_iterations, int) or self.max_iterations < 1:
            raise MLPEError("max_iterations must be a positive integer")


@dataclass(frozen=True)
class MLPEPrediction:
    """Marginal pair means on declared original and fitted target scales.

    Original-scale values inverse-transform the fitted Gaussian mean. For a
    nonlinear transform this is not the expectation of the original random
    target; no distributional mean correction is implied.
    """

    values: np.ndarray
    model_values: np.ndarray
    pairs: tuple[tuple[str, str], ...]
    target: TargetSpec
    region_name: str
    scale: str = "original"


@dataclass(frozen=True)
class MLPEHead:
    """Complete regional calibration with auditable observation identities."""

    region_name: str
    target: TargetSpec
    population_ids: tuple[str, ...]
    score_center: float
    score_scale: float
    intercept: float
    slope: float
    unit_variance: float
    residual_variance: float
    effect_mean: tuple[float, ...]
    effect_covariance: tuple[tuple[float, ...], ...]
    effect_precision_cholesky: tuple[tuple[float, ...], ...]
    calibration_pairs: tuple[tuple[str, str], ...]
    calibration_roles: tuple[str, ...]
    ml_log_likelihood: float
    config: MLPEConfig
    optimizer_iterations: int
    optimizer_message: str
    converged: bool = True

    def predict_marginal(self, scores, pairs) -> MLPEPrediction:
        """Predict labelled query pairs without targets; new effects have mean zero."""
        scores = np.asarray(scores, dtype=np.float64)
        pairs = tuple(tuple(pair) for pair in pairs)
        if scores.ndim != 1 or not np.isfinite(scores).all() or len(scores) != len(pairs):
            raise MLPEError("Query scores must be finite and aligned with labelled pairs")
        if any(
            len(pair) != 2
            or any(not isinstance(label, str) or not label for label in pair)
            or pair[0] == pair[1]
            for pair in pairs
        ):
            raise MLPEError("Query pairs require distinct nonempty sampling-unit labels")
        model_values = self.intercept + self.slope * (scores - self.score_center) / self.score_scale
        return MLPEPrediction(
            self.target.inverse(model_values), model_values, pairs, self.target, self.region_name
        )


def _profile(design, targets, incidence, variances, jitter):
    covariance = variances[0] * (incidence @ incidence.T)
    covariance += (variances[1] + jitter) * np.eye(len(targets))
    factor = np.linalg.cholesky(covariance)
    vinv_design = cho_solve((factor, True), design)
    beta = np.linalg.solve(design.T @ vinv_design, design.T @ cho_solve((factor, True), targets))
    residual = targets - design @ beta
    nll = 0.5 * (
        len(targets) * math.log(2 * math.pi)
        + 2 * np.log(np.diag(factor)).sum()
        + residual @ cho_solve((factor, True), residual)
    )
    return float(nll), beta


def calibrate_mlpe(
    scores,
    observations: PairwiseObservations,
    *,
    region_name: str,
    partition: ObservationPartition | None = None,
    config: MLPEConfig | None = None,
) -> MLPEHead:
    """Fit full ML from scores aligned to ``observations.observed_pairs``.

    ``partition`` explicitly selects a subset and records its declared data role.
    Missing observations remain absent. This operation never updates an encoder.
    """
    config = MLPEConfig() if config is None else config
    scores = np.asarray(scores, dtype=np.float64)
    pairs = observations.observed_pairs
    targets = observations.observed_values
    if not isinstance(region_name, str) or not region_name:
        raise MLPEError("region_name must be declared")
    if observations.target.kind != "dissimilarity":
        raise MLPEError(
            "Population MLPE supports declared dissimilarities; relatedness needs a model"
        )
    if scores.shape != (len(pairs),) or not np.isfinite(scores).all():
        raise MLPEError("Calibration scores must be finite and aligned with observed_pairs")
    role = "calibration"
    if partition is not None:
        if partition.role in {"support", "query"}:
            raise MLPEError("Calibration cannot consume query or support partitions")
        if partition.region_name != region_name:
            raise MLPEError("Calibration partition region_name does not match")
        selected = set(partition.pairs)
        if not selected.issubset({tuple(sorted(pair)) for pair in pairs}):
            raise MLPEError("Calibration partition selects unobserved pairs")
        mask = np.array([tuple(sorted(pair)) in selected for pair in pairs])
        pairs = tuple(pair for pair, keep in zip(pairs, mask, strict=True) if keep)
        scores, targets = scores[mask], targets[mask]
        role = partition.role
    targets = observations.target.forward(targets)
    if len(pairs) < 3:
        raise MLPEError("MLPE fitting requires at least three observed pairs")
    center, scale = float(np.mean(scores)), float(np.std(scores, ddof=1))
    if not math.isfinite(scale) or scale <= config.min_score_scale:
        raise MLPEError("Calibration scores are constant or ill-scaled; singular GLS design")
    design = np.column_stack((np.ones(len(scores)), (scores - center) / scale))
    if np.linalg.matrix_rank(design) != 2:
        raise MLPEError("Calibration GLS design is singular")
    observed = {endpoint for pair in pairs for endpoint in pair}
    population_ids = tuple(label for label in observations.sampling_unit_ids if label in observed)
    lookup = {label: index for index, label in enumerate(population_ids)}
    incidence = np.zeros((len(pairs), len(population_ids)))
    for row, pair in enumerate(pairs):
        incidence[row, [lookup[pair[0]], lookup[pair[1]]]] = 1
    if np.max(incidence.sum(axis=0)) <= 1:
        raise MLPEError("MLPE variances are unidentifiable: observed pairs share no endpoints")
    ols_beta = np.linalg.lstsq(design, targets, rcond=None)[0]
    response_variance = max(
        float(np.var(targets - design @ ols_beta, ddof=1)), config.variance_floor * 100
    )
    bounds = [(math.log(config.variance_floor), math.log(max(response_variance * 1e6, 1e4)))] * 2

    def objective(log_variances):
        try:
            nll, beta = _profile(design, targets, incidence, np.exp(log_variances), config.jitter)
            return nll if math.isfinite(nll) and np.isfinite(beta).all() else float("inf")
        except np.linalg.LinAlgError:
            return float("inf")

    candidates = []
    for fraction in (0.01, 0.1, 0.25, 0.5, 0.8):
        start = np.log(
            [
                max(response_variance * fraction / 2, config.variance_floor * 10),
                max(response_variance * (1 - fraction), config.variance_floor * 10),
            ]
        )
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": config.max_iterations, "ftol": 1e-12, "gtol": 1e-9},
        )
        if result.success and math.isfinite(result.fun):
            candidates.append(result)
    if not candidates:
        raise MLPEError("No deterministic MLPE optimization start converged to a finite likelihood")
    result = min(candidates, key=lambda candidate: candidate.fun)
    variances = np.exp(result.x)
    nll, beta = _profile(design, targets, incidence, variances, config.jitter)
    residual = targets - design @ beta
    precision = np.eye(len(population_ids)) / variances[0]
    precision += (incidence.T @ incidence) / (variances[1] + config.jitter)
    try:
        factor = np.linalg.cholesky(precision)
        covariance = cho_solve((factor, True), np.eye(len(population_ids)))
        mean = covariance @ (incidence.T @ residual) / (variances[1] + config.jitter)
    except (np.linalg.LinAlgError, ValueError) as error:
        raise MLPEError("Fitted population-effect posterior factorization failed") from error
    if not (np.isfinite(mean).all() and np.isfinite(covariance).all()):
        raise MLPEError("Fitted population-effect posterior is nonfinite")
    return MLPEHead(
        region_name,
        observations.target,
        population_ids,
        center,
        scale,
        float(beta[0]),
        float(beta[1]),
        float(variances[0]),
        float(variances[1]),
        tuple(mean),
        tuple(map(tuple, (covariance + covariance.T) / 2)),
        tuple(map(tuple, factor)),
        pairs,
        (role,) * len(pairs),
        -nll,
        config,
        int(result.nit),
        str(result.message),
    )
