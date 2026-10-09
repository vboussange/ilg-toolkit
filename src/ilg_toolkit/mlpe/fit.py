"""Float64 standalone calibration, distinct from encoder optimization."""

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ..data import ObservationPartition, PairwiseObservations, TargetSpec
from .numpy_system import NumpyPopulationSystem, endpoint_gram


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

    def __post_init__(self):
        scalars = (
            self.score_center,
            self.score_scale,
            self.intercept,
            self.slope,
            self.unit_variance,
            self.residual_variance,
            self.ml_log_likelihood,
        )
        if not all(math.isfinite(value) for value in scalars):
            raise MLPEError("MLPE head cannot retain nonfinite numerical kernel diagnostics")
        if self.score_scale <= self.config.min_score_scale or min(
            self.unit_variance, self.residual_variance
        ) < self.config.variance_floor * (1 - 8 * np.finfo(float).eps):
            raise MLPEError("MLPE head requires valid score scale and positive bounded variances")
        n = len(self.population_ids)
        mean, covariance = np.asarray(self.effect_mean), np.asarray(self.effect_covariance)
        factor = np.asarray(self.effect_precision_cholesky)
        if (
            mean.shape != (n,)
            or covariance.shape != (n, n)
            or factor.shape != (n, n)
            or not np.isfinite(mean).all()
            or not np.isfinite(covariance).all()
            or not np.isfinite(factor).all()
        ):
            raise MLPEError("MLPE head population-effect posterior must be finite and aligned")
        if (
            not self.region_name
            or len(set(self.population_ids)) != n
            or len(self.calibration_pairs) != len(self.calibration_roles)
        ):
            raise MLPEError("MLPE head region, sampling-unit identities and provenance must align")

    def predict_marginal(self, scores, pairs) -> MLPEPrediction:
        """Predict labelled query pairs without targets; new effects have mean zero."""
        from .prediction import _query

        scores, pairs, _ = _query(scores, pairs)
        model_values = self.intercept + self.slope * (scores - self.score_center) / self.score_scale
        return MLPEPrediction(
            self.target.inverse(model_values), model_values, pairs, self.target, self.region_name
        )

    def predict_known_effects(self, scores, pairs):
        """Explicit posterior-effect prediction, with independent priors for unseen units."""
        from .prediction import predict_known_effects

        return predict_known_effects(self, scores, pairs)

    def condition_on_support(self, support_scores, support_observations, *, partition):
        """Update effects using only an explicit support-role partition."""
        from .prediction import condition_on_support

        return condition_on_support(self, support_scores, support_observations, partition=partition)


def _profile(design, targets, system):
    inner, remaining, effects = system.inner_products(np.column_stack((design, targets)))
    beta = np.linalg.solve(inner[:2, :2], inner[:2, 2])
    residual_remaining = remaining[:, 2] - remaining[:, :2] @ beta
    residual_effects = effects[:, 2] - effects[:, :2] @ beta
    quadratic = (
        residual_remaining @ residual_remaining / system.residual
        + residual_effects @ residual_effects / system.unit
    )
    nll = 0.5 * (len(targets) * math.log(2 * math.pi) + system.logdet + quadratic)
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
    left = np.asarray([lookup[a] for a, b in pairs], dtype=np.int32)
    right = np.asarray([lookup[b] for a, b in pairs], dtype=np.int32)
    gram = endpoint_gram(left, right, len(population_ids))
    if np.diag(gram).max() <= 1:
        raise MLPEError("MLPE variances are unidentifiable: observed pairs share no endpoints")
    ols_beta = np.linalg.lstsq(design, targets, rcond=None)[0]
    response_variance = max(
        float(np.var(targets - design @ ols_beta, ddof=1)), config.variance_floor * 100
    )
    bounds = [(math.log(config.variance_floor), math.log(max(response_variance * 1e6, 1e4)))] * 2

    numerical_failure = None

    def objective(log_variances):
        nonlocal numerical_failure
        try:
            variances = np.exp(log_variances)
            system = NumpyPopulationSystem(
                left, right, gram, variances[0], variances[1] + config.jitter
            )
            nll, beta = _profile(design, targets, system)
            return nll if math.isfinite(nll) and np.isfinite(beta).all() else float("inf")
        except (np.linalg.LinAlgError, ValueError) as error:
            numerical_failure = str(error)
            return float("inf")

    candidates = []
    for fraction in (0.01, 0.1, 0.25, 0.5, 0.8):
        start = np.log(
            [
                max(response_variance * fraction / 2, config.variance_floor * 10),
                max(response_variance * (1 - fraction), config.variance_floor * 10),
            ]
        )
        # Rejected covariance points return infinity intentionally. SciPy's
        # finite-difference proposals can subtract infinities; failure status is
        # retained and inspected rather than emitting a redundant NumPy warning.
        with np.errstate(invalid="ignore"):
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
        raise MLPEError(
            "No deterministic MLPE optimization start converged to a finite likelihood"
            + (f"; numerical failure: {numerical_failure}" if numerical_failure else "")
        )
    result = min(candidates, key=lambda candidate: candidate.fun)
    variances = np.exp(result.x)
    try:
        system = NumpyPopulationSystem(
            left, right, gram, variances[0], variances[1] + config.jitter
        )
        nll, beta = _profile(design, targets, system)
        mean, covariance, factor = system.posterior(targets - design @ beta)
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
