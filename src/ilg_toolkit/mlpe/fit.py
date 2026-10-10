"""Float64 standalone calibration, distinct from encoder optimization."""

import math
from dataclasses import dataclass
from functools import partial
from numbers import Real

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from ..data import ObservationPartition, PairwiseObservations, TargetSpec
from .likelihood import mlpe_effect_posterior, profiled_mlpe_ml_fit


class MLPEError(ValueError):
    """Invalid or numerically unusable full-ML calibration."""


def _raw_from_log_variances(log_variances):
    variances = jnp.exp(log_variances)
    return variances + jnp.log(-jnp.expm1(-variances))


@partial(jax.jit, static_argnames=("n_populations",))
def _optimizer_value_and_gradient(log_variances, scores, targets, left, right, **options):
    def objective(parameters):
        return profiled_mlpe_ml_fit(
            scores,
            targets,
            left,
            right,
            raw_variances=_raw_from_log_variances(parameters),
            **options,
        )[0]

    return jax.value_and_grad(objective)(log_variances)


@dataclass(frozen=True)
class MLPEConfig:
    """Explicit constraints; standalone fitting uses scoped JAX float64.

    Jitter is an explicitly declared addition to residual observation variance,
    consistently included in likelihood and population-effect posterior. It is
    never increased automatically. Both fixed coefficients remain signed.
    """

    variance_floor: float = 1e-10
    min_score_scale: float = 1e-12
    jitter: float = 0.0
    max_iterations: int = 2000

    def __post_init__(self):
        for name in ("variance_floor", "min_score_scale", "jitter", "max_iterations"):
            if isinstance(getattr(self, name), (bool, np.bool_)):
                raise MLPEError(f"{name} cannot be boolean")
        for name in ("variance_floor", "min_score_scale"):
            value = getattr(self, name)
            if not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
                raise MLPEError(f"{name} must be finite and positive")
            object.__setattr__(self, name, float(value))
        if (
            not isinstance(self.jitter, Real)
            or not math.isfinite(self.jitter)
            or not 0 <= self.jitter <= 1e-6
        ):
            raise MLPEError("jitter must be finite and in [0, 1e-6]; no automatic increases")
        if not isinstance(self.max_iterations, int) or self.max_iterations < 1:
            raise MLPEError("max_iterations must be a positive integer")
        object.__setattr__(self, "jitter", float(self.jitter))


@dataclass(frozen=True)
class MLPEPrediction:
    """Marginal pair means on declared original and fitted target scales.

    Original-scale values inverse-transform the fitted Gaussian mean. For a
    nonlinear transform this is not the expectation of the original random
    target; no distributional mean correction is implied.
    """

    values: jax.Array
    model_values: jax.Array
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

    @jax.enable_x64()
    def predict_marginal(self, scores, pairs) -> MLPEPrediction:
        """Predict labelled query pairs without targets; new effects have mean zero."""
        from .prediction import _query

        scores, pairs, _ = _query(scores, pairs)
        model_values = self.intercept + self.slope * (scores - self.score_center) / self.score_scale
        return MLPEPrediction(
            jnp.asarray(self.target.inverse(model_values)),
            model_values,
            pairs,
            self.target,
            self.region_name,
        )

    def predict_known_effects(self, scores, pairs):
        """Explicit posterior-effect prediction, with independent priors for unseen units."""
        from .prediction import predict_known_effects

        return predict_known_effects(self, scores, pairs)

    def condition_on_support(self, support_scores, support_observations, *, partition):
        """Update effects using only an explicit support-role partition."""
        from .prediction import condition_on_support

        return condition_on_support(self, support_scores, support_observations, partition=partition)


@jax.enable_x64()
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
    if np.bincount(np.r_[left, right], minlength=len(population_ids)).max() <= 1:
        raise MLPEError("MLPE variances are unidentifiable: observed pairs share no endpoints")
    ols_beta = np.linalg.lstsq(design, targets, rcond=None)[0]
    response_variance = max(
        float(np.var(targets - design @ ols_beta, ddof=1)), config.variance_floor * 100
    )
    bounds = [(math.log(config.variance_floor), math.log(max(response_variance * 1e6, 1e4)))] * 2

    # Bounds enforce the standalone variance floor. Decode these bounded log
    # variances through the common softplus interface with no additional floor;
    # this preserves the exact lower bound without subtracting nearly equal
    # values or changing the joint trainer's raw-parameter policy.
    kernel_inputs = tuple(jnp.asarray(value) for value in (scores, targets, left, right))
    kernel_options = dict(
        n_populations=len(population_ids),
        score_center=center,
        score_scale=scale,
        variance_floor=0.0,
        jitter=config.jitter,
    )

    def profile(log_variances):
        return profiled_mlpe_ml_fit(
            *kernel_inputs, raw_variances=_raw_from_log_variances(log_variances), **kernel_options
        )

    def objective(log_variances):
        value, gradient = _optimizer_value_and_gradient(
            jnp.asarray(log_variances), *kernel_inputs, **kernel_options
        )
        value, gradient = float(value), np.asarray(gradient)
        if not math.isfinite(value) or not np.isfinite(gradient).all():
            return float("inf"), np.zeros_like(log_variances)
        return value, gradient

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
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": config.max_iterations, "ftol": 1e-12, "gtol": 1e-9},
        )
        if result.success and math.isfinite(result.fun):
            candidates.append(result)
    if not candidates:
        raise MLPEError(
            "No deterministic MLPE optimization start converged to a finite likelihood"
            "; check score variation and variance conditioning "
            "(eps * (1 + 2 * max_degree * unit / (residual + jitter)) <= 0.01)"
        )
    result = min(candidates, key=lambda candidate: candidate.fun)
    variances = np.exp(result.x)
    raw = _raw_from_log_variances(jnp.asarray(result.x))
    nll, beta = profile(jnp.asarray(result.x))
    mean, covariance, factor = mlpe_effect_posterior(
        *kernel_inputs, fixed_effects=beta, raw_variances=raw, **kernel_options
    )
    if not all(np.isfinite(value).all() for value in (nll, beta, mean, covariance, factor)):
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
        tuple(map(float, mean)),
        tuple(map(tuple, np.asarray(covariance))),
        tuple(map(tuple, np.asarray(factor))),
        pairs,
        (role,) * len(pairs),
        -float(nll),
        config,
        int(result.nit),
        str(result.message),
    )
