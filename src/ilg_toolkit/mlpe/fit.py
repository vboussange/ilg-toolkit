"""Float64 standalone calibration, distinct from encoder optimization."""

import math
from dataclasses import dataclass
from functools import partial
from numbers import Real
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from ..data import ObservationPartition, PairwiseObservations, TargetSpec
from ..prediction import Prediction
from .likelihood import MLPEFit, finalize_mlpe_fit, profiled_mlpe_ml_fit


class MLPEError(ValueError):
    """Invalid or numerically unusable full-ML calibration."""


def _raw_from_log_variances(log_variances):
    variances = jnp.exp(log_variances)
    return variances + jnp.log(-jnp.expm1(-variances))


@partial(jax.jit, static_argnames=("n_populations",))
def _optimizer_value_and_gradient(log_variances, distances, targets, left, right, **options):
    def objective(parameters):
        return profiled_mlpe_ml_fit(
            distances,
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
    min_distance_scale: float = 1e-12
    jitter: float = 0.0
    max_iterations: int = 2000

    def __post_init__(self):
        for name in ("variance_floor", "min_distance_scale", "jitter", "max_iterations"):
            if isinstance(getattr(self, name), (bool, np.bool_)):
                raise MLPEError(f"{name} cannot be boolean")
        for name in ("variance_floor", "min_distance_scale"):
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


@final
class MLPEPrediction(eqx.Module):
    """Marginal pair means on declared original and fitted target scales.

    Original-scale values inverse-transform the fitted Gaussian mean. For a
    nonlinear transform this is not the expectation of the original random
    target; no distributional mean correction is implied.

    Attributes:
        prediction: Original-scale points and pair identities. Matrix order is
            distinct query endpoints in first-appearance order.
        model_values: Fitted Gaussian mean vector, shape ``(m,)``, before the
            inverse target transform. Marginal predictive variance is unreported.
    """

    prediction: Prediction
    model_values: jax.Array


@final
class RegionalMLPECalibration(eqx.Module):
    """Complete regional calibration with auditable observation identities."""

    region_name: str
    target: TargetSpec
    population_ids: tuple[str, ...]
    distance_center: float
    distance_scale: float
    intercept: float
    slope: float
    unit_variance: float
    residual_variance: float
    effect_mean: jax.Array = eqx.field(converter=jnp.asarray)
    effect_covariance: jax.Array = eqx.field(converter=jnp.asarray)
    effect_precision_cholesky: jax.Array = eqx.field(converter=jnp.asarray)
    calibration_pairs: tuple[tuple[str, str], ...]
    calibration_roles: tuple[str, ...]
    ml_log_likelihood: float
    config: MLPEConfig
    optimizer_iterations: int
    optimizer_message: str
    converged: bool = True

    @classmethod
    def from_fit(
        cls,
        fit: MLPEFit,
        *,
        region_name: str,
        target: TargetSpec,
        population_ids: tuple[str, ...],
        calibration_pairs: tuple[tuple[str, str], ...],
        calibration_roles: tuple[str, ...],
        config: MLPEConfig,
        optimizer_iterations: int,
        optimizer_message: str,
        converged: bool = True,
    ) -> "RegionalMLPECalibration":
        """Attach target, observation provenance and optimizer diagnostics to a fit."""
        mean, covariance, factor = fit.posterior
        return cls(
            region_name=region_name,
            target=target,
            population_ids=population_ids,
            distance_center=float(fit.distance_center),
            distance_scale=float(fit.distance_scale),
            intercept=float(fit.coefficients[0]),
            slope=float(fit.coefficients[1]),
            unit_variance=float(fit.variances[0]),
            residual_variance=float(fit.variances[1]),
            effect_mean=mean,
            effect_covariance=covariance,
            effect_precision_cholesky=factor,
            calibration_pairs=calibration_pairs,
            calibration_roles=calibration_roles,
            ml_log_likelihood=-float(fit.nll),
            config=config,
            optimizer_iterations=optimizer_iterations,
            optimizer_message=optimizer_message,
            converged=converged,
        )

    def __post_init__(self):
        if not isinstance(self.target, TargetSpec) or self.target.kind != "dissimilarity":
            raise MLPEError("Population MLPE requires a dissimilarity TargetSpec")
        scalars = (
            self.distance_center,
            self.distance_scale,
            self.intercept,
            self.slope,
            self.unit_variance,
            self.residual_variance,
            self.ml_log_likelihood,
        )
        if not all(math.isfinite(value) for value in scalars):
            raise MLPEError("MLPE head cannot retain nonfinite numerical kernel diagnostics")
        if self.distance_scale <= self.config.min_distance_scale or min(
            self.unit_variance, self.residual_variance
        ) < self.config.variance_floor * (1 - 8 * np.finfo(float).eps):
            raise MLPEError(
                "MLPE head requires valid distance scale and positive bounded variances"
            )
        n = len(self.population_ids)
        mean, covariance, factor = (
            self.effect_mean,
            self.effect_covariance,
            self.effect_precision_cholesky,
        )
        if (
            mean.shape != (n,)
            or covariance.shape != (n, n)
            or factor.shape != (n, n)
            or not bool(jnp.isfinite(mean).all())
            or not bool(jnp.isfinite(covariance).all())
            or not bool(jnp.isfinite(factor).all())
        ):
            raise MLPEError("MLPE head population-effect posterior must be finite and aligned")
        if (
            not self.region_name
            or len(set(self.population_ids)) != n
            or len(self.calibration_pairs) != len(self.calibration_roles)
        ):
            raise MLPEError("MLPE head region, sampling-unit identities and provenance must align")

    @jax.enable_x64()
    def predict_marginal(self, distances, pairs) -> MLPEPrediction:
        """Predict labelled query pairs without targets; new effects have mean zero."""
        from .prediction import _query

        distances, pairs, _ = _query(distances, pairs)
        model_values = (
            self.intercept + self.slope * (distances - self.distance_center) / self.distance_scale
        )
        return MLPEPrediction(
            Prediction(
                self.target.inverse(model_values),
                pairs,
                self.region_name,
                tuple(dict.fromkeys(label for pair in pairs for label in pair)),
            ),
            model_values,
        )

    def predict_known_effects(self, distances, pairs):
        """Explicit posterior-effect prediction, with independent priors for unseen units."""
        from .prediction import predict_known_effects

        return predict_known_effects(self, distances, pairs)

    def condition_on_support(self, support_distances, support_observations, *, partition):
        """Update effects using only an explicit support-role partition."""
        from .prediction import condition_on_support

        return condition_on_support(
            self, support_distances, support_observations, partition=partition
        )


@jax.enable_x64()
def calibrate_mlpe(
    distances,
    observations: PairwiseObservations,
    *,
    region_name: str,
    partition: ObservationPartition | None = None,
    config: MLPEConfig | None = None,
) -> RegionalMLPECalibration:
    """Fit full ML from distances aligned to ``observations.observed_pairs``.

    ``partition`` explicitly selects a subset and records its declared data role.
    Missing observations remain absent. This operation never updates an encoder.
    """
    config = MLPEConfig() if config is None else config
    distances = jnp.asarray(distances, dtype=jnp.float64)
    pairs = observations.observed_pairs
    targets = jnp.asarray(observations.observed_values, dtype=jnp.float64)
    if not isinstance(region_name, str) or not region_name:
        raise MLPEError("region_name must be declared")
    if observations.target.kind != "dissimilarity":
        raise MLPEError(
            "Population MLPE supports declared dissimilarities; relatedness needs a model"
        )
    if distances.shape != (len(pairs),) or not bool(jnp.isfinite(distances).all()):
        raise MLPEError("Calibration distances must be finite and aligned with observed_pairs")
    role = "calibration"
    if partition is not None:
        if partition.role in {"support", "query"}:
            raise MLPEError("Calibration cannot consume query or support partitions")
        if partition.region_name != region_name:
            raise MLPEError("Calibration partition region_name does not match")
        selected = set(partition.pairs)
        if not selected.issubset({tuple(sorted(pair)) for pair in pairs}):
            raise MLPEError("Calibration partition selects unobserved pairs")
        selected_rows = tuple(tuple(sorted(pair)) in selected for pair in pairs)
        pairs = tuple(pair for pair, keep in zip(pairs, selected_rows, strict=True) if keep)
        mask = jnp.asarray(selected_rows)
        distances, targets = distances[mask], targets[mask]
        role = partition.role
    targets = observations.target.forward(targets)
    if len(pairs) < 3:
        raise MLPEError("MLPE fitting requires at least three observed pairs")
    center, scale = float(jnp.mean(distances)), float(jnp.std(distances, ddof=1))
    if not math.isfinite(scale) or scale <= config.min_distance_scale:
        raise MLPEError("Calibration distances are constant or ill-scaled; singular GLS design")
    design = jnp.column_stack((jnp.ones(len(distances)), (distances - center) / scale))
    observed = {endpoint for pair in pairs for endpoint in pair}
    population_ids = tuple(label for label in observations.sampling_unit_ids if label in observed)
    lookup = {label: index for index, label in enumerate(population_ids)}
    left = jnp.asarray([lookup[a] for a, b in pairs], dtype=jnp.int32)
    right = jnp.asarray([lookup[b] for a, b in pairs], dtype=jnp.int32)
    if jnp.bincount(jnp.concatenate((left, right)), length=len(population_ids)).max() <= 1:
        raise MLPEError("MLPE variances are unidentifiable: observed pairs share no endpoints")
    ols_beta = jnp.linalg.lstsq(design, targets, rcond=None)[0]
    response_variance = max(
        float(jnp.var(targets - design @ ols_beta, ddof=1)), config.variance_floor * 100
    )
    bounds = [(math.log(config.variance_floor), math.log(max(response_variance * 1e6, 1e4)))] * 2

    # Bounds enforce the standalone variance floor. Decode these bounded log
    # variances through the common softplus interface with no additional floor;
    # this preserves the exact lower bound without subtracting nearly equal
    # values or changing the joint trainer's raw-parameter policy.
    kernel_inputs = distances, targets, left, right
    kernel_options = dict(
        n_populations=len(population_ids),
        distance_center=center,
        distance_scale=scale,
        min_distance_scale=config.min_distance_scale,
        variance_floor=0.0,
        jitter=config.jitter,
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
            "; check distance variation and variance conditioning "
            "(eps * (1 + 2 * max_degree * unit / (residual + jitter)) <= 0.01)"
        )
    result = min(candidates, key=lambda candidate: candidate.fun)
    raw = _raw_from_log_variances(jnp.asarray(result.x))
    fit = finalize_mlpe_fit(*kernel_inputs, raw_variances=raw, **kernel_options)
    # Retain the standalone optimizer's exact bounded-log variances. The
    # inverse-softplus round trip can fall a few ulps below its declared floor.
    fit = fit._replace(variances=jnp.exp(jnp.asarray(result.x)))
    if not all(bool(jnp.isfinite(value).all()) for value in jax.tree.leaves(fit)):
        raise MLPEError("Fitted population-effect posterior is nonfinite")
    return RegionalMLPECalibration.from_fit(
        fit,
        region_name=region_name,
        target=observations.target,
        population_ids=population_ids,
        calibration_pairs=pairs,
        calibration_roles=(role,) * len(pairs),
        config=config,
        optimizer_iterations=int(result.nit),
        optimizer_message=str(result.message),
    )
