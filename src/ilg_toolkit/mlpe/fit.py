"""Standalone maximum-likelihood population effects (MLPE) calibration using float64."""

import math
from functools import partial
from typing import final, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from ..data import PairwiseObservations, TargetSpec
from .likelihood import finalize_mlpe_fit, MLPEFit, profiled_mlpe_ml_fit


if TYPE_CHECKING:
    from ..losses import MLPELoss


class MLPEError(ValueError):
    """An invalid or numerically unusable maximum-likelihood calibration."""


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


@final
class RegionalMLPECalibration(eqx.Module):
    """A fitted MLPE relationship, population posterior, and calibration-pair identities.

    ``distance_center`` and ``distance_scale`` standardize the distances used in calibration
    (sample standard deviation, ddof=1). ``intercept`` and ``slope`` are signed
    coefficients on the target's model scale. ``unit_variance`` and
    ``residual_variance`` are the two fitted variance components; configured jitter
    is added to the residual component in likelihoods and conditioning.

    ``population_ids`` orders ``effect_mean`` and ``effect_covariance``. The saved
    posterior applies to ``calibration_pairs``; support observations must be
    disjoint from those measurements. ``ml_log_likelihood`` is the marginal
    likelihood of the calibration observations, not a score on untouched targets.
    ``converged``, ``optimizer_iterations``, and ``optimizer_message`` describe the
    standalone variance optimizer. Fixed-budget joint training makes no convergence
    claim. Inspect both the likelihood and signed slope when interpreting association.
    """

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
    ml_log_likelihood: float
    loss: "MLPELoss"
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
        loss: "MLPELoss",
        optimizer_iterations: int,
        optimizer_message: str,
        converged: bool = True,
    ) -> "RegionalMLPECalibration":
        """MLPE calibration with target, calibration pairs, and optimizer diagnostics."""
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
            ml_log_likelihood=-float(fit.nll),
            loss=loss,
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
            raise MLPEError("MLPE calibration cannot retain nonfinite fit diagnostics")
        if self.distance_scale <= self.loss.min_distance_scale or min(
            self.unit_variance, self.residual_variance
        ) < self.loss.variance_floor * (1 - 8 * np.finfo(float).eps):
            raise MLPEError(
                "MLPE calibration requires a valid distance scale and valid variance parameters"
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
            raise MLPEError(
                "MLPE calibration population-effect posterior must be finite and aligned"
            )
        if not self.region_name or len(set(self.population_ids)) != n:
            raise MLPEError("MLPE calibration region and sampling-unit identities must align")


@jax.enable_x64()
def calibrate_mlpe(
    distances,
    observations: PairwiseObservations,
    *,
    region_name: str,
    loss: "MLPELoss | None" = None,
    max_iterations: int = 2000,
) -> RegionalMLPECalibration:
    """MLPE calibration from ecological distances aligned to ``observations.observed_pairs``.

    ``loss`` declares immutable numerical settings; ``max_iterations`` bounds
    each deterministic multistart optimization.
    Coefficients, variance parameters, and population effects are fitted with
    the distance model fixed.

    Args:
        distances (numpy.typing.ArrayLike):
            Finite vector with one value per ``observations.observed_pairs`` in exactly
            that order. Geographic distances are allowed for a matching MLPE baseline.
        observations: Population dissimilarities and their target transformation. At
            least three
            observed pairs must share populations, and the intercept/distance design
            must be nonsingular. Constant distances are not an intercept-only null.
        region_name: Nonempty identity retained by the calibration.
        loss: Shared ``MLPELoss`` settings, including variance floors, distance-scale
            threshold, and fixed jitter. Omitted settings use ``MLPELoss()``.
        max_iterations: Positive iteration budget for each of five deterministic
            L-BFGS-B starts.
            The finite converged start with the highest marginal likelihood is retained.

    Returns:
        RegionalMLPECalibration:
            Signed coefficients, both variance estimates, distance moments, population
            posterior, pair identities, and optimizer diagnostics.

    Raises:
        MLPEError:
            For invalid, unidentifiable, numerically unsupported, or unconverged fits.

    Notes:
        This call uses float64 internally. Joint MLPE training requires float64 enabled
        beforehand. Population-sized covariance algebra avoids a dense pair-by-pair
        covariance matrix. For population variance u, residual variance e, jitter j,
        and maximum observed population degree dmax, supported precision requires
        ``eps * (1 + 2 * dmax * u / (e + j)) <= 0.01``. Failed factorizations,
        nonfinite inputs, and unsupported variance ratios fail explicitly.

    Examples:
        To compare geographic and learned distances, use exactly the same observations,
        target transformation, numerical settings, and optimizer budget for both fits:

        >>> loss = MLPELoss(jitter=0.0)
        >>> baseline = calibrate_mlpe(geographic_distances, observations,
        ...                           region_name=region.name, loss=loss,
        ...                           max_iterations=2000)
        >>> learned = calibrate(region, observations, frozen_distance_model,
        ...                     loss=loss, max_iterations=2000)
        >>> landscape = learned.calibration.for_region(region.name)
        >>> improvement = landscape.ml_log_likelihood - baseline.ml_log_likelihood

        ``geographic_distances`` must align with ``observations.observed_pairs``.
        These observations fit and score both calibrations; the difference describes
        explanatory association. The [MLPE workflow](../examples/mlpe_workflow.md)
        demonstrates candidate selection and repeated disjoint population splits.
    """
    from ..losses import MLPELoss

    loss = MLPELoss() if loss is None else loss
    if not isinstance(loss, MLPELoss):
        raise MLPEError("calibrate_mlpe requires MLPELoss settings")
    if type(max_iterations) is not int or max_iterations < 1:
        raise MLPEError("max_iterations must be a positive integer")
    distances = jnp.asarray(distances, dtype=jnp.float64)
    pairs = observations.observed_pairs
    targets = jnp.asarray(observations.observed_values, dtype=jnp.float64)
    if not isinstance(region_name, str) or not region_name:
        raise MLPEError("region_name must be nonempty")
    if observations.target.kind != "dissimilarity":
        raise MLPEError("Population MLPE supports dissimilarities; relatedness needs a model")
    if distances.shape != (len(pairs),) or not bool(jnp.isfinite(distances).all()):
        raise MLPEError("Calibration distances must be finite and aligned with observed_pairs")
    targets = observations.target.forward(targets)
    if len(pairs) < 3:
        raise MLPEError("MLPE fitting requires at least three observed pairs")
    center, scale = float(jnp.mean(distances)), float(jnp.std(distances, ddof=1))
    if not math.isfinite(scale) or scale <= loss.min_distance_scale:
        raise MLPEError("Calibration distances are constant or ill-scaled; singular GLS design")
    design = jnp.column_stack((jnp.ones(len(distances)), (distances - center) / scale))
    observed = {endpoint for pair in pairs for endpoint in pair}
    population_ids = tuple(label for label in observations.sampling_unit_ids if label in observed)
    lookup = {label: index for index, label in enumerate(population_ids)}
    left = jnp.asarray([lookup[a] for a, b in pairs], dtype=jnp.int32)
    right = jnp.asarray([lookup[b] for a, b in pairs], dtype=jnp.int32)
    if jnp.bincount(jnp.concatenate((left, right)), length=len(population_ids)).max() <= 1:
        raise MLPEError("MLPE variances are unidentifiable: observed pairs share no sampling units")
    ols_beta = jnp.linalg.lstsq(design, targets, rcond=None)[0]
    response_variance = max(
        float(jnp.var(targets - design @ ols_beta, ddof=1)), loss.variance_floor * 100
    )
    bounds = [(math.log(loss.variance_floor), math.log(max(response_variance * 1e6, 1e4)))] * 2

    # Decode the fitted log variances through softplus while retaining the
    # standalone variance floor and numerical precision near that floor.
    kernel_inputs = distances, targets, left, right
    kernel_options = dict(
        n_populations=len(population_ids),
        distance_center=center,
        distance_scale=scale,
        min_distance_scale=loss.min_distance_scale,
        variance_floor=0.0,
        jitter=loss.jitter,
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
                max(response_variance * fraction / 2, loss.variance_floor * 10),
                max(response_variance * (1 - fraction), loss.variance_floor * 10),
            ]
        )
        result = minimize(
            objective,
            start,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iterations, "ftol": 1e-12, "gtol": 1e-9},
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
    # Retain the fitted variances: rounding during inverse softplus can place
    # reconstructed values slightly below the configured variance floor.
    fit = fit._replace(variances=jnp.exp(jnp.asarray(result.x)))
    if not all(bool(jnp.isfinite(value).all()) for value in jax.tree.leaves(fit)):
        raise MLPEError("Fitted population-effect posterior is nonfinite")
    return RegionalMLPECalibration.from_fit(
        fit,
        region_name=region_name,
        target=observations.target,
        population_ids=population_ids,
        calibration_pairs=pairs,
        loss=loss,
        optimizer_iterations=int(result.nit),
        optimizer_message=str(result.message),
    )
