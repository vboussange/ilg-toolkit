"""Maximum-likelihood population effects (MLPE) calibration of one region–species group."""

import math
from functools import partial
from typing import final, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from ..data import PairwiseObservations
from .likelihood import _finalize_mlpe_fit, _MLPEFit, _profiled_mlpe_ml_fit


if TYPE_CHECKING:
    from ..losses import MLPELoss


class MLPEError(ValueError):
    """An error raised for an invalid or numerically unusable MLPE calibration."""


@eqx.filter_jit
def _raw_from_log_variances(log_variances):
    """Returns the raw variance parameters whose softplus equals ``exp(log_variances)``."""
    variances = jnp.exp(log_variances)
    return variances + jnp.log(-jnp.expm1(-variances))


@partial(jax.jit, static_argnames=("n_populations",))
def _optimizer_value_and_gradient(log_variances, distances, targets, left, right, **options):
    """Returns the profiled negative log likelihood and its gradient in log variances."""

    def objective(parameters):
        return _profiled_mlpe_ml_fit(
            distances,
            targets,
            left,
            right,
            raw_variances=_raw_from_log_variances(parameters),
            **options,
        )[0]

    return jax.value_and_grad(objective)(log_variances)


@final
class MLPECalibration(eqx.Module):
    """One region–species group's MLPE calibration on the model scale.

    An [`MLPECalibrationHead`][ilg_toolkit.MLPECalibrationHead] holds one MLPE
    calibration per region–species group and owns their target contract and
    region names. Model-scale targets are
    ``intercept + slope * (distance - distance_center) / distance_scale`` plus the
    two endpoints' population effects and a residual.

    Attributes:
        population_ids: Sampling units of the population effects, in the order of
            ``effect_mean`` and ``effect_covariance``: those of the calibration
            observations from [`fit_calibration_head`][ilg_toolkit.fit_calibration_head],
            or every sampling unit of the region from training.
        distance_center: Mean of the calibration's ecological distances.
        distance_scale: Sample standard deviation (ddof=1) of the calibration's
            ecological distances.
        intercept: Signed intercept on the target's model scale.
        slope: Signed slope per standardized ecological distance, on the target's
            model scale.
        unit_variance: Variance of the population effects.
        residual_variance: Residual variance of each pair; the jitter of ``loss``
            is added to it in likelihoods and support conditioning.
        effect_mean: Posterior mean of the population effects.
        effect_covariance: Posterior covariance of the population effects.
        calibration_pairs: Sampling-unit pairs the posterior conditions on.
            Support observations are disjoint from them.
        ml_log_likelihood: Marginal log likelihood of the calibration observations:
            its maximum from [`fit_calibration_head`][ilg_toolkit.fit_calibration_head],
            or its value at the trained variance components from training.
        loss: MLPE loss whose settings the calibration was fitted with.
        optimizer_iterations: Iterations of the variance optimizer of
            [`fit_calibration_head`][ilg_toolkit.fit_calibration_head], or the
            training step of a calibration that training produced.
        optimizer_message: Final message of the variance optimizer, or training's
            description of its step budget.
        converged: Whether the variance optimizer converged; ``False`` for a
            calibration that training produced.

    Raises:
        MLPEError: A fitted value is nonfinite, the distance scale or a variance
            component is below the limits of ``loss``, or the posterior does not
            align with unique ``population_ids``.
    """

    population_ids: tuple[str, ...]
    distance_center: float
    distance_scale: float
    intercept: float
    slope: float
    unit_variance: float
    residual_variance: float
    effect_mean: jax.Array = eqx.field(converter=jnp.asarray)
    effect_covariance: jax.Array = eqx.field(converter=jnp.asarray)
    calibration_pairs: tuple[tuple[str, str], ...]
    ml_log_likelihood: float
    loss: "MLPELoss"
    optimizer_iterations: int
    optimizer_message: str
    converged: bool = True

    def __check_init__(self):
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
        if (
            self.effect_mean.shape != (n,)
            or self.effect_covariance.shape != (n, n)
            or not np.isfinite(self.effect_mean).all()
            or not np.isfinite(self.effect_covariance).all()
        ):
            raise MLPEError(
                "MLPE calibration population-effect posterior must be finite and aligned"
            )
        if len(set(self.population_ids)) != n:
            raise MLPEError("MLPE calibration sampling-unit identities must be unique")


def _calibration_from_fit(
    fit: _MLPEFit,
    *,
    population_ids: tuple[str, ...],
    calibration_pairs: tuple[tuple[str, str], ...],
    loss: "MLPELoss",
    optimizer_iterations: int,
    optimizer_message: str,
    converged: bool = True,
) -> MLPECalibration:
    """Builds an MLPE calibration from a finalized fit and its bookkeeping."""
    mean, covariance = fit.posterior
    return MLPECalibration(
        population_ids=population_ids,
        distance_center=float(fit.distance_center),
        distance_scale=float(fit.distance_scale),
        intercept=float(fit.coefficients[0]),
        slope=float(fit.coefficients[1]),
        unit_variance=float(fit.variances[0]),
        residual_variance=float(fit.variances[1]),
        effect_mean=mean,
        effect_covariance=covariance,
        calibration_pairs=calibration_pairs,
        ml_log_likelihood=-float(fit.nll),
        loss=loss,
        optimizer_iterations=optimizer_iterations,
        optimizer_message=optimizer_message,
        converged=converged,
    )


def _fit_mlpe(
    distances,
    observations: PairwiseObservations,
    *,
    loss: "MLPELoss",
    max_iterations: int,
) -> MLPECalibration:
    """Fits an MLPE calibration to distances aligned with ``observations.observed_pairs``.

    The observations have passed ``MLPELoss._prepare``: a dissimilarity target, at
    least three observed pairs, and pairs sharing sampling units. Coefficients are
    profiled by generalized least squares, and both variance components are
    estimated by deterministic multistart L-BFGS-B, each start bounded by
    ``max_iterations``; the finite converged start with the highest marginal
    likelihood is retained. ``max_iterations`` is a positive integer.

    Covariance algebra is population-sized, with no dense pair-by-pair covariance
    matrix. For population variance u, residual variance e, jitter j and maximum
    observed population degree dmax, supported precision requires
    ``eps * (1 + 2 * dmax * u / (e + j)) <= 0.01``. Failed factorizations,
    nonfinite inputs, constant distances and unsupported variance ratios raise
    ``MLPEError``.
    """
    # The optimizer's starts and bounds are prepared in NumPy on the host, compiling
    # no XLA program for new shapes; the compiled kernels evaluate the likelihood.
    distances = np.asarray(distances, dtype=np.float64)
    pairs = observations.observed_pairs
    targets = np.asarray(
        observations.target.forward(jnp.asarray(observations.observed_values, dtype=jnp.float64))
    )
    if distances.shape != (len(pairs),) or not np.isfinite(distances).all():
        raise MLPEError("Calibration distances must be finite and aligned with observed_pairs")
    center, scale = float(np.mean(distances)), float(np.std(distances, ddof=1))
    if not math.isfinite(scale) or scale <= loss.min_distance_scale:
        raise MLPEError("Calibration distances are constant or ill-scaled; singular GLS design")
    design = np.column_stack((np.ones(len(distances)), (distances - center) / scale))
    observed = {endpoint for pair in pairs for endpoint in pair}
    population_ids = tuple(label for label in observations.sampling_unit_ids if label in observed)
    lookup = {label: index for index, label in enumerate(population_ids)}
    left = np.asarray([lookup[a] for a, b in pairs], dtype=np.int32)
    right = np.asarray([lookup[b] for a, b in pairs], dtype=np.int32)
    ols_beta = np.linalg.lstsq(design, targets, rcond=None)[0]
    response_variance = max(
        float(np.var(targets - design @ ols_beta, ddof=1)), loss.variance_floor * 100
    )
    bounds = [(math.log(loss.variance_floor), math.log(max(response_variance * 1e6, 1e4)))] * 2

    # Decode the fitted log variances through softplus while retaining the
    # variance floor and numerical precision near that floor.
    kernel_inputs = tuple(map(jnp.asarray, (distances, targets, left, right)))
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
    fit = _finalize_mlpe_fit(*kernel_inputs, raw_variances=raw, **kernel_options)
    # Retain the fitted variances: rounding during inverse softplus can place
    # reconstructed values slightly below the configured variance floor.
    fit = fit._replace(variances=np.exp(result.x))
    if not all(np.isfinite(value).all() for value in jax.tree.leaves(fit)):
        raise MLPEError("Fitted population-effect posterior is nonfinite")
    return _calibration_from_fit(
        fit,
        population_ids=population_ids,
        calibration_pairs=pairs,
        loss=loss,
        optimizer_iterations=int(result.nit),
        optimizer_message=str(result.message),
    )
