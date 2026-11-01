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
from .covariance import _max_log_variance_ratio, _MAX_ROUNDOFF_BOUND
from .likelihood import _finalize_mlpe_fit, _MLPEFit, _profiled_mlpe_ml_fit


if TYPE_CHECKING:
    from ..losses import MLPELoss


# Largest projected gradient of the negative log likelihood that counts as a local optimum,
# per observed pair and overall. The overall term clears rounding on a few pairs.
_GRADIENT_TOLERANCE_PER_PAIR = 1e-5
_GRADIENT_TOLERANCE = 1e-2
# Fractions of the response variance in the population effects at the optimization's starts,
# with variance ratios u / e from 0.005 to 2500. The last two reach a mode at a large ratio,
# where a large u cancels an additive term of the distances.
_START_FRACTIONS = (0.01, 0.1, 0.25, 0.5, 0.8, 0.998, 0.9998)
# Difference in negative log likelihood within which runs have reached the same optimum.
_SAME_OPTIMUM_TOLERANCE = 1e-3
# Exponent of the smooth maximum of the optimizer's parameters: at a tenth of its larger
# argument, the smaller one moves the maximum by a relative 1e-9.
_SMOOTH_MAXIMUM_EXPONENT = 8


class MLPEError(ValueError):
    """An error raised for an invalid or numerically unusable MLPE calibration."""


@eqx.filter_jit
def _raw_from_log_variances(log_variances):
    """Returns the raw variance parameters whose softplus equals ``exp(log_variances)``."""
    variances = jnp.exp(log_variances)
    return variances + jnp.log(-jnp.expm1(-variances))


def _log_limit_share(parameters, *, max_log_ratio, jitter):
    """Returns the log of ``u / exp(max_log_ratio)`` over ``z + j`` at ``parameters``."""
    log_unit, log_remainder = parameters
    return log_unit - max_log_ratio - jnp.log(jnp.exp(log_remainder) + jitter)


def _log_variances(parameters, *, max_log_ratio, jitter):
    """Returns the log unit and residual variances of the optimizer's parameters.

    The parameters are ``log u`` and ``log z``. ``e + j`` is a smooth maximum of ``z + j``
    and ``u / exp(max_log_ratio)``, so it follows ``z + j`` away from the variance ratio
    limit, and ``u / (e + j)`` stays below the limit at every ``z``.
    """
    log_unit, log_remainder = parameters
    remainder = jnp.exp(log_remainder)
    excess = jnp.expm1(
        jax.nn.softplus(
            _SMOOTH_MAXIMUM_EXPONENT
            * _log_limit_share(parameters, max_log_ratio=max_log_ratio, jitter=jitter)
        )
        / _SMOOTH_MAXIMUM_EXPONENT
    )
    return jnp.stack((log_unit, jnp.log(remainder + (remainder + jitter) * excess)))


@partial(jax.jit, static_argnames=("n_populations",))
def _optimizer_value_and_gradient(
    parameters, distances, targets, left, right, *, max_log_ratio, jitter, **options
):
    """Returns the profiled negative log likelihood and its gradient in ``parameters``."""

    def objective(parameters):
        log_variances = _log_variances(parameters, max_log_ratio=max_log_ratio, jitter=jitter)
        return _profiled_mlpe_ml_fit(
            distances,
            targets,
            left,
            right,
            raw_variances=_raw_from_log_variances(log_variances),
            jitter=jitter,
            **options,
        )[0]

    return jax.value_and_grad(objective)(parameters)


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
            its highest local optimum over the optimization starts of
            [`fit_calibration_head`][ilg_toolkit.fit_calibration_head], or its value
            at the trained variance components from training.
        loss: MLPE loss whose settings the calibration was fitted with.
        optimizer_iterations: Iterations of the variance optimizer of
            [`fit_calibration_head`][ilg_toolkit.fit_calibration_head], or the
            training step of a calibration that training produced.
        optimizer_message: Final message of the variance optimizer, or training's
            description of its step budget.
        converged: Whether the variance optimizer reached a local optimum of the
            marginal likelihood; ``False`` for a calibration from training or from a
            saved model.

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
    ``max_iterations``, and the start with the highest marginal likelihood is retained.
    ``max_iterations`` is a positive integer.

    Covariance algebra is population-sized, with no dense pair-by-pair covariance
    matrix. For population variance u, residual variance e, jitter j and maximum
    observed population degree dmax, supported precision requires
    ``eps * (1 + 2 * dmax * u / (e + j)) <= 0.01``. L-BFGS-B searches ``log u`` and
    ``log z``, with ``e + j`` a smooth maximum of ``z + j`` and ``u`` over the limit of
    that bound, so no trial is rejected for its ratio, and the variance floor bounds both
    ``u`` and ``z``. ``u`` has no upper bound, so the first trial has unit length, and it
    lengthens only while the likelihood improves. A trial of an extreme value can still
    reduce L-BFGS-B's step to nothing, and at the limit ``e`` no longer depends on ``z``.
    Either stops L-BFGS-B away from a local optimum, so it restarts with ``z`` at the
    fitted ``e`` while it makes progress, within ``max_iterations``. Failed factorizations,
    nonfinite inputs, constant distances, and a retained start with ``u / (z + j)`` above
    half the limit or with a projected gradient above tolerance raise ``MLPEError``.
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
    log_floor = math.log(loss.variance_floor)
    log_upper = math.log(max(response_variance * 1e6, 1e4))

    # Decode the fitted log variances through softplus while retaining the
    # variance floor and numerical precision near that floor.
    kernel_inputs = tuple(map(jnp.asarray, (distances, targets, left, right)))
    max_log_ratio = _max_log_variance_ratio(
        int(np.bincount(np.concatenate((left, right))).max()),
        float(jnp.finfo(kernel_inputs[0].dtype).eps),
    )
    kernel_options = dict(
        n_populations=len(population_ids),
        distance_center=center,
        distance_scale=scale,
        min_distance_scale=loss.min_distance_scale,
        variance_floor=0.0,
        jitter=loss.jitter,
    )

    def objective(parameters):
        value, gradient = _optimizer_value_and_gradient(
            jnp.asarray(parameters), *kernel_inputs, max_log_ratio=max_log_ratio, **kernel_options
        )
        value, gradient = float(value), np.asarray(gradient)
        if not math.isfinite(value) or not np.isfinite(gradient).all():
            return float("inf"), np.zeros_like(parameters)
        return value, gradient

    def stationary(result):
        projected_gradient = result.x - np.clip(
            result.x - result.jac, log_floor, [np.inf, log_upper]
        )
        tolerance = _GRADIENT_TOLERANCE + _GRADIENT_TOLERANCE_PER_PAIR * len(distances)
        return np.abs(projected_gradient).max() <= tolerance

    def at_ratio_limit(result):
        share = _log_limit_share(result.x, max_log_ratio=max_log_ratio, jitter=loss.jitter)
        return share > -math.log(2)

    def descend(parameters):
        iterations, value = 0, math.inf
        while True:
            result = minimize(
                objective,
                parameters,
                jac=True,
                method="L-BFGS-B",
                bounds=[(log_floor, None), (log_floor, log_upper)],
                options={"maxiter": max_iterations - iterations, "ftol": 1e-12, "gtol": 1e-9},
            )
            iterations += result.nit
            if (
                (stationary(result) and not at_ratio_limit(result))
                or iterations >= max_iterations
                or result.fun >= value - _SAME_OPTIMUM_TOLERANCE
            ):
                result.nit = iterations
                return result
            parameters = np.asarray(
                _log_variances(
                    jnp.asarray(result.x), max_log_ratio=max_log_ratio, jitter=loss.jitter
                )
            )
            value = result.fun

    results = []
    for fraction in _START_FRACTIONS:
        unit = max(response_variance * fraction / 2, loss.variance_floor * 10)
        residual = max(response_variance * (1 - fraction), loss.variance_floor * 10)
        results.append(descend(np.log([unit, residual])))
    best = min(candidate.fun for candidate in results)
    if not math.isfinite(best):
        raise MLPEError(
            "No deterministic MLPE optimization start reached a finite likelihood"
            "; check distance variation"
        )
    # Of the starts at the same optimum, keep the one that L-BFGS-B stopped cleanly.
    result = min(
        results,
        key=lambda candidate: (
            candidate.fun - best > _SAME_OPTIMUM_TOLERANCE,
            not stationary(candidate),
            not candidate.success,
            candidate.fun,
        ),
    )
    if at_ratio_limit(result):
        raise MLPEError(
            "The highest MLPE likelihood lies at the variance ratio limit "
            f"(eps * (1 + 2 * max_degree * unit / (residual + jitter)) <= {_MAX_ROUNDOFF_BOUND})"
        )
    if not stationary(result):
        raise MLPEError(
            "The MLPE optimization start with the highest likelihood stopped away from a "
            f"local optimum after {result.nit} iterations: {result.message}"
        )
    log_variances = _log_variances(
        jnp.asarray(result.x), max_log_ratio=max_log_ratio, jitter=loss.jitter
    )
    fit = _finalize_mlpe_fit(
        *kernel_inputs, raw_variances=_raw_from_log_variances(log_variances), **kernel_options
    )
    # Retain the fitted variances: rounding during inverse softplus can place
    # reconstructed values slightly below the configured variance floor.
    fit = fit._replace(variances=np.exp(log_variances))
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
