"""Whole-group statistical criteria and their immutable numerical settings."""

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any, final, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

from ._training_utils import RegionSpeciesKey
from .data import PairwiseObservations, RegionBatch, TargetSpec


if TYPE_CHECKING:
    from .calibration import AbstractCalibrationModule


@final
class _LossContext(eqx.Module):
    targets: jax.Array
    pairs: tuple[jax.Array, jax.Array]
    population_count: int = eqx.field(static=True)


class AbstractLoss(ABC):
    """A statistical criterion for a complete region–species group.

    Each loss owns its reduction, statistical parameters, and fitted calibration.
    Training combines the resulting scalars with equal species influence and equal
    region influence within species. Observation target transformations remain
    separate from the statistical criterion.
    """

    @abstractmethod
    def _prepare(self, region, observations, *, training: bool) -> _LossContext: ...

    @abstractmethod
    def _initialize(self, groups: Mapping[RegionSpeciesKey, _LossContext]) -> Any: ...

    @abstractmethod
    def _group_state(self, state, identity: RegionSpeciesKey) -> Any: ...

    @abstractmethod
    def _value(self, distances, context: _LossContext, state, fixed=None) -> jax.Array: ...

    @abstractmethod
    def _finalize(self, distances, context, state, *, region, observations, step) -> Any: ...

    @abstractmethod
    def _fixed_parameters(self, fitted) -> Any: ...

    @abstractmethod
    def _calibration(self, target: TargetSpec, fitted: Mapping) -> "AbstractCalibrationModule": ...

    @abstractmethod
    def _fit_group(self, distances, observations, *, region_name, max_iterations) -> Any: ...

    @abstractmethod
    def _state_metadata(self, state) -> Any: ...

    @abstractmethod
    def _state_template(self, metadata) -> Any: ...


def _context(region: RegionBatch, observations: PairwiseObservations) -> _LossContext:
    pairs, values = observations.aligned_pairs(region)
    return _LossContext(observations.target.forward(values), pairs, len(region.sampling_unit_ids))


@final
@dataclass(frozen=True)
class Log1pLoss(AbstractLoss):
    r"""Log1p objective: mean squared difference of log1p ecological distances and targets.

    For a region–species group with observed pairs $\mathcal O$, ecological
    distances $s_{ij}$, and model-scale targets $t_{ij}=T(y_{ij})$ under the
    target transform $T$,

    $$L=\frac{1}{|\mathcal O|}\sum_{(i,j)\in\mathcal O}
    \left[\log(1+s_{ij})-\log(1+t_{ij})\right]^2.$$

    Training averages $L$ over regions within each species, then over species, so
    regions with more pairs do not dominate. The log transforms reduce the leverage
    of the largest genetic dissimilarities; the log1p comparison and the target
    transform are separate choices. The target must describe nonnegative
    dissimilarities.

    The fitted direct calibration returns $T^{-1}(s)$, with no coefficients or
    variance parameters, so this loss identifies an absolute connectivity scale.
    It has no statistical parameters or standalone optimization budget.

    Examples:
        >>> config = TrainingConfig(loss=Log1pLoss(), steps=100)
        >>> state = train(region, observations, distance_model=distance_model, config=config)
        >>> fitted = calibrate(region, observations, state.selected_model.distance_model,
        ...                    loss=Log1pLoss())
    """

    def _prepare(self, region, observations, *, training):
        if observations.target.kind != "dissimilarity" or jnp.any(observations.observed_values < 0):
            raise ValueError("Log1pLoss requires nonnegative dissimilarities")
        return _context(region, observations)

    def _initialize(self, groups):
        return None

    def _group_state(self, state, identity):
        return None

    def _value(self, distances, context, state, fixed=None):
        return jnp.mean(jnp.square(jnp.log1p(distances) - jnp.log1p(context.targets)))

    def _finalize(self, distances, context, state, *, region, observations, step):
        return None

    def _fixed_parameters(self, fitted):
        return None

    def _calibration(self, target, fitted):
        from .calibration import DirectCalibrationModule

        return DirectCalibrationModule(target)

    def _fit_group(self, distances, observations, *, region_name, max_iterations):
        if max_iterations is not None:
            raise ValueError("Log1pLoss calibration has no iteration budget")
        return None

    def _state_metadata(self, state):
        return None

    def _state_template(self, metadata):
        if metadata is not None:
            raise ValueError("Log1pLoss checkpoints have no statistical state")
        return None


@final
class _MLPEGroupState(eqx.Module):
    raw_variances: jax.Array


@final
class _MLPEState(eqx.Module):
    groups: dict[RegionSpeciesKey, _MLPEGroupState]


@final
class _MLPEFixedParameters(eqx.Module):
    """Training-fitted coefficients and distance moments held fixed by validation."""

    coefficients: jax.Array
    distance_center: jax.Array
    distance_scale: jax.Array


@final
@dataclass(frozen=True)
class MLPELoss(AbstractLoss):
    r"""Marginal maximum-likelihood population effects (MLPE) per observed pair.

    MLPE accounts for dependence between pairwise genetic observations sharing a
    population. With sample-standardized ecological distances
    $x_{ij}=(s_{ij}-c)/a$, model-scale targets follow

    $$t_{ij}=\alpha+\beta x_{ij}+b_i+b_j+\epsilon_{ij},\qquad
    b_i\sim N(0,u),\quad\epsilon_{ij}\sim N(0,e+j).$$

    The intercept $\alpha$ and slope $\beta$ are signed; $u$ and $e$ are positive
    population and residual variances, and $j$ is the fixed ``jitter``. A pair has
    variance $2u+e+j$, and distinct pairs sharing a population have covariance $u$.
    With incidence matrix $Z$ and design $X=[\mathbf 1, x]$, the covariance is
    $V=(e+j)I+uZZ^\top$. Generalized least squares profiles the coefficients, and
    the loss is the marginal negative log likelihood per observed pair,

    $$\ell=\frac{1}{2m}\left\{m\log(2\pi)+\log\det V+
    (t-X\hat\theta)^\top V^{-1}(t-X\hat\theta)\right\},$$

    without restricted-likelihood correction. Coefficients, variance components,
    distance moments, and population-effect posteriors are estimated separately for
    each region–species group (group-specific calibration), while distance-model
    parameters are shared across groups. Standardization makes the loss invariant
    to the scale of ecological distances, so MLPE does not identify an absolute
    connectivity scale. MLPE requires float64, population sampling units, and
    dissimilarity targets. See the [advanced MLPE API](mlpe.md) for
    identifiability, precision, diagnostics, and the standalone array-distance fitter.

    Args:
        variance_floor:
            Positive lower limit for both variance components.
        min_distance_scale: Minimum sample standard deviation of ecological distances.
        jitter: Fixed residual variance addition in [0, 1e-6].

    Examples:
        >>> loss = MLPELoss(jitter=0.0)
        >>> state = train(region, observations, distance_model=distance_model,
        ...               config=TrainingConfig(loss=loss, steps=500))
        >>> fitted = calibrate(new_region, new_observations,
        ...                    state.selected_model.distance_model,
        ...                    loss=loss, max_iterations=2000)

        Built-in training validation uses training-fitted coefficients, distance moments,
        and variances. ``train`` updates distance and variance parameters with a fixed
        Adam budget, and zero training steps only initializes the variances. Fresh
        calibration keeps the distance model fixed and estimates both variance
        components with deterministic multistart optimization.
    """

    variance_floor: float = 1e-10
    min_distance_scale: float = 1e-12
    jitter: float = 0.0

    def __post_init__(self):
        for name in ("variance_floor", "min_distance_scale", "jitter"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite and numeric")
            value = float(value)
            if name == "jitter":
                if not 0 <= value <= 1e-6:
                    raise ValueError("jitter must be in [0, 1e-6]")
            elif value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, float(value))

    def _prepare(self, region, observations, *, training):
        if not jax.config.read("jax_enable_x64"):
            raise RuntimeError("MLPE requires float64: enable JAX_ENABLE_X64 or jax.enable_x64()")
        if observations.target.kind != "dissimilarity":
            raise ValueError("Population MLPE requires dissimilarities")
        if any(kind != "population" for kind in region.sampling_unit_kinds):
            raise ValueError("Population MLPE does not support individual sampling units")
        context = _context(region, observations)
        if training:
            if len(context.targets) < 3:
                raise ValueError("MLPE training requires at least three observed pairs")
            degree = jnp.bincount(jnp.concatenate(context.pairs), length=context.population_count)
            if degree.max() <= 1:
                raise ValueError("MLPE variances are unidentifiable: pairs share no sampling units")
        return eqx.tree_at(lambda c: c.targets, context, context.targets.astype(jnp.float64))

    def _initialize(self, groups):
        values = {}
        for identity, context in groups.items():
            variance = jnp.maximum(jnp.var(context.targets, ddof=1), self.variance_floor * 100)
            variances = jnp.maximum(variance * jnp.asarray([0.25, 0.5]), self.variance_floor * 10)
            positive = variances - self.variance_floor
            values[identity] = _MLPEGroupState(positive + jnp.log(-jnp.expm1(-positive)))
        return _MLPEState(values)

    def _group_state(self, state, identity):
        if identity not in state.groups:
            raise ValueError(
                "Validation requires a training calibration for its region–species group"
            )
        return state.groups[identity]

    def _options(self, context, state):
        return dict(
            n_populations=context.population_count,
            raw_variances=state.raw_variances,
            variance_floor=self.variance_floor,
            jitter=self.jitter,
            min_distance_scale=self.min_distance_scale,
        )

    def _value(self, distances, context, state, fixed=None):
        from .mlpe.likelihood import mlpe_ml_negative_log_likelihood, profiled_mlpe_ml_fit

        distances = distances.astype(jnp.float64)
        options = self._options(context, state)
        if fixed is None:
            value = profiled_mlpe_ml_fit(distances, context.targets, *context.pairs, **options)[0]
        else:
            value = mlpe_ml_negative_log_likelihood(
                distances,
                context.targets,
                *context.pairs,
                fixed_effects=fixed.coefficients,
                distance_center=fixed.distance_center,
                distance_scale=fixed.distance_scale,
                **options,
            )
        return value / len(distances)

    def _finalize(self, distances, context, state, *, region, observations, step):
        from .mlpe.fit import RegionalMLPECalibration
        from .mlpe.likelihood import finalize_mlpe_fit

        result = eqx.filter_jit(finalize_mlpe_fit)(
            distances.astype(jnp.float64),
            context.targets,
            *context.pairs,
            **self._options(context, state),
        )
        return RegionalMLPECalibration.from_fit(
            result,
            region_name=region.name,
            target=observations.target,
            population_ids=region.sampling_unit_ids,
            calibration_pairs=observations.observed_pairs,
            loss=self,
            optimizer_iterations=step,
            optimizer_message="Fixed-budget joint Adam; no variance convergence claim",
            converged=False,
        )

    def _fixed_parameters(self, fitted):
        # Arrays rather than the calibration's Python floats, which compiled
        # validation would treat as static and retrace at every evaluation.
        return _MLPEFixedParameters(
            jnp.asarray((fitted.intercept, fitted.slope)),
            jnp.asarray(fitted.distance_center),
            jnp.asarray(fitted.distance_scale),
        )

    def _calibration(self, target, fitted):
        from .calibration import MLPECalibrationModule

        return MLPECalibrationModule(target, dict(fitted))

    def _fit_group(self, distances, observations, *, region_name, max_iterations):
        from .mlpe.fit import calibrate_mlpe

        return calibrate_mlpe(
            distances,
            observations,
            region_name=region_name,
            loss=self,
            max_iterations=2000 if max_iterations is None else max_iterations,
        )

    def _state_metadata(self, state):
        return [
            dict(
                region=region,
                species_index=index,
                shape=list(group.raw_variances.shape),
                dtype=str(group.raw_variances.dtype),
            )
            for (region, index), group in state.groups.items()
        ]

    def _state_template(self, metadata):
        return _MLPEState(
            {
                (item["region"], item["species_index"]): _MLPEGroupState(
                    jnp.zeros(tuple(item["shape"]), dtype=item["dtype"])
                )
                for item in metadata
            }
        )


def _loss_metadata(loss):
    if not isinstance(loss, (Log1pLoss, MLPELoss)):
        raise ValueError("Saving requires a built-in loss")
    return {"kind": type(loss).__name__, "settings": asdict(loss)}


def _loss_from_metadata(metadata):
    factories = {"Log1pLoss": Log1pLoss, "MLPELoss": MLPELoss}
    return factories[metadata["kind"]](**metadata["settings"])
