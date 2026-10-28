"""Losses: statistical criteria of region–species groups and their numerical settings."""

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any, final, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

from ._checks import _positive_integer
from ._training_utils import _RegionSpeciesKey
from .data import PairwiseObservations, RegionBatch, TargetSpec


if TYPE_CHECKING:
    from .calibration import AbstractCalibrationHead


@final
class _LossContext(eqx.Module):
    """Model-scale targets of one region–species group and their sampling-unit pairs."""

    targets: jax.Array
    pairs: tuple[jax.Array, jax.Array]
    population_count: int = eqx.field(static=True)


class AbstractLoss(ABC):
    """Abstract loss: a statistical criterion for one region–species group.

    A loss defines its reduction, its statistical parameter state and the
    calibration head it produces. Training averages the losses of region–species
    groups over regions within each species, then over species. The target
    transform applies to the observations before the loss.
    """

    @abstractmethod
    def _prepare(self, region, observations, *, training: bool) -> _LossContext: ...

    @abstractmethod
    def _initialize(self, groups: Mapping[_RegionSpeciesKey, _LossContext]) -> Any: ...

    @abstractmethod
    def _group_state(self, state, identity: _RegionSpeciesKey) -> Any: ...

    @abstractmethod
    def _value(self, distances, context: _LossContext, state, fixed=None) -> jax.Array: ...

    @abstractmethod
    def _finalize(self, distances, context, state, *, region, observations, step) -> Any: ...

    @abstractmethod
    def _fixed_parameters(self, fitted) -> Any: ...

    @abstractmethod
    def _calibration_head(
        self,
        target: TargetSpec,
        fitted: Mapping[_RegionSpeciesKey, Any],
        species: tuple[str, ...] | None,
    ) -> "AbstractCalibrationHead": ...

    @abstractmethod
    def _iteration_budget(self, max_iterations: int | None) -> int | None: ...

    @abstractmethod
    def _fit_group(
        self,
        distances: Callable[[], jax.Array],
        observations: PairwiseObservations,
        *,
        max_iterations: int | None,
    ) -> Any: ...

    @abstractmethod
    def _state_metadata(self, state) -> Any: ...

    @abstractmethod
    def _state_template(self, metadata) -> Any: ...


def _context(region: RegionBatch, observations: PairwiseObservations) -> _LossContext:
    """Returns the model-scale targets and index pairs of observations in a region batch."""
    pairs, values = observations.aligned_pairs(region)
    return _LossContext(observations.target.forward(values), pairs, len(region.sampling_unit_ids))


@final
@dataclass(frozen=True)
class Log1pLoss(AbstractLoss):
    r"""Log1p loss: mean squared difference of log1p ecological distances and targets.

    For a region–species group with observed pairs $\mathcal O$, ecological
    distances $s_{ij}$, and model-scale targets $t_{ij}=T(y_{ij})$ under the
    target transform $T$,

    $$L=\frac{1}{|\mathcal O|}\sum_{(i,j)\in\mathcal O}
    \left[\log(1+s_{ij})-\log(1+t_{ij})\right]^2.$$

    Training averages $L$ over regions within each species, then over species. The
    log1p comparison applies after the target transform, and the target describes
    nonnegative dissimilarities. The loss accepts individual-based region batches.

    The loss has no statistical parameters. Its direct calibration head returns
    $T^{-1}(s)$ without coefficients or variance parameters, so the loss identifies
    an absolute connectivity scale.

    Examples:
        >>> config = TrainingConfig(loss=Log1pLoss(), steps=100)
        >>> state = train(region, observations, distance_model=distance_model, config=config)
        >>> head = fit_calibration_head(region, observations,
        ...                             state.selected_model.distance_model, loss=Log1pLoss())
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

    def _calibration_head(self, target, fitted, species):
        from .calibration import DirectCalibrationHead

        return DirectCalibrationHead(target)

    def _iteration_budget(self, max_iterations):
        if max_iterations is not None:
            raise ValueError("A direct calibration head has no iteration budget")
        return None

    def _fit_group(self, distances, observations, *, max_iterations):
        return None

    def _state_metadata(self, state):
        return None

    def _state_template(self, metadata):
        if metadata is not None:
            raise ValueError("Log1pLoss checkpoints have no statistical state")
        return None


@final
class _MLPEGroupState(eqx.Module):
    """Unconstrained population and residual variances of one region–species group."""

    raw_variances: jax.Array


@final
class _MLPEState(eqx.Module):
    """MLPE statistical parameter state, keyed by region–species group."""

    groups: dict[_RegionSpeciesKey, _MLPEGroupState]


@final
class _MLPEFixedParameters(eqx.Module):
    """Coefficients and distance moments from training, held fixed in validation."""

    coefficients: jax.Array
    distance_center: jax.Array
    distance_scale: jax.Array


@final
@dataclass(frozen=True)
class MLPELoss(AbstractLoss):
    r"""MLPE loss: marginal maximum-likelihood population effects (MLPE) negative log likelihood.

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
    distance moments and population-effect posteriors are estimated for each
    region–species group (group-specific calibration), and distance-model
    parameters are shared across groups. Standardization makes the loss invariant
    to the scale of ecological distances, so MLPE identifies no absolute
    connectivity scale. MLPE requires population sampling units and dissimilarity
    targets. Each region–species group yields an
    [`MLPECalibration`][ilg_toolkit.mlpe.MLPECalibration].

    Args:
        variance_floor: Positive lower bound of both variance components.
        min_distance_scale: Positive minimum sample standard deviation of the
            ecological distances of a region–species group.
        jitter: Fixed residual variance added to $e$, in [0, 1e-6].

    Raises:
        ValueError: If a setting is nonfinite or out of range.

    Examples:
        >>> loss = MLPELoss(jitter=0.0)
        >>> state = train(region, observations, distance_model=distance_model,
        ...               config=TrainingConfig(loss=loss, steps=500))
        >>> head = fit_calibration_head(region, observations,
        ...                             state.selected_model.distance_model,
        ...                             loss=loss, max_iterations=2000)

        Validation during training uses the coefficients, distance moments and
        variances from training. ``train`` updates distance-model and variance
        parameters with the configured optimizer, and zero training steps only
        initializes the variances. ``fit_calibration_head`` keeps the distance model
        fixed and estimates both variance components with deterministic multistart
        optimization.
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
        if observations.target.kind != "dissimilarity":
            raise ValueError("MLPE losses require dissimilarity targets")
        if region.individual_based:
            raise ValueError("MLPE losses reject individual-based region batches")
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
        from .mlpe.likelihood import _mlpe_ml_negative_log_likelihood, _profiled_mlpe_ml_fit

        distances = distances.astype(jnp.float64)
        options = self._options(context, state)
        if fixed is None:
            value = _profiled_mlpe_ml_fit(distances, context.targets, *context.pairs, **options)[0]
        else:
            value = _mlpe_ml_negative_log_likelihood(
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
        from .mlpe.fit import _calibration_from_fit
        from .mlpe.likelihood import _finalize_mlpe_fit

        result = eqx.filter_jit(_finalize_mlpe_fit)(
            distances.astype(jnp.float64),
            context.targets,
            *context.pairs,
            **self._options(context, state),
        )
        return _calibration_from_fit(
            result,
            population_ids=region.sampling_unit_ids,
            calibration_pairs=observations.observed_pairs,
            loss=self,
            optimizer_iterations=step,
            optimizer_message="Variances trained jointly with the distance model",
            converged=False,
        )

    def _fixed_parameters(self, fitted):
        # Arrays keep compiled validation from retracing when the values change;
        # the calibration stores Python floats, which compilation treats as static.
        return _MLPEFixedParameters(
            jnp.asarray((fitted.intercept, fitted.slope)),
            jnp.asarray(fitted.distance_center),
            jnp.asarray(fitted.distance_scale),
        )

    def _calibration_head(self, target, fitted, species):
        from .calibration import MLPECalibrationHead

        return MLPECalibrationHead(
            target,
            {
                (region, None if species is None else species[index]): calibration
                for (region, index), calibration in fitted.items()
            },
        )

    def _iteration_budget(self, max_iterations):
        from .mlpe.fit import MLPEError

        return _positive_integer(
            "max_iterations", 2000 if max_iterations is None else max_iterations, error=MLPEError
        )

    def _fit_group(self, distances, observations, *, max_iterations):
        from .mlpe.fit import _fit_mlpe

        assert max_iterations is not None
        return _fit_mlpe(distances(), observations, loss=self, max_iterations=max_iterations)

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
    """Returns the kind and settings of a built-in loss for a checkpoint header."""
    if not isinstance(loss, (Log1pLoss, MLPELoss)):
        raise ValueError("Saving requires a built-in loss")
    return {"kind": type(loss).__name__, "settings": asdict(loss)}


def _loss_from_metadata(metadata):
    """Returns the built-in loss described by checkpoint metadata."""
    factories = {"Log1pLoss": Log1pLoss, "MLPELoss": MLPELoss}
    return factories[metadata["kind"]](**metadata["settings"])
