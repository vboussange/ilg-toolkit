"""Log-linear patch encoder from patch-averaged landscape features."""

import math
from collections.abc import Sequence
from functools import partial
from numbers import Integral, Real
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .._training_utils import (
    ObservationCollection,
    RegionCollection,
    _normalize_training_inputs,
    _prepare_region_species_inputs,
    _resolve_training_species_order,
    _validate_shared_training_contract,
)
from ..config import ResistanceSolverConfig, validate_patch_microbatch_size
from ..data import RegionBatch
from ..losses import AbstractLoss
from ..resistance import effective_resistance
from .base import ConductanceModel, prepare_distance_context
from .common import encode_patches, pixel_to_patch_nodes

# Log-spaced uniform conductances screened before golden-section refinement.
_GRID_SIZE = 256
_REFINEMENT_STEPS = 40
# Relative loss improvement required to move a scale-invariant loss's intercept.
_IMPROVEMENT_RTOL = 1e-9


@final
class LogLinearConductance(ConductanceModel):
    """A log-linear patch encoder with species-specific coefficients.

    Patch-averaged landscape features form a linear predictor for each species
    index. Conductance is ``min_conductance + exp(predictor)``, capped at
    ``max_conductance``. Coefficient columns follow the raster feature order;
    coefficients and intercepts start at zero. ``initialize_uniform_conductance``
    instead starts the intercepts at the best uniform conductance.

    Args:
        in_channels:
            Number of landscape features, in their retained feature order.
        num_species: Number of species-specific coefficient rows; default 1.
        patch_size: Side length of square raster patches, default 32. Both raster
            dimensions must be divisible by it. Sampling units in the same patch
            share a vertex.
        patch_microbatch_size: Maximum patches encoded together; ``None`` encodes all
            patches at once. This controls peak encoder memory.
        min_conductance: Positive floor, default 0.01, added to the exponentiated linear
            predictor.
        max_conductance: Positive cap, default 1000. Conductance is
            ``min(max_conductance, min_conductance + exp(features @ coefficients + intercept))``.
        solver_config: Effective-resistance tolerances and iteration limit; see
            ``ResistanceSolverConfig``. Enable JAX float64 before using conductance
            models.

    Notes:
        Patch microbatch size does not change graph resolution or ecological context.
        Prepare feature scaling and categorical reference levels before constructing
        a region batch. Use full-rank coding with the intercept (for example, k-1
        indicators for k categories). ``train`` estimates each species' coefficients
        from all of its regions (complete pooling); species share no parameters.

        The intercept scales the exponential before the floor and cap, so those limits
        can alter relative conductances. Standardized maximum-likelihood population effects
        (MLPE) does not identify an absolute connectivity scale; interpret conductance
        patterns on a relative scale.
    """

    coefficients: jax.Array
    intercepts: jax.Array
    num_species: int = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    patch_size: int = eqx.field(static=True)
    patch_microbatch_size: int | None = eqx.field(static=True)
    min_conductance: float = eqx.field(static=True)
    max_conductance: float = eqx.field(static=True)
    solver_config: ResistanceSolverConfig = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        *,
        num_species: int = 1,
        patch_size: int = 32,
        patch_microbatch_size: int | None = None,
        min_conductance: float = 0.01,
        max_conductance: float = 1000.0,
        solver_config: ResistanceSolverConfig | None = None,
    ):
        for name, value in (
            ("in_channels", in_channels),
            ("num_species", num_species),
            ("patch_size", patch_size),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            any(
                isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
                for value in (min_conductance, max_conductance)
            )
            or not 0 < min_conductance < max_conductance
        ):
            raise ValueError(
                "Require finite conductance bounds 0 < min_conductance < max_conductance"
            )
        validate_patch_microbatch_size(patch_microbatch_size)
        if solver_config is not None and not isinstance(solver_config, ResistanceSolverConfig):
            raise TypeError("solver_config must be a ResistanceSolverConfig")
        self.num_species = int(num_species)
        self.in_channels = int(in_channels)
        self.patch_size = int(patch_size)
        self.patch_microbatch_size = patch_microbatch_size
        self.min_conductance = float(min_conductance)
        self.max_conductance = float(max_conductance)
        self.solver_config = ResistanceSolverConfig() if solver_config is None else solver_config
        self.coefficients = jnp.zeros((num_species, in_channels))
        self.intercepts = jnp.zeros(num_species)

    def backbone(self, features: jax.Array) -> jax.Array:
        """Patch-averaged landscape features."""
        if features.ndim != 3 or features.shape[-1] != self.in_channels:
            raise ValueError(f"Expected row, column, feature data with {self.in_channels} features")
        return encode_patches(self, features, partial(jnp.mean, axis=(1, 2)))

    def conductance(self, latent: jax.Array, species_index: int | jax.Array) -> jax.Array:
        """Conductance surface for one species from patch-averaged landscape features."""
        predictor = latent @ self.coefficients[species_index] + self.intercepts[species_index]
        ceiling = jnp.log(self.max_conductance - self.min_conductance)
        return jnp.clip(
            jnp.exp(jnp.minimum(predictor, ceiling)) + self.min_conductance,
            self.min_conductance,
            self.max_conductance,
        )


def initialize_uniform_conductance(
    model: LogLinearConductance,
    region: RegionCollection,
    observations: ObservationCollection,
    loss: AbstractLoss,
    *,
    species: Sequence[str] | None = None,
) -> LogLinearConductance:
    """A log-linear patch encoder started at the best uniform conductance.

    Coefficients are set to zero, so every vertex of the conductance grid graph
    has the same conductance ``C``. Each species' intercept is set to the ``C``
    minimizing ``loss`` on that species' pairwise genetic observations, with the
    region and species weighting used by ``train``. Short training runs then
    spend their steps on landscape effects rather than on the global scale.

    On a uniform conductance grid graph, effective resistance scales as ``1 / C``.
    The search therefore needs one effective-resistance solve per region at unit
    conductance, followed by a one-dimensional search over ``log(C)`` in the open
    interval ``(min_conductance, max_conductance)``: a log-spaced grid, refined
    by golden-section search.

    Args:
        model: Log-linear patch encoder providing the conductance bounds, patch
            size, solver settings, and species count.
        region: One region batch, a sequence, or a mapping keyed by region name,
            as accepted by ``train``.
        observations: Pairwise genetic observations aligned with the regions, as
            accepted by ``train``.
        loss: Loss whose weighted value the uniform conductance minimizes.
        species: Species names in parameter order. Required for multispecies
            models; a single species can be inferred from the inputs.

    Returns:
        A copy of ``model`` with zero coefficients and one fitted intercept per
        species.

    Raises:
        TypeError: ``model`` is not a ``LogLinearConductance`` or ``loss`` is not
            an ``AbstractLoss``.
        ValueError: The inputs violate the contract that ``train`` enforces.

    Notes:
        The initialization only matters for losses that identify an absolute
        connectivity scale, such as ``Log1pLoss``. Standardized maximum-likelihood
        population effects (MLPE) is invariant to the uniform conductance, so an
        intercept is kept unless the loss improves on it.

    Examples:
        >>> distance_model = initialize_uniform_conductance(
        ...     LogLinearConductance(3, patch_size=8), region, observations, Log1pLoss()
        ... )
        >>> state = train(region, observations, distance_model=distance_model,
        ...               config=TrainingConfig(loss=Log1pLoss(), steps=100))
    """
    if not isinstance(model, LogLinearConductance):
        raise TypeError("initialize_uniform_conductance requires a LogLinearConductance")
    if not isinstance(loss, AbstractLoss):
        raise TypeError("loss must be an AbstractLoss")
    inputs = _normalize_training_inputs(region, observations)
    first = _validate_shared_training_contract(inputs)
    if first.region.feature_array.shape[-1] != model.in_channels:
        raise ValueError(f"Expected landscape features with {model.in_channels} channels")
    order = _resolve_training_species_order(inputs, species, model.num_species)
    groups = _prepare_region_species_inputs(inputs, order, model.num_species)
    unit_resistances = {
        item.region.name: _unit_conductance_resistances(model, item.region) for item in inputs
    }
    contexts = {
        group.key: loss._prepare(group.region, group.observations, training=True)
        for group in groups
    }
    loss_state = loss._initialize(contexts)
    lower, upper = math.log(model.min_conductance), math.log(model.max_conductance)
    intercepts = []
    for index in range(model.num_species):
        selected = [group for group in groups if group.species_index == index]
        if not selected:
            intercepts.append(model.intercepts[index])
            continue
        distances = []
        for group in selected:
            vertices, resistances = unit_resistances[group.region.name]
            nodes = np.searchsorted(vertices, _patch_vertices(model, group.region))
            left, right = contexts[group.key].pairs
            distances.append(resistances[nodes[np.asarray(left)], nodes[np.asarray(right)]])
        current = jnp.log(model.min_conductance + jnp.exp(model.intercepts[index]))
        log_conductance = _best_log_conductance(
            loss,
            tuple(distances),
            tuple(contexts[group.key] for group in selected),
            tuple(loss._group_state(loss_state, group.key) for group in selected),
            jnp.asarray([group.loss_weight for group in selected]),
            jnp.asarray([lower, upper]),
            current,
        )
        # Conductance min_conductance + exp(intercept), written stably near the floor.
        intercept = lower + jnp.log(jnp.expm1(log_conductance - lower))
        intercepts.append(intercept.astype(model.intercepts.dtype))
    return eqx.tree_at(
        lambda current: (current.coefficients, current.intercepts),
        model,
        (jnp.zeros_like(model.coefficients), jnp.stack(intercepts)),
    )


def _patch_vertices(model: LogLinearConductance, region: RegionBatch) -> np.ndarray:
    """Conductance grid graph vertices of a region's sampling units."""
    rows, columns = region.feature_array.shape[:2]
    return np.asarray(
        pixel_to_patch_nodes(
            region.pixel_nodes, raster_shape=(rows, columns), patch_size=model.patch_size
        )
    )


def _unit_conductance_resistances(
    model: LogLinearConductance, region: RegionBatch
) -> tuple[np.ndarray, jax.Array]:
    """Sampling-unit vertices and their effective resistances at unit uniform conductance."""
    context = prepare_distance_context(model, region.feature_array)
    assert context is not None
    vertices = np.unique(_patch_vertices(model, region))
    surface = jnp.ones(context.graph_shape, dtype=jnp.float64)
    return vertices, effective_resistance(surface, jnp.asarray(vertices), context=context)


def _species_loss(log_conductance, loss, distances, contexts, states, weights):
    """Weighted loss of one species' region groups at a uniform conductance."""
    conductance = jnp.exp(log_conductance)
    values = [
        loss._value(unit / conductance, context, state)
        for unit, context, state in zip(distances, contexts, states, strict=True)
    ]
    return jnp.dot(weights, jnp.stack(values))


@eqx.filter_jit
def _best_log_conductance(loss, distances, contexts, states, weights, bounds, current):
    """Log uniform conductance minimizing a species' loss inside open bounds."""

    def value(log_conductance):
        return _species_loss(log_conductance, loss, distances, contexts, states, weights)

    lower, upper = bounds
    grid = lower + (upper - lower) * jnp.arange(1, _GRID_SIZE + 1) / (_GRID_SIZE + 1)
    best = jnp.argmin(jax.vmap(value)(grid))
    edges = jnp.concatenate([lower[None], grid, upper[None]])
    ratio = (math.sqrt(5) - 1) / 2

    def refine(_, bracket):
        left, right = bracket
        inner_left = right - ratio * (right - left)
        inner_right = left + ratio * (right - left)
        keep_left = value(inner_left) < value(inner_right)
        return jnp.where(keep_left, left, inner_left), jnp.where(keep_left, inner_right, right)

    # The bracket stays wider than float64 spacing, so its midpoint is inside the bounds.
    left, right = jax.lax.fori_loop(0, _REFINEMENT_STEPS, refine, (edges[best], edges[best + 2]))
    refined = (left + right) / 2
    current_value = value(current)
    improves = value(refined) < current_value - _IMPROVEMENT_RTOL * jnp.abs(current_value)
    inside = (current > lower) & (current < upper)
    return jnp.where(improves | ~inside, refined, current)
