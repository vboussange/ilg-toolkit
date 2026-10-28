"""Log-linear patch encoder from patch-averaged landscape features."""

import math
from collections.abc import Mapping, Sequence
from functools import partial
from numbers import Real
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .._checks import _positive_integer, _require_float64
from .._training_utils import _prepare_training_inputs
from ..config import ResistanceSolverConfig
from ..data import PairwiseObservations, RegionBatch
from ..losses import AbstractLoss
from ..resistance import _effective_resistance
from .base import (
    _check_regions,
    _model_species,
    _prepare_distance_context,
    _species_names,
    _species_rows,
    ConductanceModel,
)
from .common import _encode_patches, _pixel_to_patch_nodes


# Log-spaced uniform conductances screened before golden-section refinement.
_SEARCH_GRID_SIZE = 256
_REFINEMENT_STEPS = 40
# Relative loss improvement required to move a scale-invariant loss's intercept.
_IMPROVEMENT_RTOL = 1e-9


@final
class LogLinearConductance(ConductanceModel):
    """A log-linear patch encoder with species-specific coefficients.

    Patch-averaged landscape features form a linear predictor for each species.
    Conductance is ``min_conductance + exp(predictor)``, capped at
    ``max_conductance``. Coefficient columns follow the order of the landscape
    features; coefficients and intercepts start at zero.
    ``initialize_uniform_conductance`` starts the intercepts at the best uniform
    conductance instead.

    Args:
        in_channels:
            Number of landscape features, in their retained feature order.
        species: Species names, one coefficient row each in parameter order;
            ``None`` (default) models one unnamed species.
        patch_size: Side length of square raster patches, default 32. Both raster
            dimensions must be divisible by it. Sampling units in the same patch
            share a vertex.
        patch_microbatch_size: Maximum patches encoded together; ``None`` encodes all
            patches at once. It bounds the peak memory of the backbone.
        min_conductance: Positive floor, default 0.01, added to the exponentiated linear
            predictor.
        max_conductance: Positive cap, default 1000. Conductance is
            ``min(max_conductance, min_conductance + exp(features @ coefficients + intercept))``.
        solver_config: Effective-resistance tolerances and iteration limit; see
            ``ResistanceSolverConfig``.

    Notes:
        Prepare feature scaling and categorical reference levels before building a
        region batch, and use full-rank coding with the intercept (for example,
        k-1 indicators for k categories). ``train`` estimates each species'
        coefficients from all of its regions (complete pooling); species share no
        parameters.

        The intercept scales the exponential before the floor and cap, so those
        limits change relative conductances near them. Maximum-likelihood
        population effects (MLPE) losses standardize ecological distances, so
        they identify conductances up to a common factor.
    """

    coefficients: jax.Array
    intercepts: jax.Array
    species: tuple[str, ...] | None = eqx.field(static=True)
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
        species: Sequence[str] | None = None,
        patch_size: int = 32,
        patch_microbatch_size: int | None = None,
        min_conductance: float = 0.01,
        max_conductance: float = 1000.0,
        solver_config: ResistanceSolverConfig | None = None,
    ):
        in_channels = _positive_integer("in_channels", in_channels)
        species = _species_names(species)
        patch_size = _positive_integer("patch_size", patch_size)
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
        patch_microbatch_size = _positive_integer(
            "patch_microbatch_size", patch_microbatch_size, optional=True
        )
        if solver_config is not None and not isinstance(solver_config, ResistanceSolverConfig):
            raise TypeError("solver_config must be a ResistanceSolverConfig")
        self.species = species
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.patch_microbatch_size = patch_microbatch_size
        self.min_conductance = float(min_conductance)
        self.max_conductance = float(max_conductance)
        self.solver_config = ResistanceSolverConfig() if solver_config is None else solver_config
        count = _species_rows(species)
        self.coefficients = jnp.zeros((count, in_channels))
        self.intercepts = jnp.zeros(count)

    def backbone(self, features: jax.Array) -> jax.Array:
        """Averages landscape features over each patch.

        Args:
            features: Landscape features with shape ``(rows, columns, in_channels)``.

        Returns:
            Patch means with shape ``(rows // patch_size, columns // patch_size,
            in_channels)``.
        """
        if features.ndim != 3 or features.shape[-1] != self.in_channels:
            raise ValueError(f"Expected row, column, feature data with {self.in_channels} features")
        return _encode_patches(self, features, partial(jnp.mean, axis=(1, 2)))

    def conductance(self, latent: jax.Array, species_index: int | jax.Array) -> jax.Array:
        """Computes one species' conductance surface from patch-averaged landscape features.

        Args:
            latent: Patch means returned by ``backbone``.
            species_index: Parameter row of the species, an integer or a scalar
                integer array.

        Returns:
            Conductances between ``min_conductance`` and ``max_conductance``, one
            per vertex.
        """
        predictor = latent @ self.coefficients[species_index] + self.intercepts[species_index]
        ceiling = jnp.log(self.max_conductance - self.min_conductance)
        return jnp.clip(
            jnp.exp(jnp.minimum(predictor, ceiling)) + self.min_conductance,
            self.min_conductance,
            self.max_conductance,
        )


def initialize_uniform_conductance(
    model: LogLinearConductance,
    region: RegionBatch | Sequence[RegionBatch] | Mapping[str, RegionBatch],
    observations: PairwiseObservations
    | Sequence[PairwiseObservations | Sequence[PairwiseObservations]]
    | Mapping[str, PairwiseObservations | Sequence[PairwiseObservations]],
    loss: AbstractLoss,
) -> LogLinearConductance:
    """Initializes a log-linear patch encoder at the best uniform conductance.

    Coefficients are set to zero, so every vertex of the conductance grid graph
    has the same conductance ``C``. Each species' intercept is set to the ``C``
    minimizing ``loss`` on that species' pairwise genetic observations, with the
    region and species weighting of ``train``.

    On a uniform conductance grid graph, effective resistance scales as ``1 / C``.
    The search runs one effective-resistance solve per region at unit
    conductance, then a one-dimensional search over ``log(C)`` in the open
    interval ``(min_conductance, max_conductance)``: a log-spaced grid, refined
    by golden-section search.

    Args:
        model: Log-linear patch encoder providing the conductance bounds, patch
            size, solver settings, and species.
        region: One region batch, a sequence, or a mapping keyed by region name,
            as accepted by ``train``.
        observations: Pairwise genetic observations aligned with the regions, as
            accepted by ``train``.
        loss: Loss whose weighted value the uniform conductance minimizes.

    Returns:
        A copy of ``model`` with zero coefficients and one fitted intercept per
        species.

    Raises:
        TypeError: ``model`` is not a ``LogLinearConductance`` or ``loss`` is not
            an ``AbstractLoss``.
        ValueError: The inputs violate the contract that ``train`` enforces.

    Notes:
        Losses that identify an absolute conductance scale, such as ``Log1pLoss``,
        depend on the uniform conductance. Maximum-likelihood population effects
        (MLPE) losses do not, so an intercept stays unless the loss improves on it.

    Examples:
        >>> distance_model = initialize_uniform_conductance(
        ...     LogLinearConductance(3, patch_size=8), region, observations, Log1pLoss()
        ... )
        >>> state = train(region, observations, distance_model=distance_model,
        ...               config=TrainingConfig(loss=Log1pLoss(), steps=100))
    """
    _require_float64()
    if not isinstance(model, LogLinearConductance):
        raise TypeError("initialize_uniform_conductance requires a LogLinearConductance")
    if not isinstance(loss, AbstractLoss):
        raise TypeError("loss must be an AbstractLoss")
    prepared = _prepare_training_inputs(region, observations, model.species)
    _check_regions(model, prepared.regions)
    if prepared.inputs[0].region.feature_array.shape[-1] != model.in_channels:
        raise ValueError(f"Expected landscape features with {model.in_channels} channels")
    groups = prepared.training
    regions = {item.region.name: item.region for item in prepared.inputs}
    unit_resistances = {
        name: _unit_conductance_resistances(model, batch) for name, batch in regions.items()
    }
    contexts = {
        group.key: loss._prepare(group.region, group.observations, training=True)
        for group in groups
    }
    loss_state = loss._initialize(contexts)
    lower, upper = math.log(model.min_conductance), math.log(model.max_conductance)
    intercepts = []
    for index in range(len(_model_species(model))):
        selected = [group for group in groups if group.species_index == index]
        if not selected:
            intercepts.append(model.intercepts[index])
            continue
        distances = []
        for group in selected:
            matrix_rows, resistances = unit_resistances[group.region.name]
            rows = np.asarray([matrix_rows[pixel] for pixel in group.region.pixel_nodes.tolist()])
            left, right = contexts[group.key].pairs
            distances.append(resistances[rows[np.asarray(left)], rows[np.asarray(right)]])
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


def _unit_conductance_resistances(
    model: LogLinearConductance, region: RegionBatch
) -> tuple[dict[int, int], jax.Array]:
    """Computes effective resistances among a region's sampling-unit vertices at unit conductance.

    Returns the resistance-matrix row of each sampling-unit pixel, shared by the
    sampling units of every species, and the float64 matrix.
    """
    context = _prepare_distance_context(model, region.feature_array)
    assert context is not None
    height, width = region.feature_array.shape[:2]
    pixels = np.asarray(region.pixel_nodes)
    vertices = np.asarray(
        _pixel_to_patch_nodes(pixels, raster_shape=(height, width), patch_size=model.patch_size)
    )
    unique = np.unique(vertices)
    rows = np.searchsorted(unique, vertices)
    surface = jnp.ones(context.graph_shape, dtype=model.intercepts.dtype)
    matrix_rows = dict(zip(pixels.tolist(), rows.tolist(), strict=True))
    return matrix_rows, _effective_resistance(surface, jnp.asarray(unique), context=context)


def _species_loss(log_conductance, loss, distances, contexts, states, weights):
    """Returns the weighted loss of one species' region–species groups at a uniform conductance."""
    conductance = jnp.exp(log_conductance)
    values = [
        loss._value(unit / conductance, context, state)
        for unit, context, state in zip(distances, contexts, states, strict=True)
    ]
    return jnp.dot(weights, jnp.stack(values))


@eqx.filter_jit
def _best_log_conductance(loss, distances, contexts, states, weights, bounds, current):
    """Returns the log uniform conductance minimizing a species' loss inside open bounds."""

    def value(log_conductance):
        return _species_loss(log_conductance, loss, distances, contexts, states, weights)

    lower, upper = bounds
    grid = lower + (upper - lower) * jnp.arange(1, _SEARCH_GRID_SIZE + 1) / (_SEARCH_GRID_SIZE + 1)
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
    # A nonfinite loss at the current intercept never blocks the refined optimum.
    improves = ~jnp.isfinite(current_value) | (
        value(refined) < current_value - _IMPROVEMENT_RTOL * jnp.abs(current_value)
    )
    inside = (current > lower) & (current < upper)
    return jnp.where(improves | ~inside, refined, current)
