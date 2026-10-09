"""Differentiable resistance through JAXScape, with explicit solver failures."""

from dataclasses import dataclass, field
from numbers import Integral

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxscape import GridGraph, ResistanceDistance
from jaxscape.solvers import AMJaxCGSolver, AMJaxCGSolverState

from .config import SolverConfig


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SolverContext:
    """A shape-specific solver with dynamic, reusable preconditioner arrays."""

    graph_shape: tuple[int, int] = field(metadata={"static": True})
    solver: lx.AbstractLinearSolver
    solver_state: AMJaxCGSolverState | None = None


def _require_x64():
    if not jax.config.x64_enabled:
        raise RuntimeError(
            "Effective resistance requires float64: set JAX_ENABLE_X64=true before Python"
        )


def _mean_conductance(left, right):
    return 0.5 * (left + right)


def build_solver_context(graph_shape, config: SolverConfig | None = None) -> SolverContext:
    """Build CG or an optional AMG hierarchy outside JIT/gradient transforms.

    The all-ones hierarchy is reused as a preconditioner only. Every resistance
    calculation retains the learned conductances as its differentiable operator.
    """
    _require_x64()
    if len(graph_shape) != 2 or any(
        isinstance(size, bool) or not isinstance(size, Integral) or size <= 0
        for size in graph_shape
    ):
        raise ValueError("graph_shape must contain two positive integer dimensions")
    shape = tuple(int(size) for size in graph_shape)
    if shape[0] * shape[1] < 2:
        raise ValueError("Effective resistance requires at least two graph vertices")
    config = config or SolverConfig()
    options = dict(rtol=config.rtol, atol=config.atol, max_steps=config.max_steps)
    if not config.use_amg:
        return SolverContext(shape, lx.CG(**options))
    solver = AMJaxCGSolver(**options, coarse_solver="pinv")
    template = GridGraph(jnp.ones(shape, dtype=jnp.float64), fun=_mean_conductance)
    state = ResistanceDistance(solver=solver).init_preconditioner(template)
    return SolverContext(shape, solver, state)


def effective_resistance(
    conductance,
    terminal_nodes,
    *,
    context: SolverContext | None = None,
    config: SolverConfig | None = None,
) -> jax.Array:
    """Return terminal resistance scores for a positive four-neighbour grid.

    Edge conductance is the arithmetic mean of its endpoints. Graph solves use
    float64; scores and their gradients retain the surface's floating dtype.
    The last graph vertex is grounded and may be a terminal. Coincident terminal
    nodes have zero resistance. Invalid inputs or failed solves raise, including
    under JIT; numerical failure never triggers an implicit solver replacement.
    """
    _require_x64()
    surface = jnp.asarray(conductance)
    if surface.ndim != 2 or surface.size < 2:
        raise ValueError("conductance must be a 2D surface with at least two graph vertices")
    if not jnp.issubdtype(surface.dtype, jnp.floating):
        raise TypeError("conductance must have a real floating dtype")
    surface = eqx.error_if(
        surface,
        jnp.any(~jnp.isfinite(surface) | (surface <= 0)),
        "conductance must be finite and strictly positive",
    )
    nodes = jnp.asarray(terminal_nodes)
    if nodes.ndim != 1 or nodes.size == 0 or not jnp.issubdtype(nodes.dtype, jnp.integer):
        raise ValueError("terminal_nodes must be a nonempty one-dimensional integer array")
    nodes = eqx.error_if(
        nodes,
        jnp.any((nodes < 0) | (nodes >= surface.size)),
        "terminal_nodes fall outside the conductance grid",
    ).astype(jnp.int32)
    if context is not None and config is not None:
        raise ValueError("Supply either a solver context or a solver configuration")
    context = context or build_solver_context(surface.shape, config)
    if context.graph_shape != surface.shape:
        raise ValueError(
            f"Solver context shape {context.graph_shape} does not match surface {surface.shape}"
        )
    graph = GridGraph(surface.astype(jnp.float64), fun=_mean_conductance)
    scores = ResistanceDistance(solver=context.solver).nodes_to_nodes_distance(
        graph, nodes, state=context.solver_state
    )
    scores = scores.at[jnp.diag_indices(nodes.size)].set(0)
    return scores.astype(surface.dtype)
