"""Differentiable effective-resistance layer on conductance grid graphs using JAXScape."""

import math
from dataclasses import dataclass, field

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxscape import GridGraph, ResistanceDistance
from jaxscape.solvers import AMJaxCGSolver, AMJaxCGSolverState

from .config import ResistanceSolverConfig


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _ResistanceSolverContext:
    """A resistance solver for one graph shape, with reusable preconditioner arrays."""

    graph_shape: tuple[int, ...] = field(metadata={"static": True})
    solver: lx.AbstractLinearSolver
    solver_state: AMJaxCGSolverState | None = None


def _mean_conductance(left, right):
    """Returns the conductance of an edge: the mean of its two vertices' conductances."""
    return 0.5 * (left + right)


def _resistance_context(
    graph_shape: tuple[int, ...], config: ResistanceSolverConfig
) -> _ResistanceSolverContext:
    """Prepares a reusable resistance solver for a conductance grid graph of one shape.

    Conjugate gradient, with optional algebraic multigrid preconditioning prepared
    on a uniform conductance grid graph, follows ``config``. Preparation runs
    eagerly, also inside traced functions.
    """
    if len(graph_shape) != 2 or math.prod(graph_shape) < 2:
        raise ValueError("Effective resistance requires a 2D surface with at least two vertices")
    if not config.use_amg:
        return _ResistanceSolverContext(
            graph_shape,
            lx.CG(rtol=config.rtol, atol=config.atol, max_steps=config.max_steps),
        )
    solver = AMJaxCGSolver(
        rtol=config.rtol, atol=config.atol, max_steps=config.max_steps, coarse_solver="pinv"
    )
    with jax.ensure_compile_time_eval():
        template = GridGraph(jnp.ones(graph_shape, dtype=jnp.float64), fun=_mean_conductance)
        state = ResistanceDistance(solver=solver).init_preconditioner(template)
    return _ResistanceSolverContext(graph_shape, solver, state)


def _effective_resistance(
    conductance,
    terminal_nodes,
    *,
    context: _ResistanceSolverContext,
) -> jax.Array:
    """Computes effective resistances among vertices of a conductance grid graph.

    Each edge of the four-neighbor grid graph averages its two vertices'
    conductances. The solve runs and returns in float64, with zero resistance
    between coincident vertices. Invalid inputs and failed solves raise errors.

    Args:
        conductance: Positive conductance surface with shape ``(rows, columns)``.
        terminal_nodes: Row-major vertex indices of the sampling units.
        context: Resistance solver prepared for the surface's shape.

    Returns:
        The float64 square matrix of effective resistances among ``terminal_nodes``.
    """
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
    if context.graph_shape != surface.shape:
        raise ValueError(
            f"Solver context shape {context.graph_shape} does not match surface {surface.shape}"
        )
    graph = GridGraph(surface.astype(jnp.float64), fun=_mean_conductance)
    resistances = ResistanceDistance(solver=context.solver).nodes_to_nodes_distance(
        graph, nodes, state=context.solver_state
    )
    return resistances.at[jnp.diag_indices(nodes.size)].set(0)
