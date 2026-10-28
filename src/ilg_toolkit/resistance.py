"""Differentiable effective-resistance layer on conductance grid graphs using JAXScape."""

from dataclasses import dataclass, field

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxscape import GridGraph, ResistanceDistance
from jaxscape.solvers import AMJaxCGSolverState

from .config import ResistanceSolverConfig


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _ResistanceSolverContext:
    """A shape-specific solver with dynamic, reusable preconditioner arrays."""

    graph_shape: tuple[int, int] = field(metadata={"static": True})
    solver: lx.AbstractLinearSolver
    solver_state: AMJaxCGSolverState | None = None
    config: ResistanceSolverConfig = field(
        default=ResistanceSolverConfig(), metadata={"static": True}
    )


def _mean_conductance(left, right):
    return 0.5 * (left + right)


def _effective_resistance(
    conductance,
    terminal_nodes,
    *,
    context: _ResistanceSolverContext,
) -> jax.Array:
    """Effective resistance among sampling-unit vertices of a positive conductance grid.

    Each edge in the four-neighbor grid averages its two vertices' conductances.
    Calculations use float64 and return the surface's floating data type, with
    zero resistance for coincident vertices.
    ``context`` is the solver prepared for the surface's shape. Invalid inputs
    or failed solves raise errors.
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
    resistances = resistances.at[jnp.diag_indices(nodes.size)].set(0)
    return resistances.astype(surface.dtype)
