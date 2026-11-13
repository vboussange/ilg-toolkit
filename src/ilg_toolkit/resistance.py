"""Differentiable effective-resistance layer on conductance grid graphs using JAXScape."""

import math
from dataclasses import dataclass, replace
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxscape import GridGraph, GridTopology, ResistanceDistance


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _ResistanceSolverContext:
    """A resistance solver for the conductance grid graphs of one shape, with its solver state.

    Attributes:
        topology: The edges of the conductance grid graphs the context solves, built
            once for their shape, so that each solve computes only the edge weights.
        solver: The resistance solver, a Lineax solver.
        solver_state: The solver state of the latest solve, which the next solve
            reinitializes.
    """

    topology: GridTopology
    solver: lx.AbstractLinearSolver
    solver_state: Any

    @property
    def graph_shape(self) -> tuple[int, int]:
        """Shape of the conductance grid graphs the context solves."""
        return self.topology.shape


def _mean_conductance(left, right):
    """Returns the conductance of an edge: the mean of its two vertices' conductances."""
    return 0.5 * (left + right)


def _resistance_solver(solver: lx.AbstractLinearSolver | None) -> lx.AbstractLinearSolver:
    """Returns the resistance solver that ``solver`` selects.

    Args:
        solver: A Lineax solver, or ``None`` for the default solver of JAXScape's
            ``ResistanceDistance``.

    Returns:
        ``solver``, or JAXScape's default solver for ``None``.

    Raises:
        TypeError: If ``solver`` is neither ``None`` nor a Lineax solver.
    """
    solver = ResistanceDistance().solver if solver is None else solver
    if not isinstance(solver, lx.AbstractLinearSolver):
        raise TypeError("solver must be a Lineax solver")
    return solver


def _resistance_context(
    graph_shape: tuple[int, ...], solver: lx.AbstractLinearSolver | None
) -> _ResistanceSolverContext:
    """Prepares a resistance solver for conductance grid graphs of one shape.

    The context holds the edges of the grid graphs of its shape and the solver state
    that ``ResistanceDistance.init`` prepares on a uniform conductance grid graph.
    Preparation runs eagerly, also inside traced functions.

    Args:
        graph_shape: Shape of the conductance grid graphs, ``(rows, columns)``.
        solver: The resistance solver, a Lineax solver, or ``None`` for the default
            solver of JAXScape's ``ResistanceDistance``.

    Returns:
        The context to pass to ``_effective_resistance``.

    Raises:
        ValueError: If ``graph_shape`` is not a 2D shape with at least two vertices.
        TypeError: If ``solver`` is neither ``None`` nor a Lineax solver.
    """
    if len(graph_shape) != 2 or math.prod(graph_shape) < 2:
        raise ValueError("Effective resistance requires a 2D surface with at least two vertices")
    solver = _resistance_solver(solver)
    # Build the solver state from concrete arrays, whatever the JAXScape version: a state
    # built inside a trace can lose what later solves reuse, as JAXScape's
    # `CholmodSolver` drops its factor cache and refactorizes on every solve.
    with jax.ensure_compile_time_eval():
        topology = GridTopology(graph_shape)
        template = GridGraph(
            jnp.ones(graph_shape, dtype=jnp.float64), fun=_mean_conductance, topology=topology
        )
        state = ResistanceDistance(solver=solver).init(template)
    return _ResistanceSolverContext(topology, solver, state)


@eqx.filter_jit
def _effective_resistance(
    conductance,
    terminal_nodes,
    *,
    context: _ResistanceSolverContext,
) -> tuple[jax.Array, _ResistanceSolverContext]:
    """Computes effective resistances among vertices of a conductance grid graph.

    Each edge of the four-neighbor grid graph averages its two vertices'
    conductances. The solve runs and returns in float64, with zero resistance
    between coincident vertices. Invalid inputs and failed solves raise errors.

    The returned context holds the solver state reinitialized for ``conductance``.
    Passing it to the next call lets the solver reuse it, as cuDSS reuses its
    analysis; passing an earlier state stays correct.

    Args:
        conductance: Positive conductance surface with shape ``(rows, columns)``.
        terminal_nodes: Row-major vertex indices of the sampling units.
        context: Resistance solver prepared for the surface's shape.

    Returns:
        The float64 square matrix of effective resistances among ``terminal_nodes``,
        and the context for the next solve.
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
    graph = GridGraph(surface.astype(jnp.float64), fun=_mean_conductance, topology=context.topology)
    distance = ResistanceDistance(solver=context.solver)
    state = distance.reinit(graph, context.solver_state)
    resistances = distance.nodes_to_nodes_distance(graph, nodes, state=state)
    # Solver calls with side effects, such as cuDSS's, run unordered, and the next
    # reinitialization can replace what this state holds, so it waits for these solves.
    resistances, state = jax.lax.optimization_barrier((resistances, state))
    return resistances.at[jnp.diag_indices(nodes.size)].set(0), replace(context, solver_state=state)
