"""Differentiable resistance calculation on conductance graphs using JAXScape."""

from dataclasses import dataclass, field
from numbers import Integral

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxscape import GridGraph, ResistanceDistance
from jaxscape.solvers import AMJaxCGSolver, AMJaxCGSolverState

from .config import ResistanceSolverConfig


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ResistanceSolverContext:
    """A shape-specific solver with dynamic, reusable preconditioner arrays."""

    graph_shape: tuple[int, int] = field(metadata={"static": True})
    solver: lx.AbstractLinearSolver
    solver_state: AMJaxCGSolverState | None = None
    config: ResistanceSolverConfig = field(
        default=ResistanceSolverConfig(), metadata={"static": True}
    )


def _require_x64():
    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError(
            "Effective resistance requires float64: set JAX_ENABLE_X64=true before Python"
        )


def _mean_conductance(left, right):
    return 0.5 * (left + right)


def build_resistance_context(
    graph_shape: tuple[int, int], config: ResistanceSolverConfig | None = None
) -> ResistanceSolverContext:
    """A reusable conjugate-gradient solver with optional algebraic multigrid preconditioning.

    Preparation uses a uniform conductance graph of the supplied shape.
    Resistance calculation uses the learned conductances and their gradients.
    """
    _require_x64()
    if len(graph_shape) != 2 or any(
        isinstance(size, bool) or not isinstance(size, Integral) or size <= 0
        for size in graph_shape
    ):
        raise ValueError("graph_shape must contain two positive integer dimensions")
    shape = (int(graph_shape[0]), int(graph_shape[1]))
    if shape[0] * shape[1] < 2:
        raise ValueError("Effective resistance requires at least two graph nodes")
    config = config or ResistanceSolverConfig()
    if not config.use_amg:
        return ResistanceSolverContext(
            shape,
            lx.CG(rtol=config.rtol, atol=config.atol, max_steps=config.max_steps),
            config=config,
        )
    solver = AMJaxCGSolver(
        rtol=config.rtol, atol=config.atol, max_steps=config.max_steps, coarse_solver="pinv"
    )
    template = GridGraph(jnp.ones(shape, dtype=jnp.float64), fun=_mean_conductance)
    state = ResistanceDistance(solver=solver).init_preconditioner(template)
    return ResistanceSolverContext(shape, solver, state, config)


def effective_resistance(
    conductance,
    terminal_nodes,
    *,
    context: ResistanceSolverContext | None = None,
    config: ResistanceSolverConfig | None = None,
) -> jax.Array:
    """Effective resistance among sampling-unit nodes of a positive conductance grid.

    Each edge in the four-neighbor grid averages its two nodes' conductances.
    Calculations use float64 and return the surface's floating data type, with
    zero resistance for coincident nodes.
    Invalid inputs or failed solves raise errors.
    """
    _require_x64()
    surface = jnp.asarray(conductance)
    if surface.ndim != 2 or surface.size < 2:
        raise ValueError("conductance must be a 2D surface with at least two graph nodes")
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
    context = context or build_resistance_context(surface.shape, config)
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
