"""Differentiable effective-resistance layer on conductance grid graphs using JAXScape."""

import math
from dataclasses import dataclass, field, replace
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxscape import GridGraph, ResistanceDistance
from jaxscape.solvers import AMJaxCGSolver, AMJaxCGSolverState, CuDSSSolver, SparseDirectState

from .config import ResistanceSolverConfig


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _ResistanceSolverContext:
    """A resistance solver for one graph shape, with its solver state.

    Attributes:
        graph_shape: Shape of the conductance grid graphs the context solves.
        solver: The resistance solver.
        solver_state: ``None`` for conjugate gradient, the fixed hierarchy for
            algebraic multigrid, and the state of the latest solve for the NVIDIA
            CUDA Direct Sparse Solver (cuDSS).
        carries_state: Whether a solve returns a refreshed ``solver_state``, which
            the next solve takes. True only for cuDSS.
    """

    graph_shape: tuple[int, ...] = field(metadata={"static": True})
    solver: lx.AbstractLinearSolver
    solver_state: AMJaxCGSolverState | SparseDirectState | None = None
    carries_state: bool = field(default=False, metadata={"static": True})


def _carried_state(context: _ResistanceSolverContext | None) -> SparseDirectState | None:
    """Returns the solver state a solve refreshes, or ``None`` for a context without one."""
    if context is None or not context.carries_state:
        return None
    return cast(SparseDirectState, context.solver_state)


def _with_carried_state(
    context: _ResistanceSolverContext | None, state: SparseDirectState | None
) -> _ResistanceSolverContext | None:
    """Returns the context holding a carried solver state; other contexts stay as they are."""
    if context is None or not context.carries_state:
        return context
    return replace(context, solver_state=state)


def _mean_conductance(left, right):
    """Returns the conductance of an edge: the mean of its two vertices' conductances."""
    return 0.5 * (left + right)


def _cudss_solver() -> lx.AbstractLinearSolver:
    """Returns the cuDSS solver, or raises an error of the same type naming the ``cudss`` extra."""
    try:
        return CuDSSSolver()
    except (ImportError, RuntimeError) as error:
        raise type(error)(
            "ResistanceSolverConfig(solver='cudss') requires an NVIDIA GPU with CUDA-enabled "
            f"JAX and the cudss extra, pip install 'ilg-toolkit[cudss]': {error}"
        ) from error


def _resistance_context(
    graph_shape: tuple[int, ...], config: ResistanceSolverConfig
) -> _ResistanceSolverContext:
    """Prepares a resistance solver for conductance grid graphs of one shape.

    Conjugate gradient needs no preparation. Algebraic multigrid and cuDSS prepare
    their solver state on a uniform conductance grid graph. Preparation runs
    eagerly, also inside traced functions.

    Args:
        graph_shape: Shape of the conductance grid graphs, ``(rows, columns)``.
        config: Solver settings.

    Returns:
        The context to pass to ``_effective_resistance``.

    Raises:
        ValueError: If ``graph_shape`` is not a 2D shape with at least two vertices.
        ImportError: If ``config`` selects cuDSS and spineax cannot be loaded.
        RuntimeError: If ``config`` selects cuDSS without a CUDA-enabled GPU backend.
    """
    if len(graph_shape) != 2 or math.prod(graph_shape) < 2:
        raise ValueError("Effective resistance requires a 2D surface with at least two vertices")
    if config.solver == "cudss":
        solver = _cudss_solver()
    else:
        assert config.rtol is not None and config.atol is not None and config.max_steps is not None
        if config.solver == "cg":
            return _ResistanceSolverContext(
                graph_shape,
                lx.CG(rtol=config.rtol, atol=config.atol, max_steps=config.max_steps),
            )
        solver = AMJaxCGSolver(
            rtol=config.rtol, atol=config.atol, max_steps=config.max_steps, coarse_solver="pinv"
        )
    with jax.ensure_compile_time_eval():
        template = GridGraph(jnp.ones(graph_shape, dtype=jnp.float64), fun=_mean_conductance)
        state = ResistanceDistance(solver=solver).init(template)
    return _ResistanceSolverContext(
        graph_shape, solver, state, carries_state=config.solver == "cudss"
    )


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

    With a carried solver state, the returned context holds the state refreshed for
    ``conductance``. Passing it to the next call keeps the resistance solver's
    analysis; passing an earlier state stays correct and repeats the analysis.

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
    graph = GridGraph(surface.astype(jnp.float64), fun=_mean_conductance)
    distance = ResistanceDistance(solver=context.solver)
    if context.carries_state:
        context = replace(context, solver_state=distance.reinit(graph, context.solver_state))
    resistances = distance.nodes_to_nodes_distance(graph, nodes, state=context.solver_state)
    return resistances.at[jnp.diag_indices(nodes.size)].set(0), context
