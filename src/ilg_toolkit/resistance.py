"""Differentiable resistance through JAXScape, with explicit solver failures."""

from dataclasses import dataclass, field
from numbers import Integral

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jax.experimental.sparse import BCOO
from jaxscape import GridGraph, ResistanceDistance
from jaxscape.solvers import AMJaxCGSolver, AMJaxCGSolverState, batched_linear_solve

from ._pairs import PairSelection
from .config import ResistanceSolverConfig


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ResistanceSolverContext:
    """A shape-specific solver with dynamic, reusable preconditioner arrays."""

    graph_shape: tuple[int, int] = field(metadata={"static": True})
    solver: lx.AbstractLinearSolver
    edge_sources: jax.Array
    edge_targets: jax.Array
    laplacian_indices: jax.Array
    laplacian_edges: jax.Array
    laplacian_signs: jax.Array
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
    shape = (int(graph_shape[0]), int(graph_shape[1]))
    if shape[0] * shape[1] < 2:
        raise ValueError("Effective resistance requires at least two graph vertices")
    config = config or ResistanceSolverConfig()
    vertices = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
    sources = np.concatenate((vertices[:, :-1].ravel(), vertices[:-1, :].ravel()))
    targets = np.concatenate((vertices[:, 1:].ravel(), vertices[1:, :].ravel()))
    rows = np.concatenate((sources, targets, sources, targets))
    columns = np.concatenate((sources, targets, targets, sources))
    keep = (rows < vertices.size - 1) & (columns < vertices.size - 1)
    signs = np.repeat([1.0, 1.0, -1.0, -1.0], len(sources))
    solver = (
        AMJaxCGSolver(
            rtol=config.rtol, atol=config.atol, max_steps=config.max_steps, coarse_solver="pinv"
        )
        if config.use_amg
        else lx.CG(rtol=config.rtol, atol=config.atol, max_steps=config.max_steps)
    )
    state = None
    if config.use_amg:
        template = GridGraph(jnp.ones(shape, dtype=jnp.float64), fun=_mean_conductance)
        state = ResistanceDistance(solver=solver).init_preconditioner(template)
    return ResistanceSolverContext(
        graph_shape=shape,
        solver=solver,
        solver_state=state,
        config=config,
        edge_sources=jnp.asarray(sources),
        edge_targets=jnp.asarray(targets),
        laplacian_indices=jnp.asarray(np.column_stack((rows[keep], columns[keep]))),
        laplacian_edges=jnp.asarray(np.tile(np.arange(len(sources)), 4)[keep]),
        laplacian_signs=jnp.asarray(signs[keep]),
    )


def effective_resistance(
    conductance,
    terminal_nodes,
    *,
    context: ResistanceSolverContext | None = None,
    config: ResistanceSolverConfig | None = None,
) -> jax.Array:
    """Return terminal effective resistances for a positive four-neighbour grid.

    Edge conductance is the arithmetic mean of its endpoints. Graph solves use
    float64; resistances and their gradients retain the surface's floating dtype.
    The last graph vertex is grounded and may be a terminal. Coincident terminal
    nodes have zero resistance. Invalid inputs or failed solves raise, including
    under JIT; numerical failure never triggers an implicit solver replacement.
    """
    nodes = jnp.asarray(terminal_nodes)
    if nodes.ndim != 1:
        raise ValueError("terminal_nodes must be a nonempty one-dimensional integer array")
    left, right = jnp.triu_indices(len(nodes), 1)
    values = _effective_resistance_pairs(
        conductance, PairSelection(nodes, left, right), context=context, config=config
    )
    matrix = jnp.zeros((len(nodes), len(nodes)), dtype=values.dtype)
    return matrix.at[left, right].set(values).at[right, left].set(values)


def _effective_resistance_pairs(
    conductance,
    selection: PairSelection,
    *,
    context: ResistanceSolverContext | None = None,
    config: ResistanceSolverConfig | None = None,
) -> jax.Array:
    """Solve required terminals and gather only requested potential differences.

    The full graph remains the resistance domain. For N graph vertices and U
    selected unique terminals, primal potentials and the solve's implicit VJP
    each require O(N*U) storage. No U-by-U distance or N-by-U-by-U product is
    formed. The existing batched solve's custom adjoint preserves independent
    column CG semantics; species-level bounds must also bound these solve states.
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
    nodes = jnp.asarray(selection.nodes)
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
    flat = surface.astype(jnp.float64).ravel()
    weights = _mean_conductance(flat[context.edge_sources], flat[context.edge_targets])
    laplacian = BCOO(
        (weights[context.laplacian_edges] * context.laplacian_signs, context.laplacian_indices),
        shape=(surface.size - 1, surface.size - 1),
    )
    basis = jax.nn.one_hot(nodes, surface.size - 1, dtype=jnp.float64).T
    potentials = batched_linear_solve(laplacian, basis, context.solver, state=context.solver_state)
    # The grounded terminal has a zero RHS, row and potential. Appending that
    # one row lets the same gathers handle it without a terminal-distance matrix.
    potentials = jnp.pad(potentials, ((0, 1), (0, 0)))
    diagonal = potentials[nodes, jnp.arange(len(nodes))]
    left, right = selection.left, selection.right
    values = (
        diagonal[left]
        + diagonal[right]
        - potentials[nodes[left], right]
        - potentials[nodes[right], left]
    )
    return jnp.where(nodes[left] == nodes[right], 0.0, values).astype(surface.dtype)
