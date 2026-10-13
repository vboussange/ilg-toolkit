"""Raster encoders and their ecological-distance boundary.

Extracted from the cleaned deepilg architecture. Distance models return
ecological distances; calibration determines how those distances relate to genetic
targets.
"""

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

from .._pairs import PairSelection
from ..config import ResistanceSolverConfig
from ..resistance import (
    ResistanceSolverContext,
    _effective_resistance_pairs,
    build_resistance_context,
    effective_resistance,
)
from .common import pixel_to_patch_nodes


class AbstractDistanceModel(eqx.Module):
    """Derive ecological distances from features and locations without genetic labels.

    Custom implementations implement ``prepare`` and the private pair kernel,
    accepting the common inference, random-key and preparation arguments.
    Preparation returns ``None`` when no numerical setup is needed; otherwise it
    is local to an operation, outside model parameters and JAX transformations.
    """

    @abstractmethod
    def prepare(self, features: ArrayLike) -> ResistanceSolverContext | None:
        """Prepare shape-dependent numerical work before predicting distances, or return none."""

    @abstractmethod
    def _pair_distances(
        self,
        features: ArrayLike,
        selection: PairSelection,
        *,
        context: ResistanceSolverContext | None = None,
        inference: bool = True,
        key: jax.Array | None = None,
        patch_batch_size: int | None = None,
    ) -> jax.Array:
        """Return one distance per prepared pair, without a full output matrix."""

    def _distance_nodes(self, features: ArrayLike, pixel_nodes: ArrayLike) -> jax.Array:
        return jnp.asarray(pixel_nodes, dtype=jnp.int32)

    def _prepare_pairs(self, features, pixel_nodes, pairs) -> PairSelection:
        """Deduplicate requested distance nodes on the host, before differentiation."""
        nodes = np.asarray(self._distance_nodes(features, pixel_nodes))
        endpoints = np.stack(pairs, axis=0)
        selected, inverse = np.unique(nodes[endpoints], return_inverse=True)
        left, right = inverse.reshape(endpoints.shape)
        return PairSelection(jnp.asarray(selected), jnp.asarray(left), jnp.asarray(right))

    def predict_distances(
        self,
        features,
        pixel_nodes,
        *,
        context=None,
        inference=True,
        key=None,
        patch_batch_size=None,
    ) -> jax.Array:
        """Explicit full-matrix diagnostic; ordinary workflows use prepared pairs."""
        nodes = self._distance_nodes(features, pixel_nodes)
        left, right = jnp.triu_indices(len(nodes), 1)
        values = self._pair_distances(
            features,
            PairSelection(nodes, left, right),
            context=context,
            inference=inference,
            key=key,
            patch_batch_size=patch_batch_size,
        )
        matrix = jnp.zeros((len(nodes), len(nodes)), dtype=values.dtype)
        return matrix.at[left, right].set(values).at[right, left].set(values)


class ConductanceModel(AbstractDistanceModel):
    """Encode positive patch conductance and measure graph resistance.

    Initial encoders use stateless GroupNorm; their entire Equinox model is
    retained by training and prediction. ``inference`` and ``key`` provide the
    common distance model signature and are unused by this deterministic family.
    Final implementations declare ``patch_size`` and immutable ``solver_config``
    as static fields alongside their other fields and initialization.
    """

    patch_size: eqx.AbstractVar[int]
    solver_config: eqx.AbstractVar[ResistanceSolverConfig]

    def prepare(self, features: ArrayLike) -> ResistanceSolverContext:
        """Build the model's solver for this post-patch graph outside JIT/grad."""
        shape = jnp.asarray(features).shape
        if len(shape) != 3 or shape[0] % self.patch_size or shape[1] % self.patch_size:
            raise ValueError("Raster dimensions must be divisible by model patch_size")
        return build_resistance_context(
            (shape[0] // self.patch_size, shape[1] // self.patch_size), self.solver_config
        )

    @abstractmethod
    def conductance(self, features: jax.Array, *, patch_batch_size: int | None = None) -> jax.Array:
        """Return a positive patch-grid surface from row, column, feature data."""

    def resistance_and_conductance(
        self,
        features: ArrayLike,
        pixel_nodes: ArrayLike,
        *,
        context: ResistanceSolverContext | None = None,
        patch_batch_size: int | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Return effective resistances and the conductance surface that induced them."""
        features = jnp.asarray(features)
        if context is None:
            context = self.prepare(features)
        if context.config != self.solver_config:
            raise ValueError("Solver context policy does not match the conductance model settings")
        surface = self.conductance(features, patch_batch_size=patch_batch_size)
        nodes = pixel_to_patch_nodes(
            pixel_nodes,
            raster_shape=(features.shape[0], features.shape[1]),
            patch_size=self.patch_size,
        )
        return effective_resistance(surface, nodes, context=context), surface

    def _distance_nodes(self, features: ArrayLike, pixel_nodes: ArrayLike) -> jax.Array:
        shape = jnp.asarray(features).shape
        return pixel_to_patch_nodes(
            pixel_nodes, raster_shape=(shape[0], shape[1]), patch_size=self.patch_size
        )

    def _pair_distances(
        self,
        features: ArrayLike,
        selection: PairSelection,
        *,
        context: ResistanceSolverContext | None = None,
        inference: bool = True,
        key: jax.Array | None = None,
        patch_batch_size: int | None = None,
    ) -> jax.Array:
        """Return exact requested resistances on the complete regional graph."""
        features = jnp.asarray(features)
        context = self.prepare(features) if context is None else context
        if context.config != self.solver_config:
            raise ValueError("Solver context policy does not match the conductance model settings")
        surface = self.conductance(features, patch_batch_size=patch_batch_size)
        return _effective_resistance_pairs(surface, selection, context=context)


class EmbeddingDistanceModel(AbstractDistanceModel):
    """Encode a raster and measure squared Euclidean separation of node embeddings."""

    patch_size: eqx.AbstractVar[int]

    def prepare(self, features: ArrayLike) -> None:
        """Embedding distances need no shape-dependent numerical preparation."""
        return None

    @abstractmethod
    def embedding_grid(
        self,
        features: jax.Array,
        *,
        inference: bool = True,
        key: jax.Array | None = None,
        patch_batch_size: int | None = None,
    ) -> jax.Array:
        """Return embeddings with row, column and embedding dimensions."""

    def _distance_nodes(self, features: ArrayLike, pixel_nodes: ArrayLike) -> jax.Array:
        shape = jnp.asarray(features).shape
        return pixel_to_patch_nodes(
            pixel_nodes, raster_shape=(shape[0], shape[1]), patch_size=self.patch_size
        )

    def _pair_distances(
        self,
        features: ArrayLike,
        selection: PairSelection,
        *,
        context: ResistanceSolverContext | None = None,
        inference: bool = True,
        key: jax.Array | None = None,
        patch_batch_size: int | None = None,
    ) -> jax.Array:
        """Gather selected endpoints before calculating squared separations."""
        features = jnp.asarray(features)
        grid = self.embedding_grid(
            features, inference=inference, key=key, patch_batch_size=patch_batch_size
        )
        endpoints = grid.reshape(-1, grid.shape[-1])[selection.nodes]
        return jnp.sum(jnp.square(endpoints[selection.left] - endpoints[selection.right]), axis=-1)
