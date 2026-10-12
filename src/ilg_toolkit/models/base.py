"""Raster encoders and their landscape-distance boundary.

Extracted from the cleaned deepilg architecture. Distance encoders return scores;
observation models determine how those scores relate to genetic targets.
"""

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
from numpy.typing import ArrayLike

from ..config import ResistanceSolverConfig
from ..resistance import ResistanceSolverContext, build_resistance_context, effective_resistance
from .common import pixel_to_patch_nodes, squared_embedding_distances


class AbstractDistanceModel(eqx.Module):
    """Derive landscape separation from features and locations without genetic labels.

    Custom implementations implement ``predict_distances`` and accept the common
    inference, random-key and preparation arguments. Preparation is local to an
    operation and remains outside model parameters and JAX transformations.
    """

    def prepare(self, features: ArrayLike) -> ResistanceSolverContext | None:
        """Prepare shape-dependent numerical work before scoring; default to none."""
        return None

    @abstractmethod
    def predict_distances(
        self,
        features: ArrayLike,
        pixel_nodes: ArrayLike,
        *,
        context: ResistanceSolverContext | None = None,
        inference: bool = True,
        key: jax.Array | None = None,
        patch_batch_size: int | None = None,
    ) -> jax.Array:
        """Return the square landscape-score matrix at row-major pixel locations."""


class ConductanceModel(AbstractDistanceModel):
    """Encode positive patch conductance and measure graph resistance.

    Initial encoders use stateless GroupNorm; their entire Equinox model is
    retained by training and prediction. ``inference`` and ``key`` provide the
    common distance model signature and are unused by this deterministic family.
    """

    patch_size: eqx.AbstractVar[int]
    solver_config: ResistanceSolverConfig = eqx.field(
        static=True, kw_only=True, default=ResistanceSolverConfig()
    )

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
        """Return landscape scores and the conductance surface that induced them."""
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

    def predict_distances(
        self,
        features: ArrayLike,
        pixel_nodes: ArrayLike,
        *,
        context: ResistanceSolverContext | None = None,
        inference: bool = True,
        key: jax.Array | None = None,
        patch_batch_size: int | None = None,
    ) -> jax.Array:
        """Return all pairwise resistance scores independently of genetic labels."""
        scores, _ = self.resistance_and_conductance(
            features, pixel_nodes, context=context, patch_batch_size=patch_batch_size
        )
        return scores


class EmbeddingDistanceModel(AbstractDistanceModel):
    """Encode a raster and measure squared Euclidean separation of node embeddings."""

    patch_size: eqx.AbstractVar[int]

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

    def predict_distances(
        self,
        features: ArrayLike,
        pixel_nodes: ArrayLike,
        *,
        context: ResistanceSolverContext | None = None,
        inference: bool = True,
        key: jax.Array | None = None,
        patch_batch_size: int | None = None,
    ) -> jax.Array:
        """Return all pairwise squared embedding distances at row-major pixel nodes."""
        features = jnp.asarray(features)
        grid = self.embedding_grid(
            features, inference=inference, key=key, patch_batch_size=patch_batch_size
        )
        nodes = pixel_to_patch_nodes(
            pixel_nodes,
            raster_shape=(features.shape[0], features.shape[1]),
            patch_size=self.patch_size,
        )
        return squared_embedding_distances(grid.reshape(-1, grid.shape[-1])[nodes])
