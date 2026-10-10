"""Raster encoders and their landscape-distance boundary.

Extracted from the cleaned deepilg architecture. Distance encoders return scores;
observation models determine how those scores relate to genetic targets.
"""

from abc import abstractmethod

import equinox as eqx
import jax

from ..resistance import ResistanceSolverContext, effective_resistance
from .common import pixel_to_patch_nodes, squared_embedding_distances


class ConductanceModel(eqx.Module):
    """Encode positive patch conductance and measure graph resistance.

    Initial encoders use stateless GroupNorm; their entire Equinox model is
    retained by training and prediction. ``inference`` and ``key`` provide the
    common distance model signature and are unused by this deterministic family.
    """

    patch_size: eqx.AbstractVar[int]

    @abstractmethod
    def conductance(self, features: jax.Array, *, patch_batch_size: int | None = None) -> jax.Array:
        """Return a positive 2D patch-grid surface for one HWC feature raster."""

    def resistance_and_conductance(
        self,
        features: jax.Array,
        pixel_nodes: jax.Array,
        *,
        context: ResistanceSolverContext | None = None,
        patch_batch_size: int | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Return landscape scores and the conductance surface that induced them."""
        surface = self.conductance(features, patch_batch_size=patch_batch_size)
        nodes = pixel_to_patch_nodes(
            pixel_nodes,
            raster_shape=(features.shape[0], features.shape[1]),
            patch_size=self.patch_size,
        )
        return effective_resistance(surface, nodes, context=context), surface

    def predict_distances(
        self,
        features: jax.Array,
        pixel_nodes: jax.Array,
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


class EmbeddingDistanceModel(eqx.Module):
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
        """Return an HWC grid of embeddings."""

    def predict_distances(
        self,
        features: jax.Array,
        pixel_nodes: jax.Array,
        *,
        inference: bool = True,
        key: jax.Array | None = None,
        patch_batch_size: int | None = None,
    ) -> jax.Array:
        """Return all pairwise squared embedding distances at row-major pixel nodes."""
        grid = self.embedding_grid(
            features, inference=inference, key=key, patch_batch_size=patch_batch_size
        )
        nodes = pixel_to_patch_nodes(
            pixel_nodes,
            raster_shape=(features.shape[0], features.shape[1]),
            patch_size=self.patch_size,
        )
        return squared_embedding_distances(grid.reshape(-1, grid.shape[-1])[nodes])
