"""Raster encoders and their landscape-distance boundary.

Extracted from the cleaned deepilg architecture. Distance encoders return scores;
observation models determine how those scores relate to genetic targets.
"""

from abc import abstractmethod

import equinox as eqx
import jax

from .common import pixel_to_patch_nodes, squared_embedding_distances


class EmbeddingDistanceModel(eqx.Module):
    """Encode a raster and measure squared Euclidean separation of node embeddings."""

    patch_size: eqx.AbstractVar[int]

    @abstractmethod
    def embedding_grid(self, features, *, inference=True, key=None, patch_batch_size=None):
        """Return an HWC grid of embeddings."""

    def predict_distances(
        self, features, pixel_nodes, *, inference=True, key=None, patch_batch_size=None
    ) -> jax.Array:
        """Return all pairwise squared embedding distances at row-major pixel nodes."""
        grid = self.embedding_grid(
            features, inference=inference, key=key, patch_batch_size=patch_batch_size
        )
        nodes = pixel_to_patch_nodes(
            pixel_nodes, raster_shape=features.shape[:2], patch_size=self.patch_size
        )
        return squared_embedding_distances(grid.reshape(-1, grid.shape[-1])[nodes])
