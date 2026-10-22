"""Distance models with pooled backbones and species-specific relationships."""

from abc import abstractmethod
from collections.abc import Sequence
from itertools import combinations
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
from numpy.typing import ArrayLike

from ..config import ResistanceSolverConfig
from ..data import ObservationPartition, RegionBatch
from ..prediction import Prediction
from ..resistance import ResistanceSolverContext, build_resistance_context, effective_resistance
from .common import backbone_with_randomness, pixel_to_patch_nodes

# Sampling-unit nodes, followed by pairs indexing those sampling units.
type DistancePairs = tuple[jax.Array, tuple[jax.Array, jax.Array]]


class AbstractDistanceModel(eqx.Module):
    """A pooled backbone composed with a relationship deriving ecological distances.

    Custom models implement ``backbone`` and ``ecological_distances`` and declare
    static ``num_species`` and ``patch_size`` fields, both defaulting to one.
    """

    num_species: eqx.AbstractVar[int]
    patch_size: eqx.AbstractVar[int]

    @abstractmethod
    def backbone(self, features: jax.Array) -> jax.Array:
        """Pooled landscape features derived from a row, column, feature raster."""

    @abstractmethod
    def ecological_distances(
        self, latent: jax.Array, pairs: DistancePairs, species_index: int
    ) -> jax.Array:
        """Ecological distances aligned with pairs of sampling-unit indices.

        ``pairs`` contains the sampling-unit graph nodes and two arrays indexing
        those nodes, one for each sampling unit in a pair.
        """

    def predict(
        self,
        region: RegionBatch,
        *,
        species: str | None = None,
        pairs: Sequence[tuple[str, str]] | None = None,
    ) -> Prediction:
        """Ecological distances for query pairs in a single-species region batch.

        Omitted pairs request every distinct pair in sampling-unit order.
        """
        _single_species(self, species)
        labels, indices = _query_pairs(region, pairs)
        context = prepare_distance_context(self, region.feature_array)
        values = distance_values(
            self, region.feature_array, region.pixel_nodes, indices, context=context
        )
        values = eqx.error_if(
            values,
            jnp.any(~jnp.isfinite(values) | (values < 0)),
            "Distance model produced nonfinite or negative ecological distances",
        )
        return Prediction(values, labels, region.name, region.sampling_unit_ids)


class ConductanceModel(AbstractDistanceModel):
    """A distance model deriving effective resistance from a conductance surface.

    Custom models implement ``backbone`` and ``conductance`` and declare a static
    ``solver_config`` alongside their other model fields.
    """

    solver_config: eqx.AbstractVar[ResistanceSolverConfig]

    @abstractmethod
    def conductance(self, latent: jax.Array, species_index: int) -> jax.Array:
        """Positive conductance surface for one species from pooled landscape features."""

    def ecological_distances(
        self,
        latent: jax.Array,
        pairs: DistancePairs,
        species_index: int,
        *,
        context: ResistanceSolverContext | None = None,
    ) -> jax.Array:
        """Effective resistance aligned with the requested sampling-unit pairs."""
        nodes, indices = pairs
        surface = self.conductance(latent, species_index)
        if context is not None and context.config != self.solver_config:
            raise ValueError("Solver context policy does not match the conductance model settings")
        context = context or build_resistance_context(
            (surface.shape[0], surface.shape[1]), self.solver_config
        )
        return effective_resistance(surface, nodes, context=context)[indices]

    def conductance_surface(self, region: RegionBatch, *, species: str | None = None) -> jax.Array:
        """Positive conductance surface for a single-species region batch."""
        _single_species(self, species)
        surface = self.conductance(self.backbone(region.feature_array), 0)
        return eqx.error_if(
            surface,
            jnp.any(~jnp.isfinite(surface) | (surface <= 0)),
            "Distance model produced nonfinite or nonpositive conductance",
        )


class EmbeddingModel(AbstractDistanceModel):
    """A distance model deriving squared embedding distances from landscape features."""

    @abstractmethod
    def embedding(self, latent: jax.Array, species_index: int) -> jax.Array:
        """Embedding grid for one species from pooled landscape features."""

    def ecological_distances(
        self, latent: jax.Array, pairs: DistancePairs, species_index: int
    ) -> jax.Array:
        """Squared embedding distances aligned with the requested sampling-unit pairs."""
        nodes, (left, right) = pairs
        grid = self.embedding(latent, species_index)
        embeddings = grid.reshape(-1, grid.shape[-1])[nodes]
        return jnp.sum(jnp.square(embeddings[left] - embeddings[right]), axis=-1)


def _single_species(model: AbstractDistanceModel, species: str | None) -> None:
    if model.num_species != 1 or species is not None:
        raise ValueError("This workflow requires one unnamed species")


def _query_pairs(
    region: RegionBatch, pairs: Sequence[tuple[str, str]] | None
) -> tuple[tuple[tuple[str, str], ...], tuple[jax.Array, jax.Array]]:
    raw = tuple(combinations(region.sampling_unit_ids, 2)) if pairs is None else tuple(pairs)
    try:
        if any(isinstance(pair, str) for pair in raw):
            raise ValueError("Each pair requires two labels")
        labels = cast(tuple[tuple[str, str], ...], tuple(tuple(pair) for pair in raw))
        ObservationPartition(region.name, labels, role="query")
        lookup = {label: index for index, label in enumerate(region.sampling_unit_ids)}
        indices = jnp.asarray([(lookup[a], lookup[b]) for a, b in labels], dtype=jnp.int32)
    except (TypeError, ValueError, KeyError) as error:
        raise ValueError(
            "Query pairs require unique distinct sampling units in the region"
        ) from error
    return labels, (indices[:, 0], indices[:, 1])


def prepare_distance_context(
    model: AbstractDistanceModel, features: ArrayLike
) -> ResistanceSolverContext | None:
    """Reusable resistance solver for the region's conductance graph."""
    if not isinstance(model, ConductanceModel):
        return None
    shape = jnp.asarray(features).shape
    if len(shape) != 3 or shape[0] % model.patch_size or shape[1] % model.patch_size:
        raise ValueError("Raster dimensions must be divisible by model patch_size")
    return build_resistance_context(
        (shape[0] // model.patch_size, shape[1] // model.patch_size), model.solver_config
    )


def distance_values(
    model: AbstractDistanceModel,
    features: jax.Array,
    pixel_nodes: jax.Array,
    indices: tuple[jax.Array, jax.Array],
    *,
    context: ResistanceSolverContext | None = None,
    inference: bool = True,
    key: jax.Array | None = None,
) -> jax.Array:
    """Ecological distances for located sampling-unit pairs."""
    latent = backbone_with_randomness(model, features, inference=inference, key=key)
    nodes = pixel_to_patch_nodes(
        pixel_nodes,
        raster_shape=(features.shape[0], features.shape[1]),
        patch_size=model.patch_size,
    )
    pairs = (nodes, indices)
    if isinstance(model, ConductanceModel):
        return model.ecological_distances(latent, pairs, 0, context=context)
    return model.ecological_distances(latent, pairs, 0)
