"""Distance models with pooled backbones and species-specific relationships."""

from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import replace
from itertools import combinations
from numbers import Integral
from typing import cast, overload

import equinox as eqx
import jax
import jax.numpy as jnp
from numpy.typing import ArrayLike

from ..config import ResistanceSolverConfig
from ..data import ObservationPartition, RegionBatch
from ..prediction import Prediction
from ..resistance import ResistanceSolverContext, build_resistance_context, effective_resistance
from .common import pixel_to_patch_nodes, with_dropout_mode

# Sampling-unit nodes, followed by pairs indexing those sampling units.
type DistancePairs = tuple[jax.Array, tuple[jax.Array, jax.Array]]


class AbstractDistanceModel(eqx.Module):
    """Abstract distance model combining a pooled backbone with an ecological-distance relationship.

    Custom models implement ``backbone`` and ``ecological_distances`` and supply
    static ``num_species`` and ``patch_size`` fields.
    """

    num_species: eqx.AbstractVar[int]
    patch_size: eqx.AbstractVar[int]

    @abstractmethod
    def backbone(self, features: jax.Array) -> jax.Array:
        """Pooled landscape features derived from a row, column, feature raster."""

    @abstractmethod
    def ecological_distances(
        self, latent: jax.Array, pairs: DistancePairs, species_index: int | jax.Array
    ) -> jax.Array:
        """Ecological distances aligned with pairs of sampling-unit indices.

        ``pairs`` contains the sampling-unit graph nodes and two arrays indexing
        those nodes, one for each sampling unit in a pair.
        """

    @overload
    def predict(
        self,
        region: RegionBatch,
        *,
        species: str | None = None,
        species_order: Sequence[str] | None = None,
        pairs: Sequence[tuple[str, str]] | None = None,
    ) -> Prediction: ...

    @overload
    def predict(
        self,
        region: RegionBatch,
        *,
        species: Sequence[str],
        species_order: Sequence[str] | None = None,
        pairs: Sequence[tuple[str, str]] | None = None,
    ) -> dict[str, Prediction]: ...

    def predict(
        self,
        region: RegionBatch,
        *,
        species: str | Sequence[str] | None = None,
        species_order: Sequence[str] | None = None,
        pairs: Sequence[tuple[str, str]] | None = None,
    ) -> Prediction | dict[str, Prediction]:
        """Ecological distances for query pairs of the selected species.

        Omitted pairs request every distinct pair in sampling-unit order.
        ``species_order`` associates names with parameter rows for named species.
        A species sequence returns a mapping and shares the pooled backbone.
        """
        order = validate_species_order(species_order, self.num_species)
        selected = select_species(order, species)
        if self.num_species > 1 and region.species is None:
            raise ValueError("Multispecies queries require sampling-unit species labels")
        model = with_dropout_mode(self, inference=True, key=None)
        context = prepare_distance_context(model, region.feature_array)
        latent = model.backbone(region.feature_array)
        results = {}
        for name, index in selected:
            prepared = region_for_species(region, name)
            labels, values = query_distances_from_latent(
                model,
                latent,
                prepared,
                pairs,
                species_index=index,
                context=context,
            )
            results[name] = Prediction(
                values, labels, region.name, prepared.sampling_unit_ids, name
            )
        if isinstance(species, str) or species is None:
            return results[selected[0][0]]
        return cast(dict[str, Prediction], results)


class ConductanceModel(AbstractDistanceModel):
    """A distance model deriving effective resistance from a conductance surface.

    Custom models implement ``backbone`` and ``conductance`` and declare a static
    ``solver_config`` alongside their other model fields.
    """

    solver_config: eqx.AbstractVar[ResistanceSolverConfig]

    @abstractmethod
    def conductance(self, latent: jax.Array, species_index: int | jax.Array) -> jax.Array:
        """Positive conductance surface for one species from pooled landscape features."""

    def ecological_distances(
        self,
        latent: jax.Array,
        pairs: DistancePairs,
        species_index: int | jax.Array,
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

    @overload
    def conductance_surface(
        self,
        region: RegionBatch,
        *,
        species: str | None = None,
        species_order: Sequence[str] | None = None,
    ) -> jax.Array: ...

    @overload
    def conductance_surface(
        self,
        region: RegionBatch,
        *,
        species: Sequence[str],
        species_order: Sequence[str] | None = None,
    ) -> dict[str, jax.Array]: ...

    def conductance_surface(
        self,
        region: RegionBatch,
        *,
        species: str | Sequence[str] | None = None,
        species_order: Sequence[str] | None = None,
    ) -> jax.Array | dict[str, jax.Array]:
        """Positive conductance surfaces for the selected species.

        ``species_order`` associates names with parameter rows for named species.
        A species sequence returns a mapping and shares the pooled backbone.
        """
        order = validate_species_order(species_order, self.num_species)
        selected = select_species(order, species)
        model = with_dropout_mode(self, inference=True, key=None)
        latent = model.backbone(region.feature_array)
        results = {}
        for name, index in selected:
            surface = model.conductance(latent, index)
            results[name] = eqx.error_if(
                surface,
                jnp.any(~jnp.isfinite(surface) | (surface <= 0)),
                "Distance model produced nonfinite or nonpositive conductance",
            )
        if isinstance(species, str) or species is None:
            return results[selected[0][0]]
        return cast(dict[str, jax.Array], results)


class EmbeddingModel(AbstractDistanceModel):
    """A distance model deriving squared embedding distances from landscape features."""

    @abstractmethod
    def embedding(self, latent: jax.Array, species_index: int | jax.Array) -> jax.Array:
        """Embedding grid for one species from pooled landscape features."""

    def ecological_distances(
        self, latent: jax.Array, pairs: DistancePairs, species_index: int | jax.Array
    ) -> jax.Array:
        """Squared embedding distances aligned with the requested sampling-unit pairs."""
        nodes, (left, right) = pairs
        grid = self.embedding(latent, species_index)
        embeddings = grid.reshape(-1, grid.shape[-1])[nodes]
        return jnp.sum(jnp.square(embeddings[left] - embeddings[right]), axis=-1)


def validate_species_order(order: Sequence[str] | None, num_species: int) -> tuple[str, ...] | None:
    """Species names associated with the distance model's parameter rows."""
    if isinstance(num_species, bool) or not isinstance(num_species, Integral) or num_species < 1:
        raise ValueError("num_species must be a positive integer")
    if order is None:
        if num_species != 1:
            raise ValueError("Multispecies models require a species order")
        return None
    if isinstance(order, (str, bytes)) or not isinstance(order, Sequence):
        raise ValueError("Species order must be a sequence of names")
    names = tuple(order)
    if (
        len(names) != num_species
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("Species order must contain one unique name per modeled species")
    return names


def select_species(
    order: tuple[str, ...] | None, selection: str | Sequence[str] | None
) -> tuple[tuple[str | None, int], ...]:
    """Selected species names and their parameter row indices."""
    if selection is None:
        if order is not None and len(order) != 1:
            raise ValueError("Multispecies queries require a species selection")
        return ((None if order is None else order[0], 0),)
    names = (selection,) if isinstance(selection, str) else selection
    if (
        isinstance(names, (str, bytes))
        or not isinstance(names, Sequence)
        or not names
        or order is None
        or any(not isinstance(name, str) or name not in order for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("Species selection must contain unique names from the model's order")
    return tuple((name, order.index(name)) for name in names)


def region_for_species(region: RegionBatch, name: str | None) -> RegionBatch:
    """Region batch containing one species' sampling units."""
    if region.species is None:
        return region
    indices = tuple(index for index, species in enumerate(region.species) if species == name)
    if len(indices) < 2:
        raise ValueError("A species query requires at least two sampling units in the region")
    kinds = region.sampling_unit_kinds
    assert kinds is not None
    return replace(
        region,
        sampling_unit_ids=tuple(region.sampling_unit_ids[index] for index in indices),
        grid_positions=jnp.asarray(region.grid_positions)[jnp.asarray(indices)],
        sampling_unit_kinds=tuple(kinds[index] for index in indices),
        species=None,
    )


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


def query_distances_from_latent(
    model: AbstractDistanceModel,
    latent: jax.Array,
    region: RegionBatch,
    pairs: Sequence[tuple[str, str]] | None,
    *,
    species_index: int | jax.Array,
    context: ResistanceSolverContext | None,
) -> tuple[tuple[tuple[str, str], ...], jax.Array]:
    """Query pair labels and valid ecological distances for one species."""
    labels, indices = _query_pairs(region, pairs)
    values = distances_from_latent(
        model,
        latent,
        region.pixel_nodes,
        indices,
        raster_shape=(region.feature_array.shape[0], region.feature_array.shape[1]),
        species_index=species_index,
        context=context,
    )
    values = eqx.error_if(
        values,
        jnp.any(~jnp.isfinite(values) | (values < 0)),
        "Distance model produced nonfinite or negative ecological distances",
    )
    return labels, values


def distance_values(
    model: AbstractDistanceModel,
    features: jax.Array,
    pixel_nodes: jax.Array,
    indices: tuple[jax.Array, jax.Array],
    *,
    context: ResistanceSolverContext | None = None,
    inference: bool = True,
    key: jax.Array | None = None,
    species_index: int | jax.Array = 0,
) -> jax.Array:
    """Ecological distances for located sampling-unit pairs."""
    model = with_dropout_mode(model, inference=inference, key=key)
    latent = model.backbone(features)
    return distances_from_latent(
        model,
        latent,
        pixel_nodes,
        indices,
        raster_shape=(features.shape[0], features.shape[1]),
        species_index=species_index,
        context=context,
    )


def distances_from_latent(
    model: AbstractDistanceModel,
    latent: jax.Array,
    pixel_nodes: jax.Array,
    indices: tuple[jax.Array, jax.Array],
    *,
    raster_shape: tuple[int, int],
    species_index: int | jax.Array = 0,
    context: ResistanceSolverContext | None = None,
) -> jax.Array:
    """Ecological distances from pooled landscape features for one species."""
    nodes = pixel_to_patch_nodes(
        pixel_nodes,
        raster_shape=raster_shape,
        patch_size=model.patch_size,
    )
    pairs = (nodes, indices)
    if isinstance(model, ConductanceModel):
        return model.ecological_distances(latent, pairs, species_index, context=context)
    return model.ecological_distances(latent, pairs, species_index)
