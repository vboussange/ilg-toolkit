"""Abstract distance models deriving ecological distances from patch features."""

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
from ..data import _validate_pairs, RegionBatch
from ..prediction import Prediction
from ..resistance import build_resistance_context, effective_resistance, ResistanceSolverContext
from .common import pixel_to_patch_nodes, with_dropout_mode


# Sampling-unit vertices, followed by pairs indexing those sampling units.
type DistancePairs = tuple[jax.Array, tuple[jax.Array, jax.Array]]


class AbstractDistanceModel(eqx.Module):
    """Abstract distance model deriving ecological distances from landscape features.

    A distance model maps a region's landscape features to ecological distances
    between its sampling units, without genetic observations. ``backbone`` computes
    patch features shared by every species, and ``ecological_distances`` turns one
    species' patch features into distances for requested pairs. ``train`` estimates
    one set of parameters from every supplied region (complete pooling).

    Write a custom distance model as a final concrete subclass of one abstract
    class, and implement its abstract methods:

    - ``ConductanceModel``: ``backbone(features)`` and
      ``conductance(latent, species_index)``, returning a positive conductance grid.
    - ``EmbeddingDistanceModel``: ``backbone(features)`` and
      ``embedding(latent, species_index)``, returning a grid with the embedding
      dimension last.
    - ``AbstractDistanceModel``: ``backbone(features)`` and
      ``ecological_distances(latent, pairs, species_index)``, returning ecological
      distances aligned with ``pairs``.

    Declare static ``num_species`` and ``patch_size`` fields; conductance models
    also declare a static ``solver_config`` holding a ``ResistanceSolverConfig``.
    Keep fields and initialization together in the concrete class, and compose
    existing concrete models to reuse their behavior rather than subclassing them.

    Raster features have shape ``(rows, columns, features)``, with meanings and
    order compatible across regions. Sampling locations are integer
    ``(row, column)`` cells mapped to the vertex of their containing patch. During
    training, ``species_index`` may be a scalar integer array, so select species
    parameters by array indexing. The inherited ``predict`` returns a
    ``Prediction``, and ``train`` estimates the array-valued parameters with the
    chosen loss. See the experimental [multispecies example](../examples/multispecies.md)
    for species labels and parameter order.

    Examples:
        A single-species embedding-distance model with a trainable linear feature map:

        >>> @final
        ... class LinearEmbedding(EmbeddingDistanceModel):
        ...     weights: jax.Array
        ...     num_species: int = eqx.field(static=True, default=1)
        ...     patch_size: int = eqx.field(static=True, default=1)
        ...
        ...     def backbone(self, features: jax.Array) -> jax.Array:
        ...         return features @ self.weights
        ...
        ...     def embedding(self, latent: jax.Array, species_index) -> jax.Array:
        ...         return latent
        >>> distance_model = LinearEmbedding(jnp.eye(2))
    """

    num_species: eqx.AbstractVar[int]
    patch_size: eqx.AbstractVar[int]

    @abstractmethod
    def backbone(self, features: jax.Array) -> jax.Array:
        """Patch features, shared by every species, from a row, column, feature raster."""

    @abstractmethod
    def ecological_distances(
        self, latent: jax.Array, pairs: DistancePairs, species_index: int | jax.Array
    ) -> jax.Array:
        """Ecological distances aligned with pairs of sampling-unit indices.

        ``pairs`` is ``(vertices, (left, right))``: ``vertices`` locates sampling
        units on the flattened patch grid, and ``left`` and ``right`` index those
        sampling units, one array for each end of a pair.
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
        A species sequence returns a mapping and computes patch features once.
        Explicit ``pairs`` limit the returned vector and avoid generating all
        pair labels; this does not guarantee that an underlying resistance solver
        avoids its distance matrix among sampling-unit vertices. Calling ``as_matrix()`` on the
        result explicitly allocates a sampling-unit-square array.
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
    """A patch encoder whose conductance surface forms a conductance grid graph.

    Each patch is one vertex with a positive conductance, and each edge between
    neighboring vertices averages their conductances. The effective-resistance
    layer returns the effective resistance between sampling-unit vertices as
    ecological distance. Custom models implement ``backbone`` and ``conductance``
    and declare a static ``solver_config`` alongside their other model fields.
    """

    solver_config: eqx.AbstractVar[ResistanceSolverConfig]

    @abstractmethod
    def conductance(self, latent: jax.Array, species_index: int | jax.Array) -> jax.Array:
        """Positive conductance surface for one species from patch features."""

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
        A species sequence returns a mapping and computes patch features once.
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


class EmbeddingDistanceModel(AbstractDistanceModel):
    """A distance model deriving squared embedding distances from landscape features."""

    @abstractmethod
    def embedding(self, latent: jax.Array, species_index: int | jax.Array) -> jax.Array:
        """Embedding grid for one species from patch features."""

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
        labels = _validate_pairs(raw)
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
    """Reusable resistance solver for the region's conductance grid graph."""
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
    """Ecological distances from patch features for one species."""
    nodes = pixel_to_patch_nodes(
        pixel_nodes,
        raster_shape=raster_shape,
        patch_size=model.patch_size,
    )
    pairs = (nodes, indices)
    if isinstance(model, ConductanceModel):
        return model.ecological_distances(latent, pairs, species_index, context=context)
    return model.ecological_distances(latent, pairs, species_index)
