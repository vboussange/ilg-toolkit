"""Raster encoders and their ecological-distance boundary.

Extracted from the cleaned deepilg architecture. Distance models return
ecological distances; calibration determines how those distances relate to genetic
targets.
"""

from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

from .._pairs import PairSelection
from ..config import ResistanceSolverConfig
from ..data import ObservationPartition, RegionBatch
from ..prediction import RawDistances
from ..resistance import (
    ResistanceSolverContext,
    _effective_resistance_pairs,
    build_resistance_context,
)
from .common import pixel_to_patch_nodes


def validate_species(species) -> tuple[str, ...]:
    if not isinstance(species, Sequence) or isinstance(species, (str, bytes)):
        raise ValueError("species registry must be an ordered sequence of names")
    names = tuple(species)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("species registry requires nonempty names")
    if len(set(names)) != len(names):
        raise ValueError("species registry names must be unique")
    return names


class AbstractDistanceModel(eqx.Module):
    """Native region/species/pair interface with a numerical custom-model seam.

    Custom models declare a fixed static ``species`` tuple and implement
    ``_pair_distances(features, selection, species_index, ...)``. The selection
    contains numeric deduplicated endpoints; species_index is a scalar JAX array.
    Implementations return a vector, without genetic labels or a full pair matrix.
    The inherited public interface handles labels, identities and result metadata.
    No public encoding lifecycle or legacy-subclass adapter is required.
    """

    species: eqx.AbstractVar[tuple[str, ...]]

    def __check_init__(self):
        if not isinstance(self.species, tuple) or validate_species(self.species) != self.species:
            raise ValueError("Native distance models require a fixed species tuple")

    def prepare(self, features: ArrayLike) -> ResistanceSolverContext | None:
        """Prepare shape-dependent numerical context, outside differentiation."""
        return None

    @abstractmethod
    def _pair_distances(
        self,
        features: ArrayLike,
        selection: PairSelection,
        species_index: jax.Array,
        *,
        context: ResistanceSolverContext | None = None,
        inference: bool = True,
        key: jax.Array | None = None,
        patch_batch_size: int | None = None,
    ) -> jax.Array:
        """Return requested pair values using only numerical group selections."""

    def _distance_nodes(self, features: ArrayLike, pixel_nodes: ArrayLike) -> jax.Array:
        return jnp.asarray(pixel_nodes, dtype=jnp.int32)

    def _prepare_pairs(self, features, pixel_nodes, pairs) -> PairSelection:
        nodes = np.asarray(self._distance_nodes(features, pixel_nodes))
        endpoints = np.stack(pairs, axis=0)
        selected, inverse = np.unique(nodes[endpoints], return_inverse=True)
        left, right = inverse.reshape(endpoints.shape)
        return PairSelection(jnp.asarray(selected), jnp.asarray(left), jnp.asarray(right))

    def _select_species(self, region: RegionBatch, species=None) -> tuple[str, ...]:
        if not isinstance(region, RegionBatch):
            raise TypeError("Prediction requires a RegionBatch")
        if set(region.sampling_units) - set(self.species):
            raise ValueError("Region contains unknown registered species")
        if species is None:
            if len(self.species) != 1:
                raise ValueError("Multispecies prediction requires explicit species selection")
            selected = self.species
        elif isinstance(species, str):
            selected = (species,)
        elif isinstance(species, Sequence):
            selected = tuple(species)
        else:
            raise ValueError("species selection requires a name or ordered sequence")
        selected = validate_species(selected)
        if set(selected) - set(self.species):
            raise ValueError("Unknown species selection")
        if set(selected) - set(region.sampling_units):
            raise ValueError("Selected species has no sampling units in this region")
        return selected

    def predict_raw_distances(self, region: RegionBatch, *, species=None, pairs=None):
        """Compute selected ecological pairs and return species-keyed results.

        Multispecies registries require an explicit species selection. Pair
        selections are species-keyed, retain caller orientation/order, and never
        accept targets. Omitting pairs requests all distinct pairs for that group.
        """
        selected = self._select_species(region, species)
        if pairs is not None and (not isinstance(pairs, Mapping) or set(pairs) != set(selected)):
            raise ValueError("pairs must be a mapping keyed by exactly the selected species")
        requests = []
        for name in selected:
            units = region.sampling_units[name]
            supplied = () if pairs is None else tuple(pairs[name])
            if any(isinstance(pair, str) for pair in supplied):
                raise ValueError("Each pair must contain two explicit sampling-unit labels")
            labelled = (
                tuple(
                    (units.ids[i], units.ids[j])
                    for i in range(len(units.ids))
                    for j in range(i + 1, len(units.ids))
                )
                if pairs is None
                else tuple(tuple(pair) for pair in supplied)
            )
            ObservationPartition(region.name, name, labelled, role="query")
            lookup = {label: index for index, label in enumerate(units.ids)}
            if any(label not in lookup for pair in labelled for label in pair):
                raise ValueError("Requested pair contains unknown sampling-unit labels")
            indices = tuple(
                jnp.asarray([lookup[pair[side]] for pair in labelled], dtype=jnp.int32)
                for side in (0, 1)
            )
            selection = self._prepare_pairs(
                region.feature_array, region._pixel_nodes(name), indices
            )
            requests.append((name, labelled, selection))
        context = self.prepare(region.feature_array)
        output = {}
        for name, labelled, selection in requests:
            values = self._pair_distances(
                region.feature_array,
                selection,
                jnp.asarray(self.species.index(name), dtype=jnp.int32),
                context=context,
            )
            output[name] = RawDistances(
                values, labelled, region.name, name, region.sampling_units[name].ids
            )
        return output

    def ecological_distances(self, region: RegionBatch, *, species=None):
        """Explicit complete matrix diagnostic for selected species."""
        return {
            name: result.as_matrix()
            for name, result in self.predict_raw_distances(region, species=species).items()
        }


def _pooled_resistance_pairs(
    conductance: Callable[[jax.Array, jax.Array], jax.Array],
    pooled: jax.Array,
    selection: PairSelection,
    species_index: jax.Array,
    config: ResistanceSolverConfig,
    *,
    context: ResistanceSolverContext | None = None,
) -> jax.Array:
    """Validate pooled solver policy before projecting and solving requested pairs."""
    if context is not None and context.config != config:
        raise ValueError("Solver context policy does not match the conductance model settings")
    surface = conductance(pooled, species_index)
    if context is None:
        return _effective_resistance_pairs(surface, selection, config=config)
    return _effective_resistance_pairs(surface, selection, context=context)


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
    def _conductance(
        self, features: jax.Array, species_index: jax.Array, *, patch_batch_size: int | None = None
    ) -> jax.Array:
        """Return a positive patch-grid surface from row, column, feature data."""

    def conductance_surface(self, region: RegionBatch, *, species=None):
        selected = self._select_species(region, species)
        output = {}
        for name in selected:
            surface = self._conductance(
                region.feature_array, jnp.asarray(self.species.index(name), dtype=jnp.int32)
            )
            if not bool(jnp.all(jnp.isfinite(surface) & (surface > 0))):
                raise FloatingPointError("Distance model produced invalid conductance")
            output[name] = surface
        return output

    def _distance_nodes(self, features: ArrayLike, pixel_nodes: ArrayLike) -> jax.Array:
        shape = jnp.asarray(features).shape
        return pixel_to_patch_nodes(
            pixel_nodes, raster_shape=(shape[0], shape[1]), patch_size=self.patch_size
        )

    def _pair_distances(
        self,
        features: ArrayLike,
        selection: PairSelection,
        species_index: jax.Array,
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
        surface = self._conductance(features, species_index, patch_batch_size=patch_batch_size)
        return _effective_resistance_pairs(surface, selection, context=context)


class EmbeddingDistanceModel(AbstractDistanceModel):
    """Encode a raster and measure squared Euclidean separation of node embeddings."""

    patch_size: eqx.AbstractVar[int]

    def prepare(self, features: ArrayLike) -> None:
        """Embedding distances need no shape-dependent numerical preparation."""
        return None

    @abstractmethod
    def _embedding_grid(
        self,
        features: jax.Array,
        species_index: jax.Array,
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
        species_index: jax.Array,
        *,
        context: ResistanceSolverContext | None = None,
        inference: bool = True,
        key: jax.Array | None = None,
        patch_batch_size: int | None = None,
    ) -> jax.Array:
        """Gather selected endpoints before calculating squared separations."""
        features = jnp.asarray(features)
        grid = self._embedding_grid(
            features, species_index, inference=inference, key=key, patch_batch_size=patch_batch_size
        )
        endpoints = grid.reshape(-1, grid.shape[-1])[selection.nodes]
        return jnp.sum(jnp.square(endpoints[selection.left] - endpoints[selection.right]), axis=-1)
