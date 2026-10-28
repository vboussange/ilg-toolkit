"""Abstract distance models deriving ecological distances from patch features."""

from abc import abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import combinations
from typing import cast, final, overload

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxscape import GridGraph, ResistanceDistance
from jaxscape.solvers import AMJaxCGSolver
from numpy.typing import ArrayLike

from .._checks import _require_float64
from ..config import ResistanceSolverConfig
from ..data import _validate_pairs, RegionBatch
from ..prediction import Prediction
from ..resistance import _mean_conductance, effective_resistance, ResistanceSolverContext
from .common import _inference_model, pixel_to_patch_nodes


# Sampling-unit vertices, followed by pairs indexing those sampling units.
type DistancePairs = tuple[jax.Array, tuple[jax.Array, jax.Array]]
# Ordered sampling-unit pairs, or ``None`` for every distinct pair.
type _Pairs = Sequence[tuple[str, str]] | None


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

    Declare static ``species`` and ``patch_size`` fields; conductance models
    also declare a static ``solver_config`` holding a ``ResistanceSolverConfig``.
    ``species`` is a tuple of unique species names, one per species-specific head
    in parameter order, or ``None`` for one unnamed species.
    Keep fields and initialization together in the concrete class, and compose
    existing concrete models to reuse their behavior rather than subclassing them.

    Raster features have shape ``(rows, columns, features)``, with meanings and
    order compatible across regions. Sampling locations are integer
    ``(row, column)`` cells mapped to the vertex of their containing patch. During
    training, ``species_index`` may be a scalar integer array, so select species
    parameters by array indexing. The inherited ``predict`` returns ecological
    distances for a ``query``, and ``train`` estimates the array-valued parameters with the
    chosen loss. See the experimental [multispecies example](../examples/multispecies.md)
    for species labels.

    Calibration fitting, prediction, validation and conductance surfaces evaluate
    the model in Equinox inference mode (``eqx.nn.inference_mode``). Training clears
    every ``inference`` flag and binds its own random key to each
    inference-dependent layer whose call accepts ``key``, such as
    ``eqx.nn.Dropout``, so call those layers without a key. Stateful layers
    (Equinox state indices, such as ``eqx.nn.BatchNorm``) raise a ``TypeError``.

    Examples:
        A single-species embedding-distance model with a trainable linear feature map:

        >>> @final
        ... class LinearEmbedding(EmbeddingDistanceModel):
        ...     weights: jax.Array
        ...     species: tuple[str, ...] | None = eqx.field(static=True, default=None)
        ...     patch_size: int = eqx.field(static=True, default=1)
        ...
        ...     def backbone(self, features: jax.Array) -> jax.Array:
        ...         return features @ self.weights
        ...
        ...     def embedding(self, latent: jax.Array, species_index) -> jax.Array:
        ...         return latent
        >>> distance_model = LinearEmbedding(jnp.eye(2))
    """

    species: eqx.AbstractVar[tuple[str, ...] | None]
    patch_size: eqx.AbstractVar[int]

    def __check_init__(self):
        if self.species is not None and not isinstance(self.species, tuple):
            raise ValueError("Distance-model species must be a tuple of names or None")
        _species_names(self.species)

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
        self, region: RegionBatch, query: Mapping[str, Sequence[tuple[str, str]] | None]
    ) -> dict[str, Prediction]: ...

    @overload
    def predict(
        self, region: RegionBatch, query: Sequence[tuple[str, str]] | None = None
    ) -> Prediction: ...

    def predict(
        self,
        region: RegionBatch,
        query: Mapping[str, Sequence[tuple[str, str]] | None]
        | Sequence[tuple[str, str]]
        | None = None,
    ) -> Prediction | dict[str, Prediction]:
        """Ecological distances for the queried species and sampling-unit pairs.

        ``query`` maps species names from the distance model's ``species`` to
        ordered pairs, or to ``None`` for every distinct pair in sampling-unit
        order, and returns a mapping of predictions by species in query order.
        Patch features are computed once for every queried species. A
        single-species model also accepts a sequence of pairs, or ``None`` for
        every distinct pair, and returns one ``Prediction``. Unknown species, and
        a sequence or ``None`` for a multispecies model, raise a ``ValueError``.

        Explicit pairs limit the returned vector and avoid generating all pair
        labels; this does not guarantee that an underlying resistance solver
        avoids its distance matrix among sampling-unit vertices. Calling
        ``as_matrix()`` on a result explicitly allocates a sampling-unit-square
        array.
        """
        _require_float64()
        requests, by_species = _resolve_query(self.species, query)
        results = {
            name: Prediction(values, labels, region.name, prepared.sampling_unit_ids, name)
            for name, (prepared, labels, values) in _species_distances(
                self, region, requests
            ).items()
        }
        return _query_results(results, by_species)


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
        """Effective resistance aligned with the requested sampling-unit pairs.

        ``context`` is the solver that ``prepare_distance_context`` prepares for
        this model and region.
        """
        nodes, indices = pairs
        if context is None:
            raise ValueError("Effective resistance requires a prepared resistance solver context")
        if context.config != self.solver_config:
            raise ValueError("Solver context policy does not match the conductance model settings")
        surface = self.conductance(latent, species_index)
        return effective_resistance(surface, nodes, context=context)[indices]

    def conductance_surface(self, region: RegionBatch, species: str | None = None) -> jax.Array:
        """Positive conductance surface of one species, requested by name.

        ``species`` is one of the distance model's ``species``; a single-species
        model needs no name.
        """
        _require_float64()
        index = _species_index(self.species, species)
        model = _inference_model(self)
        surface = model.conductance(model.backbone(region.feature_array), index)
        return eqx.error_if(
            surface,
            jnp.any(~jnp.isfinite(surface) | (surface <= 0)),
            "Distance model produced nonfinite or nonpositive conductance",
        )


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


def _species_names(species: Sequence[str] | None) -> tuple[str, ...] | None:
    """Unique, nonempty species names in parameter order, or ``None`` for one unnamed species."""
    if species is None:
        return None
    if isinstance(species, (str, bytes)) or not isinstance(species, Sequence):
        raise ValueError("Species must be a sequence of names or None")
    names = tuple(species)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Species must contain at least one nonempty name")
    if len(set(names)) != len(names):
        raise ValueError(f"Species names must be unique, got {names!r}")
    return names


def _model_species(model: AbstractDistanceModel) -> tuple[str | None, ...]:
    """The distance model's species names in parameter order; ``None`` names one unnamed species."""
    return (None,) if model.species is None else model.species


@final
@dataclass(frozen=True)
class _SpeciesQuery:
    """One queried species: its name, parameter row, and pairs (``None`` for every pair)."""

    name: str | None
    index: int
    pairs: tuple[tuple[str, str], ...] | None


def _species_index(species: tuple[str, ...] | None, name: str | None) -> int:
    """Parameter row of a named species; ``None`` names a single-species model's species."""
    if name is None:
        if species is not None and len(species) > 1:
            raise ValueError(f"Multispecies models require a species name from {species!r}")
        return 0
    if species is None or name not in species:
        raise ValueError(f"Unknown species {name!r}; the distance model's species are {species!r}")
    return species.index(name)


def _query_pairs_or_all(pairs: object) -> tuple[tuple[str, str], ...] | None:
    """One species' query pairs as a tuple, or ``None`` for every distinct pair."""
    if pairs is None:
        return None
    if isinstance(pairs, (str, bytes)) or not isinstance(pairs, Sequence):
        raise ValueError("Query pairs must be a sequence of sampling-unit pairs or None")
    return tuple(cast(Sequence[tuple[str, str]], pairs))


def _resolve_query(
    species: tuple[str, ...] | None, query: Mapping[str, _Pairs] | _Pairs
) -> tuple[tuple[_SpeciesQuery, ...], bool]:
    """Queried species in query order, and whether results are keyed by species.

    A mapping names species from ``species`` and returns results by species. A
    sequence of pairs, or ``None``, queries the only species of a single-species
    model and returns one result.
    """
    if isinstance(query, Mapping):
        if not query:
            raise ValueError("A species query must name at least one species")
        requests = []
        for name, pairs in query.items():
            if not isinstance(name, str):
                raise ValueError(f"Query species must be names, got {name!r}")
            requests.append(
                _SpeciesQuery(name, _species_index(species, name), _query_pairs_or_all(pairs))
            )
        return tuple(requests), True
    if species is not None and len(species) != 1:
        raise ValueError(
            "Multispecies models require a query mapping species names to pairs, "
            f"with species from {species!r}"
        )
    name = None if species is None else species[0]
    return (_SpeciesQuery(name, 0, _query_pairs_or_all(query)),), False


def _query_results[T](results: Mapping[str | None, T], by_species: bool) -> T | dict[str, T]:
    """Results by species for a species query, or the single result otherwise."""
    if by_species:
        return cast(dict[str, T], dict(results))
    (result,) = results.values()
    return result


def region_for_species(region: RegionBatch, name: str | None) -> RegionBatch:
    """Region batch containing one species' sampling units."""
    if region.species is None:
        return region
    indices = tuple(index for index, species in enumerate(region.species) if species == name)
    if len(indices) < 2:
        raise ValueError("A species query requires at least two sampling units in the region")
    return replace(
        region,
        sampling_unit_ids=tuple(region.sampling_unit_ids[index] for index in indices),
        grid_positions=jnp.asarray(region.grid_positions)[jnp.asarray(indices)],
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


def _species_distances(
    model: AbstractDistanceModel,
    region: RegionBatch,
    requests: tuple[_SpeciesQuery, ...],
    extra_pairs: Mapping[str | None, tuple[tuple[str, str], ...]] | None = None,
) -> dict[str | None, tuple[RegionBatch, tuple[tuple[str, str], ...], jax.Array]]:
    """Each queried species' region batch, query pair labels, and ecological distances.

    Patch features and the resistance solver are prepared once, in inference
    mode. ``extra_pairs`` appends a species' additional pairs after its query
    pairs, in both the labels and the distances.
    """
    if len(_model_species(model)) > 1 and region.species is None:
        raise ValueError("Multispecies queries require sampling-unit species labels")
    model = _inference_model(model)
    context = prepare_distance_context(model, region.feature_array)
    latent = model.backbone(region.feature_array)
    results = {}
    for request in requests:
        prepared = region_for_species(region, request.name)
        extra = () if extra_pairs is None else extra_pairs.get(request.name, ())
        pairs = _query_pairs(prepared, request.pairs)[0] + extra if extra else request.pairs
        labels, values = query_distances_from_latent(
            model, latent, prepared, pairs, species_index=request.index, context=context
        )
        results[request.name] = (prepared, labels, values)
    return results


def prepare_distance_context(
    model: AbstractDistanceModel, features: ArrayLike
) -> ResistanceSolverContext | None:
    """Reusable resistance solver for the region's conductance grid graph.

    Conjugate gradient, with optional algebraic multigrid preconditioning prepared
    on a uniform conductance grid graph, follows the model's ``solver_config``.
    Distance models without a conductance grid graph need no solver.
    """
    if not isinstance(model, ConductanceModel):
        return None
    shape = jnp.asarray(features).shape
    if len(shape) != 3 or shape[0] % model.patch_size or shape[1] % model.patch_size:
        raise ValueError("Raster dimensions must be divisible by model patch_size")
    graph_shape = (shape[0] // model.patch_size, shape[1] // model.patch_size)
    if graph_shape[0] * graph_shape[1] < 2:
        raise ValueError("Effective resistance requires at least two graph vertices")
    config = model.solver_config
    if not config.use_amg:
        return ResistanceSolverContext(
            graph_shape,
            lx.CG(rtol=config.rtol, atol=config.atol, max_steps=config.max_steps),
            config=config,
        )
    solver = AMJaxCGSolver(
        rtol=config.rtol, atol=config.atol, max_steps=config.max_steps, coarse_solver="pinv"
    )
    template = GridGraph(jnp.ones(graph_shape, dtype=jnp.float64), fun=_mean_conductance)
    state = ResistanceDistance(solver=solver).init_preconditioner(template)
    return ResistanceSolverContext(graph_shape, solver, state, config)


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
