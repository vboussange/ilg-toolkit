"""Ordered fitted-model ensembles and original-scale genetic predictions."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast, final, overload, TypeVar

import jax
import jax.numpy as jnp

from ._checks import _require_float64
from .data import RegionBatch
from .model import LandscapeGeneticsModel
from .models import ConductanceModel
from .prediction import Prediction


_Result = TypeVar("_Result")


@final
@dataclass(frozen=True)
class EnsemblePrediction:
    """Original-scale mean and descriptive spread across every ensemble member.

    ``member_values`` rows follow the zero-based ``member_ids`` positions and
    columns follow query-pair order. Spread is population standard deviation,
    describing member disagreement.

    Notes:
        Member spread is not statistical predictive uncertainty.
    """

    prediction: Prediction
    member_ids: tuple[int, ...]
    member_values: jax.Array
    member_spread: jax.Array


@final
@dataclass(frozen=True)
class EnsembleSurface:
    """Mean and descriptive spread of conductance surfaces in member order."""

    values: jax.Array
    member_ids: tuple[int, ...]
    member_values: jax.Array
    member_spread: jax.Array
    region_name: str


def aggregate_predictions(predictions: Sequence[Prediction]) -> EnsemblePrediction:
    """Equal-weight original-scale mean and spread of ordered, aligned predictions.

    Region, species, query pairs, sampling-unit order, and finite values must
    agree. Callers supply common target units; ``Ensemble`` validates the fitted
    models' target contracts before obtaining these predictions.
    """
    if not isinstance(predictions, (list, tuple)) or not predictions:
        raise ValueError("Aggregation requires a nonempty list or tuple of predictions")
    if any(not isinstance(value, Prediction) for value in predictions):
        raise ValueError("Aggregate inputs must be labeled original-scale Prediction values")
    first = predictions[0]
    values = []
    for index, prediction in enumerate(predictions):
        if (
            prediction.sampling_unit_ids != first.sampling_unit_ids
            or prediction.pairs != first.pairs
            or prediction.species != first.species
            or prediction.region_name != first.region_name
        ):
            raise ValueError(f"Ensemble member {index} has misaligned prediction identities")
        member_values = jnp.asarray(prediction.values, dtype=jnp.float64)
        if member_values.shape != first.values.shape or not bool(jnp.isfinite(member_values).all()):
            raise ValueError(f"Ensemble member {index} predictions must be aligned and finite")
        values.append(member_values)
    stacked = jnp.stack(values)
    mean, spread = _summarize_members(stacked)
    return EnsemblePrediction(
        Prediction(mean, first.pairs, first.region_name, first.sampling_unit_ids, first.species),
        tuple(range(len(predictions))),
        stacked,
        spread,
    )


@jax.jit
def _member_moments(values):
    # Divide before summing and scale before squaring to prevent overflow.
    # Binary exponent scaling also protects spread when the reciprocal of a
    # very large scale would underflow.
    mean = jnp.sum(values / len(values), axis=0)
    _, exponent = jnp.frexp(jnp.max(jnp.abs(values), axis=0))
    normalized = jnp.ldexp(values, -exponent)
    return mean, jnp.ldexp(jnp.std(normalized, axis=0, ddof=0), exponent)


def _summarize_members(values):
    mean, spread = _member_moments(jnp.asarray(values, dtype=jnp.float64))
    if not bool(jnp.isfinite(mean).all() & jnp.isfinite(spread).all()):
        raise FloatingPointError("Ensemble mean or descriptive spread is nonfinite")
    return mean, spread


@final
@dataclass(frozen=True)
class Ensemble:
    """An ordered, nonempty collection of compatible fitted genetic models.

    Members share target semantics, feature meanings, and species order. The
    input list or tuple is snapshotted, and repeated entries remain separate
    members. Every member must predict successfully; errors identify the failing
    member's zero-based position.

    Examples:
        Assemble models fitted independently by the caller::

            ensemble = Ensemble([first_state.selected_model, second_state.selected_model])
            result = ensemble.predict(region, pairs=[("north", "south")])
            mean = result.prediction.values
            spread = result.member_spread
            first_member_values = result.member_values[0]

        Means and spread are in the original target units. Each member applies
        its inverse target transform before aggregation.
    """

    members: Sequence[LandscapeGeneticsModel]

    def __post_init__(self):
        if not isinstance(self.members, (list, tuple)) or not self.members:
            raise ValueError("An ensemble requires a nonempty list or tuple of fitted models")
        members = tuple(self.members)
        for index, member in enumerate(members):
            if not isinstance(member, LandscapeGeneticsModel):
                raise ValueError(f"Ensemble member {index} must be a fitted LandscapeGeneticsModel")
        first = members[0]
        for index, member in enumerate(members[1:], start=1):
            if (
                member.target != first.target
                or member.feature_names != first.feature_names
                or member.species != first.species
            ):
                raise ValueError(
                    f"Ensemble member {index} must have matching target, features and species order"
                )
        object.__setattr__(self, "members", members)

    def _map_members(
        self, operation: str, function: Callable[[LandscapeGeneticsModel], _Result]
    ) -> tuple[_Result, ...]:
        values = []
        for index, model in enumerate(self.members):
            try:
                values.append(function(model))
            except Exception as error:
                raise RuntimeError(
                    f"Ensemble member {index} {operation} failed: {error}"
                ) from error
        return tuple(values)

    @overload
    def predict(
        self, region: RegionBatch, pairs=None, *, species: str | None = None
    ) -> EnsemblePrediction: ...

    @overload
    def predict(
        self, region: RegionBatch, pairs=None, *, species: Sequence[str]
    ) -> dict[str, EnsemblePrediction]: ...

    def predict(
        self, region: RegionBatch, pairs=None, *, species: str | Sequence[str] | None = None
    ) -> EnsemblePrediction | dict[str, EnsemblePrediction]:
        """Genetic predictions averaged in original target units over every member.

        A species sequence returns a mapping and computes each member's patch
        features once. See ``LandscapeGeneticsModel.predict`` for species selection.
        """
        _require_float64()
        pairs = None if pairs is None else tuple(pairs)
        predictions = self._map_members(
            "prediction", lambda model: model.predict(region, pairs=pairs, species=species)
        )
        if isinstance(species, str) or species is None:
            return aggregate_predictions(cast(tuple[Prediction, ...], predictions))
        by_species = cast(tuple[dict[str, Prediction], ...], predictions)
        return {
            name: aggregate_predictions(tuple(values[name] for values in by_species))
            for name in species
        }

    @overload
    def ecological_distances(
        self, region: RegionBatch, *, species: str | None = None
    ) -> tuple[jax.Array, ...]: ...

    @overload
    def ecological_distances(
        self, region: RegionBatch, *, species: Sequence[str]
    ) -> dict[str, tuple[jax.Array, ...]]: ...

    def ecological_distances(
        self, region: RegionBatch, *, species: str | Sequence[str] | None = None
    ) -> tuple[jax.Array, ...] | dict[str, tuple[jax.Array, ...]]:
        """Ecological-distance matrices in member order, each on its own scale."""
        _require_float64()

        def predict_member(model):
            model._validate_region(region)
            return model.distance_model.predict(
                region, species=species, species_order=model.species
            )

        predictions = self._map_members("ecological-distance prediction", predict_member)
        if isinstance(species, str) or species is None:
            return tuple(value.as_matrix() for value in predictions)
        return {name: tuple(values[name].as_matrix() for values in predictions) for name in species}

    @overload
    def conductance_surfaces(
        self, region: RegionBatch, *, species: str | None = None
    ) -> EnsembleSurface: ...

    @overload
    def conductance_surfaces(
        self, region: RegionBatch, *, species: Sequence[str]
    ) -> dict[str, EnsembleSurface]: ...

    def conductance_surfaces(
        self, region: RegionBatch, *, species: str | Sequence[str] | None = None
    ) -> EnsembleSurface | dict[str, EnsembleSurface]:
        """Mean and descriptive spread of conductance surfaces in member order."""
        _require_float64()

        def surface(model):
            model._validate_region(region)
            if not isinstance(model.distance_model, ConductanceModel):
                raise TypeError("This distance model does not produce a conductance surface")
            return model.distance_model.conductance_surface(
                region, species=species, species_order=model.species
            )

        surfaces = self._map_members("conductance-surface prediction", surface)

        def summarize(values) -> EnsembleSurface:
            arrays = [jnp.asarray(value, dtype=jnp.float64) for value in values]
            for index, value in enumerate(arrays):
                if value.shape != arrays[0].shape:
                    raise ValueError(
                        f"Ensemble member {index} conductance surface shape {value.shape} "
                        f"does not match member 0 shape {arrays[0].shape}"
                    )
                if not bool(jnp.isfinite(value).all()):
                    raise ValueError(f"Ensemble member {index} conductance surface is nonfinite")
            stacked = jnp.stack(arrays)
            mean, spread = _summarize_members(stacked)
            return EnsembleSurface(
                mean, tuple(range(len(self.members))), stacked, spread, region.name
            )

        if isinstance(species, str) or species is None:
            return summarize(surfaces)
        return {name: summarize(tuple(values[name] for values in surfaces)) for name in species}
