"""Ordered fitted-model ensembles and original-scale genetic predictions."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast, final, overload

import equinox as eqx
import jax.numpy as jnp

from ._checks import _require_float64
from .data import RegionBatch
from .model import LandscapeGeneticsModel
from .models.base import _query_results, _resolve_query
from .prediction import Prediction


@final
class EnsemblePrediction(eqx.Module):
    """Original-scale mean of the members' genetic predictions, and the member predictions.

    ``mean`` holds the equal-weight mean of the member values, aligned with the
    query pairs. ``members`` holds each member's own ``Prediction``, extras
    included, in ensemble member order, for spread statistics of the caller's
    choice.
    """

    mean: Prediction
    members: tuple[Prediction, ...]


def _average_predictions(predictions: tuple[Prediction, ...]) -> EnsemblePrediction:
    """Equal-weight original-scale mean of aligned member predictions."""
    values = jnp.stack([jnp.asarray(value.values, dtype=jnp.float64) for value in predictions])
    for index, member in enumerate(values):
        if not bool(jnp.isfinite(member).all()):
            raise ValueError(f"Ensemble member {index} predictions must be finite")
    # Divide before summing so that averaging large finite values cannot overflow.
    mean = jnp.sum(values / len(predictions), axis=0)
    first = predictions[0]
    return EnsemblePrediction(
        Prediction(mean, first.pairs, first.region_name, first.sampling_unit_ids, first.species),
        predictions,
    )


@final
@dataclass(frozen=True)
class Ensemble:
    """An ordered, nonempty collection of compatible fitted genetic models.

    Members share target semantics, feature meanings, and species. The
    input list or tuple is snapshotted, and repeated entries remain separate
    members. Every member must predict successfully; errors identify the failing
    member's zero-based position. Conductance surfaces and ecological distances
    come from each member's own distance model.

    Examples:
        Assemble models fitted independently by the caller::

            ensemble = Ensemble([first_state.selected_model, second_state.selected_model])
            result = ensemble.predict(region, [("north", "south")])
            mean = result.mean.values
            member_values = jnp.stack([member.values for member in result.members])
            spread = member_values.std(axis=0)

        Means and member predictions are in the original target units. Each member
        applies its inverse target transform before averaging. Member surfaces
        come from the members themselves::

            surfaces = [
                member.distance_model.conductance_surface(region)
                for member in ensemble.members
            ]
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
                or member.feature_count != first.feature_count
                or member.feature_names != first.feature_names
                or member.species != first.species
            ):
                raise ValueError(
                    f"Ensemble member {index} must have matching target, features and species"
                )
        object.__setattr__(self, "members", members)

    @overload
    def predict(
        self, region: RegionBatch, query: Mapping[str, Sequence[tuple[str, str]] | None]
    ) -> dict[str, EnsemblePrediction]: ...

    @overload
    def predict(
        self, region: RegionBatch, query: Sequence[tuple[str, str]] | None = None
    ) -> EnsemblePrediction: ...

    def predict(
        self,
        region: RegionBatch,
        query: Mapping[str, Sequence[tuple[str, str]] | None]
        | Sequence[tuple[str, str]]
        | None = None,
    ) -> EnsemblePrediction | dict[str, EnsemblePrediction]:
        """Genetic predictions averaged in original target units over every member.

        ``query`` follows ``LandscapeGeneticsModel.predict``: a mapping from
        species name to pairs, or to ``None`` for every distinct pair, returns
        results by species; a sequence of pairs, or ``None``, on a single-species
        ensemble returns one result. Each result holds the mean and every member's
        prediction.
        """
        _require_float64()
        requests, by_species = _resolve_query(self.members[0].species, query)
        by_member = []
        for index, model in enumerate(self.members):
            try:
                prediction = model.predict(region, query)
            except Exception as error:
                raise RuntimeError(f"Ensemble member {index} prediction failed: {error}") from error
            by_member.append(
                cast(dict[str | None, Prediction], prediction)
                if by_species
                else {requests[0].name: cast(Prediction, prediction)}
            )
        results = {
            request.name: _average_predictions(
                tuple(predictions[request.name] for predictions in by_member)
            )
            for request in requests
        }
        return _query_results(results, by_species)
