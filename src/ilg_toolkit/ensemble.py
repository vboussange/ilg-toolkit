"""Ensembles of landscape genetics models and their averaged genetic predictions."""

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
    """Ensemble prediction: the mean of the members' genetic predictions and each member's.

    Attributes:
        mean: Equal-weight mean of the members' original-scale values, aligned with
            the query pairs, without extras.
        members: Each ensemble member's ``Prediction``, extras included, in ensemble
            order.
    """

    mean: Prediction
    members: tuple[Prediction, ...]


def _average_predictions(predictions: tuple[Prediction, ...]) -> EnsemblePrediction:
    """Returns the equal-weight original-scale mean of aligned member predictions."""
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
    """Ensemble: an ordered collection of compatible landscape genetics models.

    Ensemble members share their target contract, feature names and species. The
    ensemble stores the members as a tuple, and a repeated model counts as several
    members. Conductance surfaces and ecological distances come from each
    member's own distance model.

    Args:
        members: Nonempty list or tuple of landscape genetics models.

    Raises:
        ValueError: If ``members`` is empty, not a list or tuple, holds an object
            that is not a landscape genetics model, or holds incompatible models.

    Examples:
        Combine the selected models of two training runs::

            ensemble = Ensemble([first_state.selected_model, second_state.selected_model])
            result = ensemble.predict(region, [("north", "south")])
            mean = result.mean.values
            member_values = jnp.stack([member.values for member in result.members])
            spread = member_values.std(axis=0)

        Each member applies its inverse target transform before averaging, so means
        and member predictions are on the original scale. Conductance surfaces come
        from the members::

            surfaces = [
                member.distance_model.conductance_surface(region)
                for member in ensemble.members
            ]
    """

    members: Sequence[LandscapeGeneticsModel]

    def __post_init__(self):
        if not isinstance(self.members, (list, tuple)) or not self.members:
            raise ValueError(
                "An ensemble requires a nonempty list or tuple of landscape genetics models"
            )
        members = tuple(self.members)
        for index, member in enumerate(members):
            if not isinstance(member, LandscapeGeneticsModel):
                raise ValueError(f"Ensemble member {index} must be a LandscapeGeneticsModel")
        first = members[0]
        for index, member in enumerate(members[1:], start=1):
            if (
                member.target != first.target
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
        """Returns genetic predictions averaged over the ensemble members on the original scale.

        An error of a member's prediction propagates with a note naming the member's
        zero-based position.

        Args:
            region: Region batch with the landscape features and the query pairs'
                sampling units.
            query: Query as in
                [`LandscapeGeneticsModel.predict`][ilg_toolkit.LandscapeGeneticsModel.predict]:
                a mapping from species name to query pairs, or to ``None`` for every
                distinct pair; for a single-species ensemble, a sequence of query
                pairs, or ``None`` for every distinct pair.

        Returns:
            Ensemble predictions by species for a mapping query, or one
            ``EnsemblePrediction`` for a sequence of pairs or ``None``.

        Raises:
            ValueError: If the query is invalid, or a member's predictions are not
                finite.
            RuntimeError: If float64 is disabled.
        """
        _require_float64()
        requests, by_species = _resolve_query(self.members[0].species, query)
        by_member = []
        for index, model in enumerate(self.members):
            try:
                prediction = model.predict(region, query)
            except Exception as error:
                error.add_note(f"Raised by ensemble member {index}")
                raise
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
