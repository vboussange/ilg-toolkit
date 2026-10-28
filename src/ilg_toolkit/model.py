"""Landscape genetics models combining ecological distances and genetic calibration."""

from collections.abc import Mapping, Sequence
from typing import final, overload

import equinox as eqx
import jax.numpy as jnp

from ._checks import _require_float64
from .calibration import AbstractCalibrationModule
from .data import PairwiseObservations, RegionBatch, TargetSpec
from .models import AbstractDistanceModel
from .models.base import (
    _model_species,
    _query_results,
    _resolve_query,
    _species_distances,
    _SpeciesQuery,
)
from .prediction import Prediction


@final
class LandscapeGeneticsModel(eqx.Module):
    """A distance model composed with fitted genetic calibration.

    Species names come from the distance model. Feature meanings define the query
    contract. The calibration owns target semantics and statistical prediction
    behavior.
    """

    distance_model: AbstractDistanceModel
    calibration: AbstractCalibrationModule
    feature_count: int
    feature_names: tuple[str, ...] | None = None

    def __post_init__(self):
        if not isinstance(self.distance_model, AbstractDistanceModel):
            raise TypeError("distance_model must implement AbstractDistanceModel")
        if not isinstance(self.calibration, AbstractCalibrationModule):
            raise TypeError("calibration must implement AbstractCalibrationModule")

    @property
    def species(self) -> tuple[str, ...] | None:
        """The distance model's species names, or ``None`` for one unnamed species."""
        return self.distance_model.species

    @property
    def target(self) -> TargetSpec:
        """The target contract owned by the calibration module."""
        return self.calibration.target

    def _validate_region(self, region: RegionBatch):
        if region.feature_array.shape[-1] != self.feature_count:
            raise ValueError(f"Expected {self.feature_count} feature channels")
        if region.feature_names != self.feature_names:
            raise ValueError(
                "Query feature contract must match training feature meanings and order"
            )

    @overload
    def predict(
        self,
        region: RegionBatch,
        query: Mapping[str, Sequence[tuple[str, str]] | None],
        *,
        known_effects: bool = False,
        support: PairwiseObservations | Sequence[PairwiseObservations] | None = None,
    ) -> dict[str, Prediction]: ...

    @overload
    def predict(
        self,
        region: RegionBatch,
        query: Sequence[tuple[str, str]] | None = None,
        *,
        known_effects: bool = False,
        support: PairwiseObservations | Sequence[PairwiseObservations] | None = None,
    ) -> Prediction: ...

    def predict(
        self,
        region: RegionBatch,
        query: Mapping[str, Sequence[tuple[str, str]] | None]
        | Sequence[tuple[str, str]]
        | None = None,
        *,
        known_effects: bool = False,
        support: PairwiseObservations | Sequence[PairwiseObservations] | None = None,
    ) -> Prediction | dict[str, Prediction]:
        """Predict genetic relationships on the target's original scale.

        ``query`` maps species names from the distance model's ``species`` to
        ordered pairs, or to ``None`` for every distinct pair, and returns a mapping
        of predictions by species in query order. A single-species model also
        accepts a sequence of pairs, or ``None`` for every distinct pair, and
        returns one ``Prediction``. Requested pairs retain their order and
        orientation. The distance model computes ecological distances, and fitted calibration
        converts these into genetic predictions. Maximum-likelihood population effects
        (MLPE) defaults to marginal means;
        ``known_effects=True`` and ``support=...`` condition population effects.

        | Mode | Population effects | Statistical extras |
        | --- | --- | --- |
        | Default | Marginal mean, without fitted effects | `model_values` |
        | `known_effects=True` | Saved posterior from calibration | Mean + variances |
        | `support=observations` | Saved posterior updated with support | Mean + variances |

        Args:
            region: Landscape features and located sampling units for prediction.
            query: Mapping from species name to ordered sampling-unit pairs, or to
                ``None`` for every distinct pair; for a single-species model, a
                sequence of pairs, or ``None`` for every distinct pair.
            known_effects: Include the saved MLPE population-effect posterior.
            support: Additional genetic observations that update the saved MLPE
                population-effect posterior for this call, one per queried species.
                Support pairs must be disjoint from calibration and query pairs.

        Returns:
            Pair-aligned genetic predictions by species for a mapping query, or one
            ``Prediction`` for a sequence of pairs or ``None``.

        Raises:
            ValueError: The query names an unknown species, or a multispecies model
                receives a sequence of pairs or ``None``.

        Examples:
            Prepare one landscape containing six located populations and fit MLPE
            calibration from measurements among its first four populations::

                import coordax as cx
                import jax
                import jax.numpy as jnp

                jax.config.update("jax_enable_x64", True)
                from ilg_toolkit import (
                    MLPELoss, PairwiseObservations, RegionBatch, TargetSpec, calibrate,
                )
                from ilg_toolkit.models import LogLinearConductance

                region = RegionBatch(
                    "valley", cx.field(jnp.arange(6.0).reshape(1, 6, 1),
                                       "row", "column", "feature"),
                    tuple(f"p{i}" for i in range(6)),
                    [[0, i] for i in range(6)], feature_names=("elevation",),
                )
                observations = PairwiseObservations.from_pairs(
                    [("p0", "p1"), ("p0", "p2"), ("p0", "p3"),
                     ("p1", "p2"), ("p1", "p3"), ("p2", "p3")],
                    [1.2, 1.6, 2.9, 1.7, 2.0, 1.3],
                    target=TargetSpec("genetic divergence", units="index"),
                )
                model = calibrate(region, observations,
                                  LogLinearConductance(1, patch_size=1), loss=MLPELoss())
                query_pairs = [("p4", "p2"), ("p5", "p3")]

            Marginal predictions use only fitted intercept, slope, and distance
            standardization. Population identities do not add saved effects::

                marginal = model.predict(region, query_pairs)
                marginal.values                 # Genetic means in original units.
                marginal.extras["model_values"] # Means before the target inverse transform.
                marginal.pairs                  # Same alignment as query_pairs.

            Saved-effect predictions add posterior population means estimated from the
            calibration observations. Unobserved populations retain zero prior mean
            and prior population variance; calibration/query overlap is allowed::

                saved = model.predict(region, query_pairs, known_effects=True)
                saved.extras["effect_variance"] # Uncertainty in the summed population effects.
                saved.extras["model_variance"]  # Effect + residual variance + fixed jitter.

            Additional support observations update that saved posterior, including
            newly observed populations, for this prediction only. The landscape
            parameters, coefficients, standardization, and variance components remain
            fixed; the model itself is unchanged::

                support = PairwiseObservations.from_pairs(
                    [("p4", "p0"), ("p4", "p1")], [2.7, 2.2], target=model.target,
                )
                conditioned = model.predict(region, query_pairs, support=support)
                conditioned.values
                conditioned.extras["model_values"]
                conditioned.extras["model_variance"]
                conditioned.extras["effect_variance"]

            All point predictions inverse-transform model-scale means. Conditional
            variances stay on the model scale. Use ``as_matrix()`` or
            ``as_matrix("effect_variance")`` to arrange the corresponding values by
            sampling-unit identity.

        Notes:
            With nonlinear target transforms, conditional variances are not original-scale
            variances, and inverse-transformed means are not bias-corrected expectations.

            Support must have matching target/species metadata and located sampling
            units, with pairs disjoint from both calibration and query pairs, including
            reversals. Shared population identities are allowed. Querying every pair
            with support requests all pairs and consequently creates invalid overlap.
            Passing both ``known_effects=True`` and support is invalid; direct
            calibration supports neither conditional mode. Every MLPE mode requires
            existing calibration for the query region/species and population units.

            See the [comprehensive MLPE workflow notebook](../examples/mlpe_workflow.md)
            for training, calibration, and evaluation, and the
            [ensemble workflow](../examples/ensemble.md) for checkpoint continuation.
        """
        _require_float64()
        self._validate_region(region)
        requests, by_species = _resolve_query(self.species, query)
        support_by_species = self._support_observations(support, requests)
        distances_by_species = _species_distances(
            self.distance_model,
            region,
            requests,
            {name: value.observed_pairs for name, value in support_by_species.items()},
        )
        results = {}
        for request in requests:
            prepared, labels, distances = distances_by_species[request.name]
            observations = support_by_species.get(request.name)
            query_count = len(labels) - (
                0 if observations is None else len(observations.observed_pairs)
            )
            result = self.calibration.predict(
                distances[:query_count],
                labels[:query_count],
                region_name=region.name,
                species_index=request.index,
                sampling_unit_kinds=prepared.sampling_unit_kinds,
                known_effects=known_effects,
                support=observations,
                support_distances=None if observations is None else distances[query_count:],
            )
            results[request.name] = Prediction(
                jnp.asarray(result.values),
                labels[:query_count],
                region.name,
                prepared.sampling_unit_ids,
                species=request.name,
                extras=result.extras,
            )
        return _query_results(results, by_species)

    def _support_observations(
        self,
        support: PairwiseObservations | Sequence[PairwiseObservations] | None,
        requests: tuple[_SpeciesQuery, ...],
    ) -> dict[str | None, PairwiseObservations]:
        if support is None:
            return {}
        observations = (support,) if isinstance(support, PairwiseObservations) else support
        if (
            not isinstance(observations, Sequence)
            or not observations
            or any(not isinstance(value, PairwiseObservations) for value in observations)
        ):
            raise ValueError("Support requires observations for each queried species")
        by_species = {value.species: value for value in observations}
        if len(by_species) != len(observations):
            raise ValueError("Support requires one observations object per species")
        if len(_model_species(self.distance_model)) == 1 and set(by_species) == {None}:
            by_species = {requests[0].name: observations[0]}
        if set(by_species) != {request.name for request in requests}:
            raise ValueError("Support observations must match the queried species")
        return by_species
