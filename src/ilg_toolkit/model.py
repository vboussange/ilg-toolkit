"""Landscape genetics models combining ecological distances and genetic calibration."""

from collections.abc import Sequence
from typing import cast, final, overload

import equinox as eqx
import jax.numpy as jnp

from .calibration import AbstractCalibrationModule, GroupKey, MLPECalibrationModule
from .data import PairwiseObservations, RegionBatch, TargetSpec
from .mlpe import RegionalMLPECalibration
from .models import AbstractDistanceModel
from .models.base import (
    _query_pairs,
    prepare_distance_context,
    query_distances_from_latent,
    region_for_species,
    select_species,
    validate_species_order,
)
from .models.common import with_dropout_mode
from .prediction import Prediction


@final
class LandscapeGeneticsModel(eqx.Module):
    """A distance model composed with genetic calibration and observation-use records.

    ``species`` associates names with the distance model's parameter rows.
    The calibration owns the target contract and returns genetic predictions
    on the original scale. Training and validation pairs record observation use
    for each region–species group.
    """

    distance_model: AbstractDistanceModel
    calibration: AbstractCalibrationModule
    feature_count: int
    feature_names: tuple[str, ...] | None = None
    species: tuple[str, ...] | None = eqx.field(default=None, kw_only=True)
    training_objective: str | None = eqx.field(default=None, kw_only=True)
    training_pairs: dict[GroupKey, tuple[tuple[str, str], ...]] = eqx.field(default_factory=dict)
    validation_pairs: dict[GroupKey, tuple[tuple[str, str], ...]] = eqx.field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.distance_model, AbstractDistanceModel):
            raise TypeError("distance_model must implement AbstractDistanceModel")
        if not isinstance(self.calibration, AbstractCalibrationModule):
            raise TypeError("calibration must implement AbstractCalibrationModule")
        self.species = validate_species_order(self.species, self.distance_model.num_species)
        if self.training_objective not in {None, "direct_log1p", "mlpe"}:
            raise ValueError("Training objective provenance must be direct_log1p, mlpe or None")
        for name in ("training_pairs", "validation_pairs"):
            access = {
                region: tuple(tuple(sorted(pair)) for pair in pairs)
                for region, pairs in getattr(self, name).items()
            }
            setattr(self, name, access)

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

    def _regional_head(self, region: RegionBatch, species_index: int) -> RegionalMLPECalibration:
        self._validate_region(region)
        if not isinstance(self.calibration, MLPECalibrationModule):
            raise ValueError("This operation requires an MLPE model")
        index = species_index if self.distance_model.num_species > 1 else None
        head = self.calibration.for_region(region.name, index)
        kinds = region.sampling_unit_kinds
        assert kinds is not None
        if any(kind != "population" for kind in kinds):
            raise ValueError("Population MLPE prediction requires population sampling units")
        return head

    @overload
    def predict(
        self,
        region: RegionBatch,
        *,
        species: str | None = None,
        pairs: Sequence[tuple[str, str]] | None = None,
        known_effects: bool = False,
        support: PairwiseObservations | Sequence[PairwiseObservations] | None = None,
    ) -> Prediction: ...

    @overload
    def predict(
        self,
        region: RegionBatch,
        *,
        species: Sequence[str],
        pairs: Sequence[tuple[str, str]] | None = None,
        known_effects: bool = False,
        support: PairwiseObservations | Sequence[PairwiseObservations] | None = None,
    ) -> dict[str, Prediction]: ...

    def predict(
        self,
        region: RegionBatch,
        *,
        species: str | Sequence[str] | None = None,
        pairs: Sequence[tuple[str, str]] | None = None,
        known_effects: bool = False,
        support: PairwiseObservations | Sequence[PairwiseObservations] | None = None,
    ) -> Prediction | dict[str, Prediction]:
        """Genetic predictions on the original scale, mapped by species for a species sequence.

        Maximum-likelihood population effects (MLPE) modes ``known_effects=True``
        and ``support=observations`` are mutually exclusive, require MLPE
        calibration, and use saved or support-conditioned population effects.
        Their extras are ``model_values`` (model-scale values), ``model_variance``
        (effect variance plus residual variance and jitter), and ``effect_variance``
        (population-effect variance); residual variance and jitter remain on the
        model's calibration, and support pairs must be disjoint from calibration
        and query pairs.
        Values inverse-transform model means, variances stay on the model scale,
        and the distance model, calibration coefficients and variance parameters
        are held fixed.
        """
        if known_effects and support is not None:
            raise ValueError("known_effects and support are mutually exclusive")
        self._validate_region(region)
        selected = select_species(self.species, species)
        if self.distance_model.num_species > 1 and region.species is None:
            raise ValueError("Multispecies queries require sampling-unit species labels")
        support_by_species = self._support_observations(support, selected)
        distance_model = with_dropout_mode(self.distance_model, inference=True, key=None)
        latent = distance_model.backbone(region.feature_array)
        context = prepare_distance_context(distance_model, region.feature_array)
        results = {}
        for name, index in selected:
            prepared = region_for_species(region, name)
            head = None
            if (
                known_effects
                or support is not None
                or isinstance(self.calibration, MLPECalibrationModule)
            ):
                head = self._regional_head(prepared, index)
            observations = support_by_species.get(name)
            query_pairs = pairs
            if observations is not None:
                labels, _ = _query_pairs(prepared, pairs)
                query_pairs = labels + observations.observed_pairs
            labels, distances = query_distances_from_latent(
                distance_model,
                latent,
                prepared,
                query_pairs,
                species_index=index,
                context=context,
            )
            if observations is not None:
                labels = labels[: len(labels) - len(observations.observed_pairs)]
            extras = {}
            if known_effects or observations is not None:
                assert head is not None
                result = head.predict(
                    distances[: len(labels)],
                    labels,
                    known_effects=known_effects,
                    support=observations,
                    support_distances=None if observations is None else distances[len(labels) :],
                )
                values, extras = result.values, result.extras
            else:
                values = self.calibration.predict(
                    distances,
                    labels,
                    region_name=region.name,
                    species_index=index if distance_model.num_species > 1 else None,
                )
            results[name] = Prediction(
                jnp.asarray(values),
                labels,
                region.name,
                prepared.sampling_unit_ids,
                species=name,
                extras=extras,
            )
        if isinstance(species, str) or species is None:
            return results[selected[0][0]]
        return cast(dict[str, Prediction], results)

    def _support_observations(
        self,
        support: PairwiseObservations | Sequence[PairwiseObservations] | None,
        selected: tuple[tuple[str | None, int], ...],
    ) -> dict[str | None, PairwiseObservations]:
        if support is None:
            return {}
        observations = (support,) if isinstance(support, PairwiseObservations) else support
        if (
            not isinstance(observations, Sequence)
            or not observations
            or any(not isinstance(value, PairwiseObservations) for value in observations)
        ):
            raise ValueError("Support requires observations for each selected species")
        by_species = {value.species: value for value in observations}
        if len(by_species) != len(observations):
            raise ValueError("Support requires one observations object per species")
        if self.distance_model.num_species == 1 and set(by_species) == {None}:
            by_species = {selected[0][0]: observations[0]}
        if set(by_species) != {name for name, _ in selected}:
            raise ValueError("Support observations must match the selected species")
        return by_species
