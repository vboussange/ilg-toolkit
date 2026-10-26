"""Genetic calibration from ecological distances to original-scale predictions."""

from abc import abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, final

import equinox as eqx

from .data import PairwiseObservations, TargetSpec
from .losses import GroupKey
from .mlpe import RegionalMLPECalibration
from .prediction import Prediction

if TYPE_CHECKING:
    from ._groups import ObservationCollection, RegionCollection
    from .losses import AbstractLoss
    from .model import LandscapeGeneticsModel
    from .models import AbstractDistanceModel


class AbstractCalibrationModule(eqx.Module):
    """Abstract fitted genetic prediction behavior and target contract."""

    target: eqx.AbstractVar[TargetSpec]

    @abstractmethod
    def predict(
        self,
        distances,
        pairs,
        *,
        region_name: str,
        species_index: int = 0,
        sampling_unit_kinds: tuple[str, ...] | None = None,
        known_effects: bool = False,
        support: PairwiseObservations | None = None,
        support_distances=None,
    ) -> Prediction:
        """Genetic predictions and statistical extras for pair-aligned ecological distances."""


@final
class DirectCalibrationModule(AbstractCalibrationModule):
    """Direct genetic calibration applying the inverse target transform."""

    target: TargetSpec

    def __post_init__(self):
        if not isinstance(self.target, TargetSpec):
            raise ValueError("Direct calibration requires a TargetSpec")

    def predict(
        self,
        distances,
        pairs,
        *,
        region_name: str,
        species_index: int = 0,
        sampling_unit_kinds: tuple[str, ...] | None = None,
        known_effects: bool = False,
        support: PairwiseObservations | None = None,
        support_distances=None,
    ) -> Prediction:
        if known_effects or support is not None or support_distances is not None:
            raise ValueError("Direct calibration does not support population-effect conditioning")
        return Prediction(
            self.target.inverse(distances),
            pairs,
            region_name,
            tuple(dict.fromkeys(label for pair in pairs for label in pair)),
        )


@final
class MLPECalibrationModule(AbstractCalibrationModule):
    """Unpooled maximum-likelihood population effects (MLPE) genetic calibration.

    ``regions`` maps normalized ``(region_name, species_index)`` identities to
    fitted calibrations. Single-species models use species index zero.
    """

    target: TargetSpec
    regions: dict[GroupKey, RegionalMLPECalibration] = eqx.field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.target, TargetSpec) or self.target.kind != "dissimilarity":
            raise ValueError("Population MLPE requires a dissimilarity target")
        regions = dict(self.regions)
        if any(not isinstance(value, RegionalMLPECalibration) for value in regions.values()):
            raise ValueError("MLPE regions require fitted RegionalMLPECalibration values")
        for key, value in regions.items():
            if (
                not isinstance(key, tuple)
                or len(key) != 2
                or not isinstance(key[0], str)
                or not key[0]
                or type(key[1]) is not int
                or key[1] < 0
            ):
                raise ValueError("MLPE calibration keys require a region and species index")
            if key[0] != value.region_name or value.target != self.target:
                raise ValueError(
                    "Regional calibration names and target contracts must match the module"
                )
        self.regions = regions

    def for_region(self, region_name: str, species_index: int = 0) -> RegionalMLPECalibration:
        """Fitted MLPE calibration for one region–species group."""
        key = (region_name, species_index)
        value = self.regions.get(key)
        if value is None:
            raise ValueError(f"Region–species group {key!r} has no MLPE calibration; use calibrate")
        return value

    def predict(
        self,
        distances,
        pairs,
        *,
        region_name: str,
        species_index: int = 0,
        sampling_unit_kinds: tuple[str, ...] | None = None,
        known_effects: bool = False,
        support: PairwiseObservations | None = None,
        support_distances=None,
    ) -> Prediction:
        if sampling_unit_kinds is not None and any(
            kind != "population" for kind in sampling_unit_kinds
        ):
            raise ValueError("Population MLPE prediction requires population sampling units")
        return self.for_region(region_name, species_index).predict(
            distances,
            pairs,
            known_effects=known_effects,
            support=support,
            support_distances=support_distances,
        )


def calibrate(
    region: "RegionCollection",
    observations: "ObservationCollection",
    distance_model: "AbstractDistanceModel",
    *,
    loss: "AbstractLoss",
    species: Sequence[str] | None = None,
    max_iterations: int | None = None,
) -> "LandscapeGeneticsModel":
    """Fit genetic calibration while preserving every distance-model parameter.

    Target semantics come from observations. Only supplied region–species groups
    are fitted; earlier calibrations are not merged. Distances are computed once
    per group with dropout disabled, and MLPE estimates both variance components
    with deterministic multistart optimization (2,000 iterations by default).
    ``Log1pLoss`` constructs a direct calibration and rejects an iteration budget.
    Transferring a multispecies distance model requires its original full species
    order, even when observations cover only one species.

    Examples
    --------
    Train a landscape relationship, then fit new genetic calibration without
    further landscape updates::

        loss = MLPELoss(jitter=0.0)
        state = train(region, observations, distance_model=distance_model,
                      config=TrainingConfig(loss=loss, steps=500))
        adapted = calibrate(new_region, new_observations,
                            state.selected_model.distance_model, loss=loss,
                            species=state.selected_model.species, max_iterations=2000)
        prediction = adapted.predict(new_region, pairs=query_pairs)

    For parameter-free direct prediction, use ``loss=Log1pLoss()`` and omit
    ``max_iterations``. For updating only population effects of an existing MLPE
    calibration, see the worked modes in
    [`LandscapeGeneticsModel.predict`][ilg_toolkit.LandscapeGeneticsModel.predict]
    and the [comprehensive MLPE workflow notebook](../examples/mlpe_workflow.md).
    """
    from ._groups import _index_inputs, _normalize_inputs
    from .losses import AbstractLoss
    from .model import LandscapeGeneticsModel
    from .models import AbstractDistanceModel
    from .models.base import (
        prepare_distance_context,
        query_distances_from_latent,
        validate_species_order,
    )
    from .models.common import with_dropout_mode

    if not isinstance(distance_model, AbstractDistanceModel):
        raise TypeError("distance_model must implement AbstractDistanceModel")
    if not isinstance(loss, AbstractLoss):
        raise TypeError("loss must implement AbstractLoss")
    inputs = _normalize_inputs(region, observations)
    if species is None and distance_model.num_species == 1:
        observed_names = {
            value.observations.species for value in inputs if value.observations.species is not None
        }
        observed_names.update(name for value in inputs for name in (value.region.species or ()))
        if len(observed_names) == 1:
            species = tuple(observed_names)
    order = validate_species_order(species, distance_model.num_species)
    groups = _index_inputs(inputs, order, distance_model.num_species)
    first = groups[0]
    target = first.observations.target
    feature_count, feature_names = first.region.feature_array.shape[-1], first.region.feature_names
    if len({group.region.name for group in groups}) > 1 and feature_names is None:
        raise ValueError("Multi-region calibration requires feature_names meanings and order")
    for group in groups:
        if group.observations.target != target:
            raise ValueError("Calibration observations must share one target contract")
        if (
            group.region.feature_array.shape[-1] != feature_count
            or group.region.feature_names != feature_names
        ):
            raise ValueError("Calibration regions must share feature meanings and order")
        loss._prepare(group.region, group.observations, training=True)
    inference_model = with_dropout_mode(distance_model, inference=True, key=None)
    prepared_regions = {}
    fitted = {}
    for group in groups:
        if group.region.name not in prepared_regions:
            features = group.region.feature_array
            prepared_regions[group.region.name] = (
                inference_model.backbone(features),
                prepare_distance_context(inference_model, features),
            )
        latent, context = prepared_regions[group.region.name]
        _, distances = query_distances_from_latent(
            inference_model,
            latent,
            group.region,
            group.observations.observed_pairs,
            species_index=group.species_index,
            context=context,
        )
        fitted[group.identity] = loss._fit_group(
            distances,
            group.observations,
            region_name=group.region.name,
            max_iterations=max_iterations,
        )
    return LandscapeGeneticsModel(
        distance_model,
        loss._calibration(target, fitted),
        feature_count,
        feature_names=feature_names,
        species=order,
    )
