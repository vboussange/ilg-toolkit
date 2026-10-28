"""Genetic calibration from ecological distances to original-scale predictions."""

from abc import abstractmethod
from typing import final, TYPE_CHECKING

import equinox as eqx

from ._checks import _require_float64
from ._training_utils import RegionSpeciesKey
from .data import ObservationCollection, PairwiseObservations, RegionCollection, TargetSpec
from .mlpe import RegionalMLPECalibration
from .mlpe.prediction import _predict_mlpe
from .prediction import Prediction


if TYPE_CHECKING:
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
        individual_based: bool = False,
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
        individual_based: bool = False,
        known_effects: bool = False,
        support: PairwiseObservations | None = None,
        support_distances=None,
    ) -> Prediction:
        """Original-scale genetic predictions from the inverse target transform."""
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
    """Group-specific maximum-likelihood population effects (MLPE) calibration.

    Each region–species group keeps its own fitted calibration. ``regions`` maps
    normalized ``(region_name, species_index)`` identities to fitted calibrations.
    Single-species models use species index zero.
    """

    target: TargetSpec
    regions: dict[RegionSpeciesKey, RegionalMLPECalibration] = eqx.field(default_factory=dict)

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
                    "MLPE calibration region names and target contracts must match the module"
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
        individual_based: bool = False,
        known_effects: bool = False,
        support: PairwiseObservations | None = None,
        support_distances=None,
    ) -> Prediction:
        """Genetic predictions from the requested region–species MLPE calibration.

        The ``model_values`` extra gives model-scale values; conditional modes
        also give ``model_variance`` and ``effect_variance`` on that scale.
        ``known_effects`` uses saved population effects, while ``support`` uses
        disjoint observations and aligned ``support_distances`` to condition them.
        """
        if individual_based:
            raise ValueError(
                "Population MLPE rejects individual-based regions: its effects model populations"
            )
        return _predict_mlpe(
            self.for_region(region_name, species_index),
            distances,
            pairs,
            known_effects=known_effects,
            support=support,
            support_distances=support_distances,
        )


def calibrate(
    region: RegionCollection,
    observations: ObservationCollection,
    distance_model: "AbstractDistanceModel",
    *,
    loss: "AbstractLoss",
    max_iterations: int | None = None,
) -> "LandscapeGeneticsModel":
    """Fit genetic calibration while preserving every distance-model parameter.

    Target semantics come from observations, and the supplied region–species groups
    receive fitted calibration. Distances are computed once
    per group in Equinox inference mode, and MLPE estimates both variance components
    with deterministic multistart optimization (2,000 iterations by default).
    ``Log1pLoss`` constructs a direct calibration and rejects an iteration budget.
    Species names come from the distance model.

    Args:
        region: One region batch or a sequence or mapping of region batches.
        observations: Genetic observations aligned with regions and their species.
        distance_model: Distance model whose parameters and species remain fixed
            during calibration.
        loss: Statistical criterion and numerical settings for genetic calibration.
        max_iterations: Per-start calibration optimization budget. MLPE defaults to
            2,000 iterations; direct calibration requires this to be omitted.

    Returns:
        A landscape genetics model with calibration for the supplied region–species groups.

    Notes:
        Earlier calibrations are not merged into the returned model.

    Examples:
        Train a patch encoder, then refit genetic calibration on the same region
        without further distance-model updates::

            loss = MLPELoss(jitter=0.0)
            state = train(region, observations, distance_model=distance_model,
                          config=TrainingConfig(loss=loss, steps=500))
            refitted = calibrate(region, observations,
                                 state.selected_model.distance_model, loss=loss,
                                 max_iterations=2000)
            prediction = refitted.predict(region, pairs=query_pairs)

        For parameter-free direct prediction, use ``loss=Log1pLoss()`` and omit
        ``max_iterations``. For updating only population effects of an existing MLPE
        calibration, see the worked modes in
        [`LandscapeGeneticsModel.predict`][ilg_toolkit.LandscapeGeneticsModel.predict]
        and the [comprehensive MLPE workflow notebook](../examples/mlpe_workflow.md).
    """
    from ._training_utils import _prepare_training_inputs
    from .losses import AbstractLoss
    from .model import LandscapeGeneticsModel
    from .models import AbstractDistanceModel
    from .models.base import (
        prepare_distance_context,
        query_distances_from_latent,
    )
    from .models.common import _inference_model

    _require_float64()
    if not isinstance(distance_model, AbstractDistanceModel):
        raise TypeError("distance_model must implement AbstractDistanceModel")
    inference_model = _inference_model(distance_model)
    if not isinstance(loss, AbstractLoss):
        raise TypeError("loss must implement AbstractLoss")
    prepared = _prepare_training_inputs(region, observations, distance_model.species)
    first, groups = prepared.inputs[0], prepared.training
    target = first.observations.target
    for group in groups:
        loss._prepare(group.region, group.observations, training=True)
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
        fitted[group.key] = loss._fit_group(
            distances,
            group.observations,
            region_name=group.region.name,
            max_iterations=max_iterations,
        )
    return LandscapeGeneticsModel(
        distance_model,
        loss._calibration(target, fitted),
        first.region.feature_names,
    )
