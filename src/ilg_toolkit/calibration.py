"""Calibration heads mapping ecological distances to original-scale genetic predictions."""

from abc import abstractmethod
from collections.abc import Mapping, Sequence
from typing import final, TYPE_CHECKING

import equinox as eqx
import jax

from ._checks import _require_float64
from .data import PairwiseObservations, RegionBatch, TargetSpec
from .mlpe import MLPECalibration
from .mlpe.prediction import _predict_mlpe
from .prediction import Prediction


if TYPE_CHECKING:
    from .losses import AbstractLoss
    from .models import AbstractDistanceModel


class AbstractCalibrationHead(eqx.Module):
    """Abstract calibration head mapping ecological distances to genetic predictions.

    A calibration head is the part of a landscape genetics model that owns the
    target contract and returns genetic predictions on the original scale.
    [`DirectCalibrationHead`][ilg_toolkit.DirectCalibrationHead] and
    [`MLPECalibrationHead`][ilg_toolkit.MLPECalibrationHead] implement it;
    [`fit_calibration_head`][ilg_toolkit.fit_calibration_head] fits one for a frozen
    distance model, and training produces one through its loss.

    Attributes:
        target: Target contract of the genetic observations.
    """

    target: eqx.AbstractVar[TargetSpec]

    @abstractmethod
    def predict(
        self,
        distances: jax.Array,
        pairs: Sequence[tuple[str, str]],
        *,
        region_name: str,
        species: str | None = None,
        individual_based: bool = False,
        known_effects: bool = False,
        support: PairwiseObservations | None = None,
        support_distances: jax.Array | None = None,
    ) -> Prediction:
        """Returns genetic predictions from pair-aligned ecological distances.

        Args:
            distances: Ecological distances, one per pair in ``pairs``.
            pairs: Ordered sampling-unit pairs of the region.
            region_name: Name of the region holding the pairs.
            species: Species of the sampling units, from the distance model's
                ``species``; ``None`` for a distance model without species names.
            individual_based: Whether the region's sampling units are individuals.
            known_effects: Whether to include the saved population-effect posterior.
            support: Observations, disjoint from calibration and query pairs, that
                condition the saved population-effect posterior.
            support_distances: Ecological distances aligned with
                ``support.observed_pairs``.

        Returns:
            Original-scale genetic predictions aligned with ``pairs``.
        """


@final
class DirectCalibrationHead(AbstractCalibrationHead):
    """Calibration head applying the inverse target transform in any region.

    Training and [`fit_calibration_head`][ilg_toolkit.fit_calibration_head] with
    [`Log1pLoss`][ilg_toolkit.Log1pLoss] produce this calibration head. It has no
    fitted parameters and predicts in any region, including regions without
    genetic observations.

    Args:
        target: Target contract of the genetic observations.

    Raises:
        ValueError: If ``target`` is not a ``TargetSpec``.
    """

    target: TargetSpec

    def __check_init__(self):
        if not isinstance(self.target, TargetSpec):
            raise ValueError("A direct calibration head requires a TargetSpec")

    def predict(
        self,
        distances: jax.Array,
        pairs: Sequence[tuple[str, str]],
        *,
        region_name: str,
        species: str | None = None,
        individual_based: bool = False,
        known_effects: bool = False,
        support: PairwiseObservations | None = None,
        support_distances: jax.Array | None = None,
    ) -> Prediction:
        """Returns genetic predictions as the inverse target transform of the distances.

        Args:
            distances: Ecological distances, one per pair in ``pairs``.
            pairs: Ordered sampling-unit pairs of the region.
            region_name: Name of the region holding the pairs.
            species: Unused; the direct calibration head applies to every species.
            individual_based: Unused; the direct calibration head applies to
                individuals and populations.
            known_effects: Must be ``False``.
            support: Must be ``None``.
            support_distances: Must be ``None``.

        Returns:
            Original-scale genetic predictions aligned with ``pairs``.

        Raises:
            ValueError: If ``known_effects``, ``support`` or ``support_distances``
                requests population-effect conditioning.
        """
        if known_effects or support is not None or support_distances is not None:
            raise ValueError(
                "A direct calibration head does not support population-effect conditioning"
            )
        return Prediction(
            self.target.inverse(distances),
            tuple(pairs),
            region_name,
            tuple(dict.fromkeys(label for pair in pairs for label in pair)),
        )


@final
class MLPECalibrationHead(AbstractCalibrationHead):
    """Calibration head holding one MLPE calibration per region–species group.

    Training and [`fit_calibration_head`][ilg_toolkit.fit_calibration_head] with
    [`MLPELoss`][ilg_toolkit.MLPELoss] produce this calibration head. Each
    [`MLPECalibration`][ilg_toolkit.mlpe.MLPECalibration] is fitted on the model
    scale of the calibration head's target, and genetic predictions use the MLPE
    calibration of the queried region and species.

    Args:
        target: Dissimilarity target contract shared by every region–species group.
        calibrations: MLPE calibrations keyed by region name and species name, or
            by region name and ``None`` for a distance model without species names.

    Raises:
        ValueError: If ``target`` is not a dissimilarity ``TargetSpec``, or a key or
            value of ``calibrations`` is invalid.

    Examples:
        Inspect the calibration of one region–species group::

            head = fit_calibration_head(region, observations, distance_model,
                                        loss=MLPELoss())
            calibration = head.for_region(region.name, "frog")
            calibration.slope, calibration.ml_log_likelihood
    """

    target: TargetSpec
    calibrations: dict[tuple[str, str | None], MLPECalibration]

    def __init__(
        self,
        target: TargetSpec,
        calibrations: dict[tuple[str, str | None], MLPECalibration],
    ):
        if not isinstance(target, TargetSpec) or target.kind != "dissimilarity":
            raise ValueError("An MLPE calibration head requires a dissimilarity TargetSpec")
        if not isinstance(calibrations, dict):
            raise ValueError("MLPE calibrations must be a mapping keyed by region and species")
        for key, value in calibrations.items():
            if (
                not isinstance(key, tuple)
                or len(key) != 2
                or not isinstance(key[0], str)
                or not key[0]
                or not (key[1] is None or isinstance(key[1], str))
            ):
                raise ValueError(
                    "MLPE calibration keys require a region name and a species name or None"
                )
            if not isinstance(value, MLPECalibration):
                raise ValueError("MLPE calibration head values must be MLPECalibration records")
        self.target = target
        self.calibrations = dict(calibrations)

    def for_region(self, region_name: str, species: str | None = None) -> MLPECalibration:
        """Returns the MLPE calibration of one region–species group.

        Args:
            region_name: Name of a region the calibration head was fitted on.
            species: Species name; optional when every MLPE calibration of the
                calibration head belongs to one species.

        Returns:
            The region–species group's MLPE calibration.

        Raises:
            ValueError: If the calibration head holds several species and
                ``species`` is ``None``, or the region–species group has no MLPE
                calibration.
        """
        if species is None:
            names = {name for _, name in self.calibrations}
            if len(names) > 1:
                raise ValueError(
                    f"Name one of the species {sorted(names, key=str)!r} to look up a "
                    "multispecies MLPE calibration"
                )
            species = next(iter(names), None)
        calibration = self.calibrations.get((region_name, species))
        if calibration is None:
            raise ValueError(
                f"Region {region_name!r}, species {species!r} has no MLPE calibration; "
                "fit one with fit_calibration_head"
            )
        return calibration

    def predict(
        self,
        distances: jax.Array,
        pairs: Sequence[tuple[str, str]],
        *,
        region_name: str,
        species: str | None = None,
        individual_based: bool = False,
        known_effects: bool = False,
        support: PairwiseObservations | None = None,
        support_distances: jax.Array | None = None,
    ) -> Prediction:
        """Returns genetic predictions from the region–species group's MLPE calibration.

        The ``model_values`` extra gives model-scale means; conditional modes also
        give ``model_variance`` and ``effect_variance`` on that scale.

        Args:
            distances: Ecological distances, one per pair in ``pairs``.
            pairs: Ordered sampling-unit pairs of the region.
            region_name: Name of a region the calibration head was fitted on.
            species: Species of the sampling units; optional when every MLPE
                calibration of the calibration head belongs to one species.
            individual_based: Must be ``False``.
            known_effects: Whether to include the saved population-effect posterior.
            support: Observations with the calibration head's target, disjoint from
                calibration and query pairs, that condition the saved
                population-effect posterior.
            support_distances: Ecological distances aligned with
                ``support.observed_pairs``.

        Returns:
            Original-scale genetic predictions aligned with ``pairs``.

        Raises:
            ValueError: If the region is individual-based or the region–species group
                has no MLPE calibration.
            MLPEError: If the distances, pairs or support are invalid.
        """
        if individual_based:
            raise ValueError("MLPE calibration heads reject individual-based region batches")
        return _predict_mlpe(
            self.for_region(region_name, species),
            distances,
            pairs,
            target=self.target,
            region_name=region_name,
            known_effects=known_effects,
            support=support,
            support_distances=support_distances,
        )


def fit_calibration_head(
    region: RegionBatch | Sequence[RegionBatch] | Mapping[str, RegionBatch],
    observations: PairwiseObservations
    | Sequence[PairwiseObservations | Sequence[PairwiseObservations]]
    | Mapping[str, PairwiseObservations | Sequence[PairwiseObservations]],
    distance_model: "AbstractDistanceModel",
    *,
    loss: "AbstractLoss",
    max_iterations: int | None = None,
) -> AbstractCalibrationHead:
    """Fits a calibration head for a frozen distance model.

    The distance model is evaluated in Equinox inference mode and stays unchanged,
    including its backbone and species-specific heads. Species names come from the
    distance model, and the target contract from the observations. With
    [`MLPELoss`][ilg_toolkit.MLPELoss], the calibration head holds one MLPE
    calibration per supplied region–species group, whose coefficients and variance
    components come from deterministic multistart optimization (2,000 iterations
    per start by default). With [`Log1pLoss`][ilg_toolkit.Log1pLoss], it is a
    direct calibration head.

    Args:
        region: One region batch, a sequence of region batches, or a mapping keyed
            by region name.
        observations: Genetic observations aligned with the regions; each region
            may supply observations for several species.
        distance_model: Frozen distance model giving the ecological distances.
        loss: Loss defining the calibration head and its numerical settings.
        max_iterations: Iteration budget of each MLPE optimization start; ``None``
            for a direct calibration head.

    Returns:
        The calibration head; compose it with the distance model in a
        [`LandscapeGeneticsModel`][ilg_toolkit.LandscapeGeneticsModel].

    Raises:
        TypeError: If the distance model or the loss has an unsupported type, or
            the distance model has stateful layers.
        ValueError: If the regions or observations are invalid, a raster is not a
            multiple of the distance model's patch size, ``max_iterations`` is set
            for a direct calibration head, or MLPE receives individual-based regions.
        MLPEError: If an MLPE calibration cannot be fitted.
        RuntimeError: If float64 is disabled.

    Examples:
        Train a patch encoder, then refit its calibration head on the training
        region without further distance-model updates::

            loss = MLPELoss(jitter=0.0)
            state = train(region, observations, distance_model=distance_model,
                          config=TrainingConfig(loss=loss, steps=500))
            distance_model = state.selected_model.distance_model
            head = fit_calibration_head(region, observations, distance_model,
                                        loss=loss, max_iterations=2000)
            model = LandscapeGeneticsModel(distance_model, head, region.feature_names)
            prediction = model.predict(region, query_pairs)

        To compare the ecological distances of an IBD model and a patch encoder, fit
        both calibration heads on the same observations with the same loss and
        iteration budget, and compare
        ``head.for_region(region.name).ml_log_likelihood``.
        [`LandscapeGeneticsModel.predict`][ilg_toolkit.LandscapeGeneticsModel.predict]
        conditions population effects without refitting.
    """
    from ._training_utils import _prepare_training_inputs
    from .losses import AbstractLoss
    from .models import AbstractDistanceModel
    from .models.base import (
        _check_regions,
        _prepare_distance_context,
        _query_distances_from_latent,
    )
    from .models.common import _inference_model

    _require_float64()
    if not isinstance(distance_model, AbstractDistanceModel):
        raise TypeError("distance_model must implement AbstractDistanceModel")
    inference_model = _inference_model(distance_model)
    if not isinstance(loss, AbstractLoss):
        raise TypeError("loss must implement AbstractLoss")
    prepared = _prepare_training_inputs(region, observations, distance_model.species)
    _check_regions(distance_model, prepared.regions)
    groups = prepared.training
    for group in groups:
        loss._prepare(group.region, group.observations, training=True)
    prepared_regions = {}
    fitted = {}
    for group in groups:
        if group.region.name not in prepared_regions:
            features = group.region.feature_array
            prepared_regions[group.region.name] = (
                inference_model.backbone(features),
                _prepare_distance_context(inference_model, features),
            )
        latent, context = prepared_regions[group.region.name]
        _, distances = _query_distances_from_latent(
            inference_model,
            latent,
            group.region,
            group.observations.observed_pairs,
            species_index=group.species_index,
            context=context,
        )
        fitted[group.key] = loss._fit_group(
            distances, group.observations, max_iterations=max_iterations
        )
    return loss._calibration_head(
        prepared.inputs[0].observations.target, fitted, distance_model.species
    )
