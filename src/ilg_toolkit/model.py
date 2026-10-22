"""Label-free ecological distances and explicit target-scale predictions."""

from typing import final

import equinox as eqx
import jax.numpy as jnp

from .calibration import AbstractCalibrationModule, MLPECalibrationModule
from .data import PairwiseObservations, RegionBatch, TargetSpec
from .mlpe import RegionalMLPECalibration
from .models import AbstractDistanceModel
from .prediction import Prediction


@final
class LandscapeGeneticsModel(eqx.Module):
    """Compose ecological distances and an authoritative genetic calibration module.

    ``distance_model`` derives ecological distances without genetic observations.
    ``calibration`` owns the target and converts selected distances into genetic
    predictions. Frozen calibration retains the entire distance model. Recorded
    pair access describes distance-model training and validation selection;
    regional calibrations retain their own pair identities and data roles.
    ``training_objective`` records the actual fitting objective, or is ``None`` for
    a composition built without a fit; it never selects inference behavior.
    """

    distance_model: AbstractDistanceModel
    calibration: AbstractCalibrationModule
    feature_count: int
    feature_names: tuple[str, ...] | None = None
    training_objective: str | None = eqx.field(default=None, kw_only=True)
    training_pairs: dict[str, tuple[tuple[str, str], ...]] = eqx.field(default_factory=dict)
    validation_pairs: dict[str, tuple[tuple[str, str], ...]] = eqx.field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.distance_model, AbstractDistanceModel):
            raise TypeError("distance_model must implement AbstractDistanceModel")
        if not isinstance(self.calibration, AbstractCalibrationModule):
            raise TypeError("calibration must implement AbstractCalibrationModule")
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

    def _regional_head(self, region: RegionBatch) -> RegionalMLPECalibration:
        self._validate_region(region)
        if not isinstance(self.calibration, MLPECalibrationModule):
            raise ValueError("This operation requires an MLPE model")
        head = self.calibration.for_region(region.name)
        kinds = region.sampling_unit_kinds
        assert kinds is not None
        if any(kind != "population" for kind in kinds):
            raise ValueError("Population MLPE prediction requires population sampling units")
        return head

    def predict(
        self,
        region: RegionBatch,
        *,
        species: str | None = None,
        pairs=None,
        known_effects: bool = False,
        support: PairwiseObservations | None = None,
    ) -> Prediction:
        """Genetic predictions on the original scale for query pairs, or all pairs if omitted.

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
        head = None
        if (
            known_effects
            or support is not None
            or isinstance(self.calibration, MLPECalibrationModule)
        ):
            head = self._regional_head(region)
        distances = self.distance_model.predict(region, species=species, pairs=pairs)
        if known_effects or support is not None:
            assert head is not None
            support_distances = None
            if support is not None:
                support_distances = self.distance_model.predict(
                    region, species=species, pairs=support.observed_pairs
                ).values
            result = head.predict(
                distances.values,
                distances.pairs,
                known_effects=known_effects,
                support=support,
                support_distances=support_distances,
            )
            return Prediction(
                result.values,
                distances.pairs,
                region.name,
                region.sampling_unit_ids,
                extras=result.extras,
            )
        values = self.calibration.predict(
            distances.values, distances.pairs, region_name=region.name
        )
        return Prediction(
            jnp.asarray(values), distances.pairs, region.name, region.sampling_unit_ids
        )
