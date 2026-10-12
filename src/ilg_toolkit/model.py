"""Label-free landscape scoring and explicit target-scale predictions."""

from collections.abc import Sequence
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .calibration import AbstractCalibrationModule, MLPECalibrationModule
from .data import ObservationPartition, PairwiseObservations, RegionBatch, TargetSpec
from .mlpe import MLPEConditionalPrediction, RegionalMLPECalibration
from .models import AbstractDistanceModel, ConductanceModel


@final
class Prediction(eqx.Module):
    """Pairwise predictions with their sampling-unit order and declared scale."""

    values: jax.Array
    sampling_unit_ids: tuple[str, ...]
    target: TargetSpec
    scale: str = "original"


@final
class PairPrediction(eqx.Module):
    """Marginal predictions for explicit labelled pairs in the requested order."""

    values: jax.Array
    pairs: tuple[tuple[str, str], ...]
    target: TargetSpec
    region_name: str
    scale: str = "original"


@final
class LandscapeGeneticsModel(eqx.Module):
    """Compose landscape distances and an authoritative genetic calibration module.

    ``distance_model`` derives scores without genetic observations. ``calibration``
    owns the target and converts selected scores into genetic predictions. Frozen
    calibration retains the entire distance model. Recorded pair access describes
    distance-model training and validation selection; regional calibrations retain
    their own pair identities and data roles. ``training_objective`` records the
    actual fitting objective, or is ``None`` for a composition built without a fit;
    it never selects inference behavior.
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

    def landscape_scores(self, region: RegionBatch) -> jax.Array:
        """Predict scores for prepared query locations without genetic observations."""
        self._validate_region(region)
        context = self.distance_model.prepare(region.feature_array)
        values = self.distance_model.predict_distances(
            region.feature_array, region.pixel_nodes, context=context
        )
        if not bool(jnp.isfinite(values).all()) or bool((values < 0).any()):
            raise FloatingPointError("Distance model produced nonfinite or negative distances")
        return values

    def conductance_surface(self, region: RegionBatch) -> jax.Array:
        """Expose a fitted conductance surface separately from genetic predictions."""
        if not isinstance(self.distance_model, ConductanceModel):
            raise TypeError("This distance model does not produce a conductance surface")
        self._validate_region(region)
        surface = self.distance_model.conductance(region.feature_array)
        if not bool(jnp.all(jnp.isfinite(surface) & (surface > 0))):
            raise FloatingPointError("Distance model produced nonfinite or nonpositive conductance")
        return surface

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

    def predict(self, region: RegionBatch) -> Prediction:
        """Return original-scale predictions; MLPE uses the region's marginal head.

        No query targets are accepted. An unseen MLPE region requires explicit
        calibration even though its landscape scores can already be computed.
        """
        count = len(region.sampling_unit_ids)
        left, right = np.triu_indices(count, 1)
        pairs = tuple(
            (region.sampling_unit_ids[i], region.sampling_unit_ids[j])
            for i, j in zip(left, right, strict=True)
        )
        marginal = self.predict_pairs(region, pairs)
        values = jnp.zeros((count, count), dtype=marginal.values.dtype)
        values = values.at[left, right].set(marginal.values)
        values = values.at[right, left].set(marginal.values)
        return Prediction(values, region.sampling_unit_ids, self.target)

    def predict_pairs(self, region: RegionBatch, pairs) -> PairPrediction:
        """Predict a nonempty selection of unique pairs without query targets.

        Preserve the requested order and orientation. Select landscape scores
        before regional calibration or target inversion, so unrequested pairs
        cannot cause an inverse-transform failure. MLPE predictions are marginal.
        """
        self._validate_region(region)
        try:
            raw_pairs = tuple(pairs)
            if any(isinstance(pair, str) for pair in raw_pairs):
                raise ValueError("Each pair requires two labels")
            pairs = tuple(tuple(pair) for pair in raw_pairs)
            ObservationPartition(region.name, pairs, role="query")
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Pairs require distinct labels and unique unordered identities"
            ) from error
        if isinstance(self.calibration, MLPECalibrationModule):
            self._regional_head(region)
        scores = _scores_for_pairs(self.landscape_scores(region), region, pairs)
        values = self.calibration.predict(scores, pairs, region_name=region.name)
        return PairPrediction(jnp.asarray(values), pairs, self.target, region.name)

    def calibrate(self, region: RegionBatch, observations, *, partitions=None, config=None):
        """Fit or replace regional MLPE calibration with this distance model frozen."""
        from .calibration import calibrate

        return calibrate(self, region, observations, partitions=partitions, config=config)

    def predict_known_effects(self, region: RegionBatch, pairs) -> MLPEConditionalPrediction:
        """Predict explicit labelled pairs using stored regional population effects."""
        head = self._regional_head(region)
        pairs = tuple(pairs)
        scores = _scores_for_pairs(self.landscape_scores(region), region, pairs)
        return head.predict_known_effects(scores, pairs)

    def predict_with_support(
        self,
        region: RegionBatch,
        pairs,
        support_observations: PairwiseObservations,
        *,
        support_partition: ObservationPartition,
    ) -> MLPEConditionalPrediction:
        """Predict disjoint queries using only declared support genetic observations.

        The prepared region supplies locations for every support and query endpoint.
        This operation leaves the distance model and stored regional calibration unchanged.
        """
        head = self._regional_head(region)
        pairs = tuple(pairs)
        scores = self.landscape_scores(region)
        support_scores = _scores_for_pairs(scores, region, support_observations.observed_pairs)
        conditioned = head.condition_on_support(
            support_scores, support_observations, partition=support_partition
        )
        return conditioned.predict(_scores_for_pairs(scores, region, pairs), pairs)


def _scores_for_pairs(
    scores: jax.Array, region: RegionBatch, pairs: Sequence[tuple[str, str]]
) -> jax.Array:
    lookup = {label: index for index, label in enumerate(region.sampling_unit_ids)}
    try:
        if not pairs:
            raise ValueError("At least one pair is required")
        indices = jnp.asarray([(lookup[a], lookup[b]) for a, b in pairs], dtype=jnp.int32)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Pairs must contain two sampling-unit labels with locations in the query region"
        ) from error
    return scores[indices[:, 0], indices[:, 1]]
