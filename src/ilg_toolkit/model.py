"""Label-free ecological distances and explicit target-scale predictions."""

from collections.abc import Sequence
from typing import cast, final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .calibration import AbstractCalibrationModule, MLPECalibrationModule
from .data import ObservationPartition, PairwiseObservations, RegionBatch, TargetSpec
from .mlpe import MLPEConditionalPrediction, RegionalMLPECalibration
from .models import AbstractDistanceModel, ConductanceModel
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

    def ecological_distances(self, region: RegionBatch) -> jax.Array:
        """Return pairwise ecological distances at query locations without genetic observations."""
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

    def _distances_for_pairs(self, region: RegionBatch, pairs) -> jax.Array:
        self._validate_region(region)
        indices = _pair_indices(region, pairs)
        selection = self.distance_model._prepare_pairs(
            region.feature_array, region.pixel_nodes, indices
        )
        context = self.distance_model.prepare(region.feature_array)
        values = self.distance_model._pair_distances(
            region.feature_array, selection, context=context
        )
        if not bool(jnp.isfinite(values).all()) or bool((values < 0).any()):
            raise FloatingPointError("Distance model produced nonfinite or negative distances")
        return values

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

    def predict(self, region: RegionBatch, pairs=None) -> Prediction:
        """Return original-scale predictions; MLPE uses the region's marginal head.

        No query targets are accepted. An unseen MLPE region requires explicit
        calibration even though its ecological distances can already be computed.

        Args:
            region: Query features and locations matching the model's feature
                contract; no genetic observations are required.
            pairs: Optional nonempty unique unordered pair selection. Values
                retain caller order and orientation. Omission requests every
                distinct pair in region order.

        Returns:
            Original-scale pair points retaining the full region sampling-unit
            order. Use ``as_matrix()`` for explicit matrix representation.
            Requested distances are selected before calibration or inverse
            transformation, excluding unrequested transform failures.

        Raises:
            ValueError: Features, pair labels, locations or regional calibration
                are incompatible, or pairs are empty, self-pairs or duplicates.
            FloatingPointError: Distance prediction or inverse transformation
                produces invalid numerical values.
        """
        if pairs is None:
            left, right = np.triu_indices(len(region.sampling_unit_ids), 1)
            pairs = tuple(
                (region.sampling_unit_ids[i], region.sampling_unit_ids[j])
                for i, j in zip(left, right, strict=True)
            )
        self._validate_region(region)
        try:
            raw_pairs = tuple(pairs)
            if any(isinstance(pair, str) for pair in raw_pairs):
                raise ValueError("Each pair requires two labels")
            pairs = cast(tuple[tuple[str, str], ...], tuple(tuple(pair) for pair in raw_pairs))
            ObservationPartition(region.name, pairs, role="query")
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Pairs require distinct labels and unique unordered identities"
            ) from error
        if isinstance(self.calibration, MLPECalibrationModule):
            self._regional_head(region)
        distances = self._distances_for_pairs(region, pairs)
        values = self.calibration.predict(distances, pairs, region_name=region.name)
        return Prediction(
            jnp.asarray(values), pairs, region.name, self.target, region.sampling_unit_ids
        )

    def calibrate(self, region: RegionBatch, observations, *, partitions=None, config=None):
        """Fit or replace regional MLPE calibration with this distance model frozen."""
        from .calibration import calibrate

        return calibrate(self, region, observations, partitions=partitions, config=config)

    def predict_known_effects(self, region: RegionBatch, pairs) -> MLPEConditionalPrediction:
        """Predict conditional genetic points using saved MLPE population effects.

        This explicit deployment operation applies no OOF holdout rule. Saved
        calibration targets remain actual access; populations absent from the
        posterior receive independent prior effects. No query targets are read.

        Args:
            region: Prepared query features and locations matching this model's
                feature contract and regional population MLPE calibration.
            pairs: Nonempty unique unordered labelled pairs with located
                endpoints, retaining caller order/orientation. Self-pairs and
                reversed duplicates are rejected.

        Returns:
            ``MLPEConditionalPrediction`` composing original-scale points
            ``(m,)`` and full region sampling-unit order, model-scale means and
            predictive variances ``(m,)``, variance components and actual access
            provenance. Variance includes effect-posterior, residual and jitter
            contributions, excluding distance-model, coefficient and variance-
            parameter uncertainty. Nonlinear inverse-transformed means are not
            original-scale expectations. Conditional calculations use scoped
            float64; effective resistances require enabled x64. Global precision,
            model parameters and stored posterior remain unchanged.

        Raises:
            ValueError: Regional MLPE calibration, population kinds, feature
                contract, pairs or numerical posterior are incompatible.
            FloatingPointError: Distance or inverse target calculations fail.
        """
        head = self._regional_head(region)
        pairs = tuple(pairs)
        distances = self._distances_for_pairs(region, pairs)
        result = head.predict_known_effects(distances, pairs)
        return eqx.tree_at(
            lambda value: value.prediction.sampling_unit_ids, result, region.sampling_unit_ids
        )

    def predict_with_support(
        self,
        region: RegionBatch,
        pairs,
        support_observations: PairwiseObservations,
        *,
        support_partition: ObservationPartition,
    ) -> MLPEConditionalPrediction:
        """Condition saved MLPE effects on declared support, then predict queries.

        Support supplies additional measured targets; query targets are never
        inputs. This deployment operation imposes no endpoint holdout rule. It
        leaves distance parameters, regional calibration and saved effect
        posterior unchanged; returned provenance records the additional access.

        Args:
            region: Prepared features and locations for every support/query
                endpoint, matching the feature contract and regional population
                MLPE calibration.
            pairs: Nonempty unique unordered query pairs with located endpoints;
                caller order/orientation is retained. Queries must be disjoint
                from support identities, including reversed pairs.
            support_observations: Original-scale ``PairwiseObservations`` with
                the calibrated ``TargetSpec`` and exactly the declared support
                measurements. No calibration pair may be consumed again.
            support_partition: Matching regional ``ObservationPartition`` with
                role ``"support"`` and exactly the measured support identities.

        Returns:
            ``MLPEConditionalPrediction`` composing original-scale query points
            ``(m,)`` and full region matrix order, model-scale means/variances
            ``(m,)``, variance components and calibration/support provenance.
            Conditioning holds distance parameters, coefficients and variances
            fixed; reported variance includes effect-posterior, residual and
            jitter contributions only. Nonlinear inverse-transformed means are
            not original-scale expectations. Conditional calculations use scoped
            float64 without changing global precision; resistance requires
            explicitly enabled x64 before distance calculation.

        Raises:
            ValueError: Features, regional calibration, population kinds, pair
                labels, support target/role/set, calibration reuse, support/query
                separation or numerical posterior are incompatible.
            FloatingPointError: Distance or inverse target calculations fail.
        """
        head = self._regional_head(region)
        pairs = tuple(pairs)
        support_pairs = support_observations.observed_pairs
        distances = self._distances_for_pairs(region, support_pairs + pairs)
        support_distances = distances[: len(support_pairs)]
        conditioned = head.condition_on_support(
            support_distances, support_observations, partition=support_partition
        )
        result = conditioned.predict(distances[len(support_pairs) :], pairs)
        return eqx.tree_at(
            lambda value: value.prediction.sampling_unit_ids, result, region.sampling_unit_ids
        )


def _pair_indices(region: RegionBatch, pairs: Sequence[tuple[str, str]]):
    lookup = {label: index for index, label in enumerate(region.sampling_unit_ids)}
    try:
        if not pairs:
            raise ValueError("At least one pair is required")
        indices = jnp.asarray([(lookup[a], lookup[b]) for a, b in pairs], dtype=jnp.int32)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Pairs must contain two sampling-unit labels with locations in the query region"
        ) from error
    return indices[:, 0], indices[:, 1]
