"""Label-free landscape scoring and explicit target-scale predictions."""

from dataclasses import dataclass, field

import numpy as np

from .config import ResistanceSolverConfig
from .data import ObservationPartition, PairwiseObservations, RegionBatch, TargetSpec
from .mlpe import MLPEConditionalPrediction, MLPEHead
from .models import ConductanceModel, EmbeddingDistanceModel
from .resistance import build_resistance_context


@dataclass(frozen=True)
class Prediction:
    """Pairwise predictions with their sampling-unit order and declared scale."""

    values: np.ndarray
    sampling_unit_ids: tuple[str, ...]
    target: TargetSpec
    scale: str = "original"


@dataclass(frozen=True)
class PairPrediction:
    """Marginal predictions for explicit labelled pairs in the requested order."""

    values: np.ndarray
    pairs: tuple[tuple[str, str], ...]
    target: TargetSpec
    region_name: str
    scale: str = "original"


@dataclass(frozen=True)
class CalibratedModel:
    """Frozen landscape encoder with explicit direct or regional MLPE prediction.

    The cleaned encoder families are stateless apart from their Equinox model
    parameters. Recalibration retains that entire encoder unchanged. Recorded
    pair access describes encoder training and validation-based selection; head
    calibration retains its own independent pair identities and data roles.
    """

    encoder: ConductanceModel | EmbeddingDistanceModel
    target: TargetSpec
    feature_count: int
    feature_names: tuple[str, ...] | None = None
    solver_config: ResistanceSolverConfig = field(default_factory=ResistanceSolverConfig)
    objective: str = "direct_log1p"
    calibrations: dict[str, MLPEHead] = field(default_factory=dict)
    training_pairs: dict[str, tuple[tuple[str, str], ...]] = field(default_factory=dict)
    validation_pairs: dict[str, tuple[tuple[str, str], ...]] = field(default_factory=dict)

    def __post_init__(self):
        if self.objective not in {"direct_log1p", "mlpe"}:
            raise ValueError("CalibratedModel objective must be direct_log1p or mlpe")
        if self.objective == "mlpe" and self.target.kind != "dissimilarity":
            raise ValueError("Population MLPE requires a dissimilarity target")
        heads = dict(self.calibrations)
        if any(not isinstance(head, MLPEHead) for head in heads.values()):
            raise ValueError("Regional calibrations must contain fitted MLPEHead values")
        if any(
            name != head.region_name or head.target != self.target for name, head in heads.items()
        ):
            raise ValueError(
                "Regional calibration names and target contracts must match the model"
            )
        object.__setattr__(self, "calibrations", heads)
        for name in ("training_pairs", "validation_pairs"):
            access = {
                region: tuple(tuple(sorted(pair)) for pair in pairs)
                for region, pairs in getattr(self, name).items()
            }
            object.__setattr__(self, name, access)

    def _validate_region(self, region: RegionBatch):
        if region.feature_array.shape[-1] != self.feature_count:
            raise ValueError(f"Expected {self.feature_count} feature channels")
        if region.feature_names != self.feature_names:
            raise ValueError(
                "Query feature contract must match training feature meanings and order"
            )

    def landscape_scores(self, region: RegionBatch) -> np.ndarray:
        """Predict scores for prepared query locations without genetic observations."""
        self._validate_region(region)
        options = {}
        if isinstance(self.encoder, ConductanceModel):
            height, width = region.feature_array.shape[:2]
            if height % self.encoder.patch_size or width % self.encoder.patch_size:
                raise ValueError("Raster dimensions must be divisible by model patch_size")
            options["context"] = build_resistance_context(
                (height // self.encoder.patch_size, width // self.encoder.patch_size),
                self.solver_config,
            )
        values = np.asarray(
            self.encoder.predict_distances(region.feature_array, region.pixel_nodes, **options)
        )
        if not np.isfinite(values).all() or (values < 0).any():
            raise FloatingPointError("Encoder produced nonfinite or negative distances")
        return values

    def conductance_surface(self, region: RegionBatch) -> np.ndarray:
        """Expose a fitted conductance surface separately from genetic predictions."""
        if not isinstance(self.encoder, ConductanceModel):
            raise TypeError("This encoder does not produce a conductance surface")
        self._validate_region(region)
        surface = np.asarray(self.encoder.conductance(region.feature_array))
        if not np.all(np.isfinite(surface) & (surface > 0)):
            raise FloatingPointError("Encoder produced nonfinite or nonpositive conductance")
        return surface

    def _regional_head(self, region: RegionBatch) -> MLPEHead:
        self._validate_region(region)
        if self.objective != "mlpe":
            raise ValueError("This operation requires an MLPE model")
        head = self.calibrations.get(region.name)
        if head is None:
            raise ValueError(
                f"Region {region.name!r} has no MLPE calibration; calibrate it explicitly"
            )
        if any(kind != "population" for kind in region.sampling_unit_kinds):
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
        values = np.zeros((count, count), dtype=marginal.values.dtype)
        values[left, right] = marginal.values
        values[right, left] = marginal.values
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
        head = self._regional_head(region) if self.objective == "mlpe" else None
        scores = _scores_for_pairs(self.landscape_scores(region), region, pairs)
        values = (
            self.target.inverse(scores)
            if head is None
            else head.predict_marginal(scores, pairs).values
        )
        return PairPrediction(values, pairs, self.target, region.name)

    def recalibrate(self, region: RegionBatch, observations, *, partitions=None, config=None):
        """Return a new MLPE model while retaining this frozen encoder."""
        from .calibration import recalibrate

        return recalibrate(self, region, observations, partitions=partitions, config=config)

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
        This operation leaves the encoder and stored regional calibration unchanged.
        """
        head = self._regional_head(region)
        pairs = tuple(pairs)
        scores = self.landscape_scores(region)
        support_scores = _scores_for_pairs(scores, region, support_observations.observed_pairs)
        conditioned = head.condition_on_support(
            support_scores, support_observations, partition=support_partition
        )
        return conditioned.predict(_scores_for_pairs(scores, region, pairs), pairs)


def _scores_for_pairs(scores, region, pairs):
    lookup = {label: index for index, label in enumerate(region.sampling_unit_ids)}
    try:
        return np.array([scores[lookup[a], lookup[b]] for a, b in pairs])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Pairs must contain two sampling-unit labels with locations in the query region"
        ) from error
