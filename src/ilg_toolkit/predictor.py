"""Label-free landscape scoring and explicit target-scale predictions."""

from dataclasses import dataclass, field

import numpy as np

from .config import SolverConfig
from .data import PreparedRegion, TargetSpec
from .mlpe import MLPEHead
from .models import ConductanceModel, EmbeddingDistanceModel
from .solver import build_solver_context


@dataclass(frozen=True)
class Prediction:
    """Pairwise predictions with their sampling-unit order and declared scale."""

    values: np.ndarray
    sampling_unit_ids: tuple[str, ...]
    target: TargetSpec
    scale: str = "original"


@dataclass(frozen=True)
class Predictor:
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
    solver_config: SolverConfig = field(default_factory=SolverConfig)
    objective: str = "direct_log1p"
    calibrations: dict[str, MLPEHead] = field(default_factory=dict)
    training_pairs: dict[str, tuple[tuple[str, str], ...]] = field(default_factory=dict)
    validation_pairs: dict[str, tuple[tuple[str, str], ...]] = field(default_factory=dict)

    def __post_init__(self):
        if self.objective not in {"direct_log1p", "mlpe"}:
            raise ValueError("Predictor objective must be direct_log1p or mlpe")
        if self.objective == "mlpe" and self.target.kind != "dissimilarity":
            raise ValueError("Population MLPE requires a dissimilarity target")
        heads = dict(self.calibrations)
        if any(not isinstance(head, MLPEHead) for head in heads.values()):
            raise ValueError("Regional calibrations must contain fitted MLPEHead values")
        if any(
            name != head.region_name or head.target != self.target for name, head in heads.items()
        ):
            raise ValueError(
                "Regional calibration names and target contracts must match the predictor"
            )
        object.__setattr__(self, "calibrations", heads)
        for name in ("training_pairs", "validation_pairs"):
            access = {
                region: tuple(tuple(sorted(pair)) for pair in pairs)
                for region, pairs in getattr(self, name).items()
            }
            object.__setattr__(self, name, access)

    def _validate_region(self, region: PreparedRegion):
        if region.features.shape[-1] != self.feature_count:
            raise ValueError(f"Expected {self.feature_count} feature channels")
        if region.feature_names != self.feature_names:
            raise ValueError(
                "Query feature contract must match training feature meanings and order"
            )

    def landscape_scores(self, region: PreparedRegion) -> np.ndarray:
        """Predict scores for prepared query locations without genetic observations."""
        self._validate_region(region)
        options = {}
        if isinstance(self.encoder, ConductanceModel):
            height, width = region.features.shape[:2]
            if height % self.encoder.patch_size or width % self.encoder.patch_size:
                raise ValueError("Raster dimensions must be divisible by model patch_size")
            options["context"] = build_solver_context(
                (height // self.encoder.patch_size, width // self.encoder.patch_size),
                self.solver_config,
            )
        values = np.asarray(
            self.encoder.predict_distances(region.features, region.pixel_nodes, **options)
        )
        if not np.isfinite(values).all() or (values < 0).any():
            raise FloatingPointError("Encoder produced nonfinite or negative distances")
        return values

    def conductance_surface(self, region: PreparedRegion) -> np.ndarray:
        """Expose a fitted conductance surface separately from genetic predictions."""
        if not isinstance(self.encoder, ConductanceModel):
            raise TypeError("This encoder does not produce a conductance surface")
        self._validate_region(region)
        surface = np.asarray(self.encoder.conductance(region.features))
        if not np.all(np.isfinite(surface) & (surface > 0)):
            raise FloatingPointError("Encoder produced nonfinite or nonpositive conductance")
        return surface

    def _regional_head(self, region: PreparedRegion) -> MLPEHead:
        self._validate_region(region)
        if self.objective != "mlpe":
            raise ValueError("This operation requires an MLPE predictor")
        head = self.calibrations.get(region.name)
        if head is None:
            raise ValueError(
                f"Region {region.name!r} has no MLPE calibration; calibrate it explicitly"
            )
        if any(kind != "population" for kind in region.sampling_unit_kinds):
            raise ValueError("Population MLPE prediction requires population sampling units")
        return head

    def predict(self, region: PreparedRegion) -> Prediction:
        """Return original-scale predictions; MLPE uses the region's marginal head.

        No query targets are accepted. An unseen MLPE region requires explicit
        calibration even though its landscape scores can already be computed.
        """
        self._validate_region(region)
        if self.objective == "direct_log1p":
            values = self.target.inverse(self.landscape_scores(region))
        else:
            head = self._regional_head(region)
            scores = self.landscape_scores(region)
            left, right = np.triu_indices(len(region.sampling_unit_ids), 1)
            pairs = tuple(
                (region.sampling_unit_ids[i], region.sampling_unit_ids[j])
                for i, j in zip(left, right, strict=True)
            )
            marginal = head.predict_marginal(scores[left, right], pairs)
            values = np.zeros_like(scores, dtype=np.float64)
            values[left, right] = marginal.values
            values[right, left] = marginal.values
        return Prediction(values, region.sampling_unit_ids, self.target)

    def recalibrate(self, region: PreparedRegion, observations, *, partitions=None, config=None):
        """Return a new MLPE predictor while retaining this frozen encoder."""
        from .calibration import recalibrate

        return recalibrate(self, region, observations, partitions=partitions, config=config)
