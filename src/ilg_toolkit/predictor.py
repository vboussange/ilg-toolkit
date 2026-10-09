"""Label-free landscape scoring and explicit target-scale predictions."""

from dataclasses import dataclass, field

import numpy as np

from .config import SolverConfig
from .data import PreparedRegion, TargetSpec
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
    """A direct-regression encoder; no supplementary genetic calibration is required."""

    encoder: ConductanceModel | EmbeddingDistanceModel
    target: TargetSpec
    feature_count: int
    feature_names: tuple[str, ...] | None = None
    solver_config: SolverConfig = field(default_factory=SolverConfig)

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

    def predict(self, region: PreparedRegion) -> Prediction:
        """Return direct predictions on the declared genetic measurement scale."""
        return Prediction(
            self.target.inverse(self.landscape_scores(region)),
            region.sampling_unit_ids,
            self.target,
        )
