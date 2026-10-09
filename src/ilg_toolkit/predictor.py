"""Label-free landscape scoring and explicit target-scale predictions."""

from dataclasses import dataclass

import numpy as np

from .data import PreparedRegion, TargetSpec
from .models import EmbeddingDistanceModel


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

    encoder: EmbeddingDistanceModel
    target: TargetSpec
    feature_count: int
    feature_names: tuple[str, ...] | None = None

    def landscape_scores(self, region: PreparedRegion) -> np.ndarray:
        """Predict scores for prepared query locations without genetic observations."""
        if region.features.shape[-1] != self.feature_count:
            raise ValueError(f"Expected {self.feature_count} feature channels")
        if region.feature_names != self.feature_names:
            raise ValueError(
                "Query feature contract must match training feature meanings and order"
            )
        values = np.asarray(self.encoder.predict_distances(region.features, region.pixel_nodes))
        if not np.isfinite(values).all() or (values < 0).any():
            raise FloatingPointError("Encoder produced nonfinite or negative distances")
        return values

    def predict(self, region: PreparedRegion) -> Prediction:
        """Return direct predictions on the declared genetic measurement scale."""
        return Prediction(
            self.target.inverse(self.landscape_scores(region)),
            region.sampling_unit_ids,
            self.target,
        )
