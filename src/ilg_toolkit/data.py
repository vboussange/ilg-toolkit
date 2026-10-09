"""Prepared landscape inputs, separate from labelled genetic observations."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TargetSpec:
    """Meaning and returned scale of genetic observations.

    Direct log1p regression requires nonnegative dissimilarities. ``log1p`` is
    the loss's comparison scale; it does not change returned measurement units.
    """

    name: str
    units: str = "unspecified"
    kind: str = "dissimilarity"
    transform: str = "identity"

    def __post_init__(self):
        if not self.name or not self.units:
            raise ValueError("Target name and units must be declared")
        if self.transform != "identity":
            raise ValueError("Only an explicit identity target transform is supported currently")


@dataclass(frozen=True)
class PreparedRegion:
    """HWC features and labelled sampling-unit locations in row/column grid coordinates.

    Features are already prepared. This boundary performs no raster alignment,
    coordinate conversion, or implicit normalization.
    """

    name: str
    features: np.ndarray
    sampling_unit_ids: tuple[str, ...]
    grid_positions: np.ndarray

    def __post_init__(self):
        features = np.array(self.features, dtype=np.float32, copy=True)
        ids = tuple(self.sampling_unit_ids)
        positions = np.asarray(self.grid_positions)
        if not self.name:
            raise ValueError("Region name must be nonempty")
        if features.ndim != 3 or min(features.shape) < 1 or not np.isfinite(features).all():
            raise ValueError("features must be a finite nonempty HWC array")
        if (
            len(ids) < 2
            or len(set(ids)) != len(ids)
            or any(not isinstance(x, str) or not x for x in ids)
        ):
            raise ValueError("sampling_unit_ids must contain at least two unique nonempty labels")
        if positions.shape != (len(ids), 2) or not np.issubdtype(positions.dtype, np.integer):
            raise ValueError(
                "grid_positions must be an integer (sampling units, 2) row/column array"
            )
        if (positions < 0).any() or (positions >= np.asarray(features.shape[:2])).any():
            raise ValueError("grid_positions fall outside the raster")
        features.setflags(write=False)
        positions = np.array(positions, dtype=np.int32, copy=True)
        positions.setflags(write=False)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "sampling_unit_ids", ids)
        object.__setattr__(self, "grid_positions", positions)

    @property
    def pixel_nodes(self) -> np.ndarray:
        """Return labelled locations as row-major native-resolution pixel indices."""
        return self.grid_positions[:, 0] * self.features.shape[1] + self.grid_positions[:, 1]


@dataclass(frozen=True)
class PairwiseObservations:
    """Complete labelled symmetric dissimilarity matrix, excluding its diagonal."""

    sampling_unit_ids: tuple[str, ...]
    values: np.ndarray
    target: TargetSpec

    def __post_init__(self):
        ids = tuple(self.sampling_unit_ids)
        values = np.array(self.values, dtype=np.float32, copy=True)
        if (
            len(ids) < 2
            or len(set(ids)) != len(ids)
            or any(not isinstance(x, str) or not x for x in ids)
        ):
            raise ValueError("Observation identifiers must be unique nonempty sampling-unit labels")
        if values.shape != (len(ids), len(ids)) or not np.isfinite(values).all():
            raise ValueError("Observation matrix must be complete, finite, and aligned with labels")
        if not np.allclose(values, values.T) or not np.allclose(np.diag(values), 0):
            raise ValueError("Observation matrix must be symmetric with zero diagonal")
        values.setflags(write=False)
        object.__setattr__(self, "sampling_unit_ids", ids)
        object.__setattr__(self, "values", values)

    @classmethod
    def from_matrix(cls, sampling_unit_ids, values, *, target):
        """Declare matrix row/column identities and target scale explicitly."""
        return cls(tuple(sampling_unit_ids), values, target)

    def aligned_values(self, region: PreparedRegion) -> np.ndarray:
        """Align observation labels to the prepared sampling-unit order."""
        if set(self.sampling_unit_ids) != set(region.sampling_unit_ids):
            raise ValueError("Observation labels and region sampling_unit_ids must match")
        order = [self.sampling_unit_ids.index(label) for label in region.sampling_unit_ids]
        return self.values[np.ix_(order, order)]
