"""Prepared landscape inputs, separate from labelled genetic observations."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TargetSpec:
    """Meaning and returned scale of genetic observations.

    ``transform`` explicitly maps original measurements into the encoder's
    regression scale. Supported invertible choices are identity, log1p, and
    sqrt. Prediction inverts this transform and returns original ``units``.
    The direct objective also uses log1p as a loss comparison; that separate
    operation never implies an unrequested measurement transformation.
    """

    name: str
    units: str = "unspecified"
    kind: str = "dissimilarity"
    transform: str = "identity"

    def __post_init__(self):
        if any(
            not isinstance(value, str) or not value
            for value in (self.name, self.units, self.kind, self.transform)
        ):
            raise ValueError("Target name, units, kind and transform must be nonempty strings")
        if self.kind not in {"dissimilarity", "relatedness", "similarity"}:
            raise ValueError("Target kind must be dissimilarity, relatedness, or similarity")
        if self.transform not in {"identity", "log1p", "sqrt"}:
            raise ValueError("Target transform must be identity, log1p, or sqrt")

    def forward(self, values) -> np.ndarray:
        """Transform finite original observations, without automatic clipping."""
        values = np.asarray(values)
        if not np.isfinite(values).all():
            raise ValueError("Observed target values must be finite")
        if self.transform != "identity" and (values < 0).any():
            raise ValueError(f"Target transform {self.transform} requires nonnegative values")
        if self.transform == "log1p":
            return np.log1p(values)
        if self.transform == "sqrt":
            return np.sqrt(values)
        return values

    def inverse(self, values) -> np.ndarray:
        """Return predictions on original measurement units."""
        values = np.asarray(values)
        if self.transform == "sqrt" and (values < 0).any():
            raise ValueError("sqrt inverse requires nonnegative transformed predictions")
        with np.errstate(over="ignore", invalid="ignore"):
            if self.transform == "log1p":
                values = np.expm1(values)
            elif self.transform == "sqrt":
                values = np.square(values)
        if not np.isfinite(values).all():
            raise FloatingPointError("Target inverse transformation produced nonfinite predictions")
        return values


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
    feature_names: tuple[str, ...] | None = None
    sampling_unit_kinds: tuple[str, ...] | None = None

    def __post_init__(self):
        features = np.array(self.features, dtype=np.float32, copy=True)
        ids = tuple(self.sampling_unit_ids)
        positions = np.asarray(self.grid_positions)
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Region name must be a nonempty string")
        if features.ndim != 3 or min(features.shape) < 1 or not np.isfinite(features).all():
            raise ValueError("features must be a finite nonempty HWC array")
        if (
            len(ids) < 2
            or any(not isinstance(x, str) or not x for x in ids)
            or len(set(ids)) != len(ids)
        ):
            raise ValueError("sampling_unit_ids must contain at least two unique nonempty labels")
        if positions.shape != (len(ids), 2) or not np.issubdtype(positions.dtype, np.integer):
            raise ValueError(
                "grid_positions must be an integer (sampling units, 2) row/column array"
            )
        if (positions < 0).any() or (positions >= np.asarray(features.shape[:2])).any():
            raise ValueError("grid_positions fall outside the raster")
        names = None if self.feature_names is None else tuple(self.feature_names)
        if names is not None and (
            len(names) != features.shape[-1]
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("feature_names must uniquely declare each channel's meaning and order")
        kinds = (
            ("population",) * len(ids)
            if self.sampling_unit_kinds is None
            else tuple(self.sampling_unit_kinds)
        )
        if len(kinds) != len(ids) or any(
            kind not in {"population", "individual"} for kind in kinds
        ):
            raise ValueError(
                "sampling_unit_kinds must declare population or individual for each label"
            )
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "sampling_unit_kinds", kinds)
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
class ObservationPartition:
    """Explicit observed unordered pairs assigned a role within one named region.

    Partitions select observations; they do not create measurements. Endpoint
    holdout fold generation is a separate convenience rather than an assumption
    built into this pair-level input contract.
    """

    region_name: str
    pairs: tuple[tuple[str, str], ...]
    role: str = "training"

    def __post_init__(self):
        if not isinstance(self.region_name, str) or not self.region_name:
            raise ValueError("Partition region_name must be declared")
        if self.role not in {"training", "validation", "calibration", "support", "query"}:
            raise ValueError(
                "Partition role must be training, validation, calibration, support, or query"
            )
        pairs = tuple(tuple(pair) for pair in self.pairs)
        if not pairs or any(
            len(pair) != 2
            or any(not isinstance(x, str) or not x for x in pair)
            or pair[0] == pair[1]
            for pair in pairs
        ):
            raise ValueError("Partition pairs must be nonempty distinct labelled endpoints")
        pairs = tuple(tuple(sorted(pair)) for pair in pairs)
        if len(set(pairs)) != len(pairs):
            raise ValueError("Partition pairs must be unique, including reversed duplicates")
        object.__setattr__(self, "pairs", pairs)


@dataclass(frozen=True)
class PairwiseObservations:
    """Labelled unordered observations; NaN entries represent absent pairs.

    Use ``from_pairs`` for incomplete observations or ``from_matrix`` for a
    labelled symmetric matrix. No missing pair is replaced by a zero target.
    """

    sampling_unit_ids: tuple[str, ...]
    values: np.ndarray
    target: TargetSpec

    def __post_init__(self):
        ids = tuple(self.sampling_unit_ids)
        values = np.array(self.values, dtype=np.float64, copy=True)
        if (
            len(ids) < 2
            or any(not isinstance(x, str) or not x for x in ids)
            or len(set(ids)) != len(ids)
        ):
            raise ValueError("Observation identifiers must be unique nonempty sampling-unit labels")
        if values.shape != (len(ids), len(ids)) or np.isinf(values).any():
            raise ValueError(
                "Observation matrix must be aligned with labels and contain no infinity"
            )
        if not np.allclose(values, values.T, equal_nan=True) or not np.allclose(np.diag(values), 0):
            raise ValueError("Observation matrix must be symmetric with zero diagonal")
        if not np.isfinite(values[np.triu_indices(len(ids), 1)]).any():
            raise ValueError("At least one observed pair is required")
        values.setflags(write=False)
        object.__setattr__(self, "sampling_unit_ids", ids)
        object.__setattr__(self, "values", values)

    @classmethod
    def from_matrix(cls, sampling_unit_ids, values, *, target):
        """Declare matrix identities and scale; symmetric off-diagonal NaNs are absent."""
        return cls(tuple(sampling_unit_ids), values, target)

    @classmethod
    def from_pairs(cls, pairs, values, *, target, sampling_unit_ids=None):
        """Build from explicit unordered endpoint labels and finite measurements.

        Repeated pairs, including reversed duplicates, are rejected. Optional
        ``sampling_unit_ids`` declares a larger endpoint universe containing
        sampling units with no observed pairs.
        """
        pairs = tuple(tuple(pair) for pair in pairs)
        values = np.asarray(values, dtype=np.float64)
        if not pairs or values.shape != (len(pairs),) or not np.isfinite(values).all():
            raise ValueError("Explicit pairs require one finite observed value per nonempty pair")
        if any(
            len(pair) != 2 or any(not isinstance(x, str) or not x for x in pair) for pair in pairs
        ):
            raise ValueError("Each pair must contain two nonempty sampling-unit labels")
        ids = (
            tuple(sampling_unit_ids)
            if sampling_unit_ids is not None
            else tuple(sorted({endpoint for pair in pairs for endpoint in pair}))
        )
        if any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != len(ids):
            raise ValueError("Observation identifiers must be unique nonempty labels")
        lookup = {label: index for index, label in enumerate(ids)}
        matrix = np.full((len(ids), len(ids)), np.nan, dtype=np.float64)
        np.fill_diagonal(matrix, 0)
        seen = set()
        for pair, value in zip(pairs, values, strict=True):
            if pair[0] == pair[1]:
                raise ValueError("Observed pairs must have distinct endpoints")
            if any(endpoint not in lookup for endpoint in pair):
                raise ValueError("Observed pair endpoint is absent from sampling_unit_ids")
            canonical = tuple(sorted(pair))
            if canonical in seen:
                raise ValueError("Observed pairs must be unique, including reversed pairs")
            seen.add(canonical)
            left, right = (lookup[endpoint] for endpoint in pair)
            matrix[left, right] = matrix[right, left] = value
        return cls(ids, matrix, target)

    @property
    def observed_pairs(self) -> tuple[tuple[str, str], ...]:
        """Return only measured unordered pairs, in deterministic matrix order."""
        left, right = np.triu_indices(len(self.sampling_unit_ids), 1)
        present = np.isfinite(self.values[left, right])
        return tuple(
            (self.sampling_unit_ids[i], self.sampling_unit_ids[j])
            for i, j in zip(left[present], right[present], strict=True)
        )

    @property
    def observed_values(self) -> np.ndarray:
        """Return original-scale measurements aligned with ``observed_pairs``."""
        upper = self.values[np.triu_indices(len(self.sampling_unit_ids), 1)]
        return upper[np.isfinite(upper)]

    def aligned_values(self, region: PreparedRegion) -> np.ndarray:
        """Align known observation labels to a region, retaining NaNs for absent pairs."""
        if not set(self.sampling_unit_ids).issubset(region.sampling_unit_ids):
            raise ValueError(
                "Observation labels contain endpoints outside region sampling_unit_ids"
            )
        matrix = np.full(
            (len(region.sampling_unit_ids), len(region.sampling_unit_ids)), np.nan, dtype=np.float64
        )
        np.fill_diagonal(matrix, 0)
        lookup = {label: index for index, label in enumerate(region.sampling_unit_ids)}
        order = [lookup[label] for label in self.sampling_unit_ids]
        matrix[np.ix_(order, order)] = self.values
        return matrix

    def aligned_pairs(self, region: PreparedRegion, partition: ObservationPartition | None = None):
        """Return observed index pairs and measurements in the prepared unit order."""
        values = self.aligned_values(region)
        left, right = np.triu_indices(len(region.sampling_unit_ids), 1)
        present = np.isfinite(values[left, right])
        if partition is not None:
            if partition.region_name != region.name:
                raise ValueError("Partition region_name does not match the prepared region")
            endpoints = {endpoint for pair in partition.pairs for endpoint in pair}
            if not endpoints.issubset(region.sampling_unit_ids):
                raise ValueError("Partition contains endpoints outside the prepared region")
            observed = {tuple(sorted(pair)) for pair in self.observed_pairs}
            if not set(partition.pairs).issubset(observed):
                raise ValueError("Partition selects an unobserved pair")
            selected = set(partition.pairs)
            present &= np.array(
                [
                    tuple(sorted((region.sampling_unit_ids[i], region.sampling_unit_ids[j])))
                    in selected
                    for i, j in zip(left, right, strict=True)
                ]
            )
        pairs = left[present], right[present]
        if not len(pairs[0]):
            raise ValueError("Selected observations must contain at least one observed pair")
        return pairs, values[pairs]
