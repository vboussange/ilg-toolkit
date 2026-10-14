"""Prepared landscape inputs, separate from labelled genetic observations."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import coordax as cx
import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike


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

    def forward(self, values) -> jax.Array:
        """Transform finite original observations, without automatic clipping."""
        values = jnp.asarray(values)
        if not bool(jnp.isfinite(values).all()):
            raise ValueError("Observed target values must be finite")
        if self.transform != "identity" and (values < 0).any():
            raise ValueError(f"Target transform {self.transform} requires nonnegative values")
        if self.transform == "log1p":
            return jnp.log1p(values)
        if self.transform == "sqrt":
            return jnp.sqrt(values)
        return values

    def inverse(self, values) -> jax.Array:
        """Return predictions on original measurement units."""
        values = jnp.asarray(values)
        if self.transform == "sqrt" and (values < 0).any():
            raise ValueError("sqrt inverse requires nonnegative transformed predictions")
        if self.transform == "log1p":
            values = jnp.expm1(values)
        elif self.transform == "sqrt":
            values = jnp.square(values)
        if not bool(jnp.isfinite(values).all()):
            raise FloatingPointError("Target inverse transformation produced nonfinite predictions")
        return values


@dataclass(frozen=True)
class SamplingUnits:
    """Identified populations/individuals at prepared raster row/column cells."""

    ids: tuple[str, ...]
    grid_positions: ArrayLike
    kinds: tuple[str, ...] | None = None

    def __post_init__(self):
        ids = tuple(self.ids)
        positions = np.asarray(self.grid_positions)
        if (
            len(ids) < 2
            or len(set(ids)) != len(ids)
            or any(not isinstance(label, str) or not label for label in ids)
        ):
            raise ValueError("Sampling unit ids require at least two unique nonempty labels")
        if positions.shape != (len(ids), 2) or not np.issubdtype(positions.dtype, np.integer):
            raise ValueError("grid_positions must be an integer (sampling units, 2) array")
        if bool(((positions < 0) | (positions > np.iinfo(np.int32).max)).any()):
            raise ValueError("grid_positions must be nonnegative and fit the int32 index range")
        kinds = ("population",) * len(ids) if self.kinds is None else tuple(self.kinds)
        if len(kinds) != len(ids) or any(
            kind not in {"population", "individual"} for kind in kinds
        ):
            raise ValueError("kinds must declare population or individual for each sampling unit")
        object.__setattr__(self, "ids", ids)
        object.__setattr__(self, "grid_positions", jnp.asarray(positions, dtype=jnp.int32))
        object.__setattr__(self, "kinds", kinds)


@dataclass(frozen=True)
class RegionBatch:
    """One prepared Coordax raster with species-keyed sampling-unit groups."""

    name: str
    features: cx.Field
    sampling_units: Mapping[str, SamplingUnits]
    feature_names: tuple[str, ...] | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Region name must be a nonempty string")
        names = None if self.feature_names is None else tuple(self.feature_names)
        if not isinstance(self.features, cx.Field):
            raise TypeError("features must be a coordax.Field with named row, column, feature axes")
        if self.features.ndim != 3 or set(self.features.dims) != {"row", "column", "feature"}:
            raise ValueError("Labelled features require named axes row, column, feature")
        field = self.features.order_as("row", "column", "feature")
        axes = dict(field.axes)
        coordinate = field.axes.get("feature")
        if isinstance(coordinate, cx.LabeledAxis):
            labels = tuple(coordinate.ticks.tolist())
            if any(not isinstance(label, str) or not label for label in labels):
                raise ValueError("feature coordinates require nonempty string labels")
            if names is not None and names != labels:
                raise ValueError("Feature coordinates must match feature_names meanings and order")
            names = labels
        features = jnp.asarray(field.data)
        if min(features.shape) < 1 or not bool(jnp.isfinite(features).all()):
            raise ValueError("features must contain finite nonempty row, column, feature data")
        if names is not None and (
            len(names) != features.shape[-1]
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("feature_names must uniquely declare each channel's meaning and order")
        if not isinstance(self.sampling_units, Mapping) or not self.sampling_units:
            raise ValueError("sampling_units must be a nonempty species-keyed mapping")
        groups = dict(self.sampling_units)
        for species, units in groups.items():
            if not isinstance(species, str) or not species or not isinstance(units, SamplingUnits):
                raise ValueError("sampling_units must map nonempty species names to SamplingUnits")
            if bool((jnp.asarray(units.grid_positions) >= jnp.asarray(features.shape[:2])).any()):
                raise ValueError(f"grid_positions for {species!r} fall outside the raster")
        object.__setattr__(self, "sampling_units", groups)
        object.__setattr__(self, "feature_names", names)
        if names is not None:
            axes["feature"] = cx.LabeledAxis("feature", np.asarray(names))
        object.__setattr__(
            self, "features", cx.Field(features, dims=("row", "column", "feature"), axes=axes)
        )

    @property
    def feature_array(self) -> jax.Array:
        """Canonical row, column, feature numerical input."""
        return cast(jax.Array, self.features.data)

    def _pixel_nodes(self, species: str) -> jax.Array:
        positions = jnp.asarray(self.sampling_units[species].grid_positions)
        return positions[:, 0] * self.features.shape[1] + positions[:, 1]


@dataclass(frozen=True)
class ObservationPartition:
    """Explicit observed unordered pairs assigned a role within one named region.

    Partitions select observations; they do not create measurements. Endpoint
    holdout fold generation is a separate convenience rather than an assumption
    built into this pair-level input contract.
    """

    region_name: str
    species_name: str
    pairs: tuple[tuple[str, str], ...]
    role: str = "training"

    def __post_init__(self):
        if not isinstance(self.region_name, str) or not self.region_name:
            raise ValueError("Partition region_name must be declared")
        if not isinstance(self.species_name, str) or not self.species_name:
            raise ValueError("Partition species_name must be declared")
        if self.role not in {"training", "validation", "calibration", "support", "query"}:
            raise ValueError(
                "Partition role must be training, validation, calibration, support, or query"
            )
        if any(isinstance(pair, str) for pair in self.pairs):
            raise ValueError("Each pair must contain two explicit sampling-unit labels")
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
    labelled symmetric matrix. Floating input precision is retained; ordinary
    array-like conversion follows the configured JAX precision. Enable float64
    before creating or converting precise measurements: later promotion cannot
    recover rounded values. No missing pair is replaced by a zero target.
    """

    sampling_unit_ids: tuple[str, ...]
    values: jax.Array
    target: TargetSpec

    def __post_init__(self):
        ids = tuple(self.sampling_unit_ids)
        values = jnp.asarray(self.values)
        if (
            len(ids) < 2
            or any(not isinstance(x, str) or not x for x in ids)
            or len(set(ids)) != len(ids)
        ):
            raise ValueError("Observation identifiers must be unique nonempty sampling-unit labels")
        if values.shape != (len(ids), len(ids)) or jnp.isinf(values).any():
            raise ValueError(
                "Observation matrix must be aligned with labels and contain no infinity"
            )
        if not jnp.allclose(values, values.T, equal_nan=True) or not jnp.allclose(
            jnp.diag(values), 0
        ):
            raise ValueError("Observation matrix must be symmetric with zero diagonal")
        if not jnp.isfinite(values[jnp.triu_indices(len(ids), 1)]).any():
            raise ValueError("At least one observed pair is required")
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
        values = jnp.asarray(values)
        if not pairs or values.shape != (len(pairs),) or not jnp.isfinite(values).all():
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
        dtype = (
            values.dtype
            if jnp.issubdtype(values.dtype, jnp.inexact)
            else jnp.result_type(values, 0.0)
        )
        matrix = jnp.full((len(ids), len(ids)), jnp.nan, dtype=dtype)
        matrix = matrix.at[jnp.diag_indices(len(ids))].set(0)
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
            matrix = matrix.at[left, right].set(value).at[right, left].set(value)
        return cls(ids, matrix, target)

    @property
    def observed_pairs(self) -> tuple[tuple[str, str], ...]:
        """Return only measured unordered pairs, in deterministic matrix order."""
        left, right = jnp.triu_indices(len(self.sampling_unit_ids), 1)
        present = jnp.isfinite(self.values[left, right])
        return tuple(
            (self.sampling_unit_ids[i], self.sampling_unit_ids[j])
            for i, j in zip(left[present], right[present], strict=True)
        )

    @property
    def observed_values(self) -> jax.Array:
        """Return original-scale measurements aligned with ``observed_pairs``."""
        upper = self.values[jnp.triu_indices(len(self.sampling_unit_ids), 1)]
        return upper[jnp.isfinite(upper)]

    def aligned_values(self, region: RegionBatch, species: str) -> jax.Array:
        """Align known observation labels to a region, retaining NaNs for absent pairs."""
        if not set(self.sampling_unit_ids).issubset(region.sampling_units[species].ids):
            raise ValueError(
                "Observation labels contain endpoints outside region sampling_unit_ids"
            )
        dtype = (
            self.values.dtype
            if jnp.issubdtype(self.values.dtype, jnp.inexact)
            else jnp.result_type(self.values, 0.0)
        )
        matrix = jnp.full(
            (len(region.sampling_units[species].ids), len(region.sampling_units[species].ids)),
            jnp.nan,
            dtype=dtype,
        )
        matrix = matrix.at[jnp.diag_indices(len(region.sampling_units[species].ids))].set(0)
        lookup = {label: index for index, label in enumerate(region.sampling_units[species].ids)}
        order = jnp.asarray([lookup[label] for label in self.sampling_unit_ids])
        return matrix.at[jnp.ix_(order, order)].set(self.values)

    def aligned_pairs(
        self, region: RegionBatch, species: str, partition: ObservationPartition | None = None
    ):
        """Return observed index pairs and measurements in the prepared unit order."""
        values = self.aligned_values(region, species)
        left, right = jnp.triu_indices(len(region.sampling_units[species].ids), 1)
        present = jnp.isfinite(values[left, right])
        if partition is not None:
            if partition.species_name != species:
                raise ValueError("Partition species_name does not match the observation group")
            if partition.region_name != region.name:
                raise ValueError("Partition region_name does not match the prepared region")
            endpoints = {endpoint for pair in partition.pairs for endpoint in pair}
            if not endpoints.issubset(region.sampling_units[species].ids):
                raise ValueError("Partition contains endpoints outside the prepared region")
            observed = {tuple(sorted(pair)) for pair in self.observed_pairs}
            if not set(partition.pairs).issubset(observed):
                raise ValueError("Partition selects an unobserved pair")
            selected = set(partition.pairs)
            present &= jnp.asarray(
                [
                    tuple(
                        sorted(
                            (
                                region.sampling_units[species].ids[i],
                                region.sampling_units[species].ids[j],
                            )
                        )
                    )
                    in selected
                    for i, j in zip(left, right, strict=True)
                ]
            )
        pairs = left[present], right[present]
        if not len(pairs[0]):
            raise ValueError("Selected observations must contain at least one observed pair")
        return pairs, values[pairs]
