"""Region batches and labeled pairwise genetic observations."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast, NamedTuple

import coordax as cx
import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike


class _TargetTransform(NamedTuple):
    """Maps between the original and model scales and the values each accepts."""

    forward: Callable[[jax.Array], jax.Array]
    inverse: Callable[[jax.Array], jax.Array]
    # Accepted finite original-scale measurements and their description; None accepts all.
    domain: tuple[Callable[[jax.Array], jax.Array], str] | None
    # Whether the inverse rejects negative model-scale predictions.
    nonnegative_predictions: bool


def _nonnegative(values: jax.Array) -> jax.Array:
    return values >= 0


def _unit_interval(values: jax.Array) -> jax.Array:
    return (values >= 0) & (values < 1)


_TARGET_TRANSFORMS = {
    "identity": _TargetTransform(lambda values: values, lambda values: values, None, False),
    "log1p": _TargetTransform(jnp.log1p, jnp.expm1, (_nonnegative, "nonnegative values"), False),
    "sqrt": _TargetTransform(jnp.sqrt, jnp.square, (_nonnegative, "nonnegative values"), True),
    "odds": _TargetTransform(
        lambda values: values / (1 - values),
        lambda values: values / (1 + values),
        (_unit_interval, "values in [0, 1)"),
        True,
    ),
}


@dataclass(frozen=True)
class TargetSpec:
    """Meaning, original units, and model-scale transform of pairwise genetic observations.

    ``transform`` maps measurements from the original scale to the model scale,
    and genetic predictions apply its inverse:

    - ``"identity"``: ``x``.
    - ``"log1p"``: ``log(1 + x)``, for nonnegative values.
    - ``"sqrt"``: ``sqrt(x)``, for nonnegative values.
    - ``"odds"``: ``x / (1 - x)``, for values in [0, 1), with inverse ``y / (1 + y)``.
      It linearizes dissimilarities bounded by one; Rousset's linearized Fst,
      ``Fst / (1 - Fst)``, is one example. Negative estimates are rejected rather
      than clipped, so clean them before constructing observations.

    The direct objective compares ecological distances and model-scale targets
    using log1p.
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
        if self.transform not in _TARGET_TRANSFORMS:
            *names, last = _TARGET_TRANSFORMS
            raise ValueError(f"Target transform must be {', '.join(names)}, or {last}")

    def forward(self, values) -> jax.Array:
        """Model-scale observations transformed from finite original-scale measurements."""
        values = jnp.asarray(values)
        if not bool(jnp.isfinite(values).all()):
            raise ValueError("Observed target values must be finite")
        transform = _TARGET_TRANSFORMS[self.transform]
        if transform.domain is not None:
            accepts, description = transform.domain
            if not bool(accepts(values).all()):
                raise ValueError(f"Target transform {self.transform} requires {description}")
        return transform.forward(values)

    def inverse(self, values) -> jax.Array:
        """Genetic predictions in the original measurement units."""
        values = jnp.asarray(values)
        transform = _TARGET_TRANSFORMS[self.transform]
        if transform.nonnegative_predictions and (values < 0).any():
            raise ValueError(
                f"{self.transform} inverse requires nonnegative transformed predictions"
            )
        values = transform.inverse(values)
        if not bool(jnp.isfinite(values).all()):
            raise FloatingPointError("Target inverse transformation produced nonfinite predictions")
        return values


@dataclass(frozen=True)
class RegionBatch:
    """Prepared landscape features and sampling-unit locations for one region.

    Optional species labels distinguish sampling units whose identifiers repeat
    across species. Features use named row, column, and feature dimensions.
    """

    name: str
    features: cx.Field
    sampling_unit_ids: tuple[str, ...]
    grid_positions: ArrayLike
    feature_names: tuple[str, ...] | None = None
    sampling_unit_kinds: tuple[str, ...] | None = None
    species: tuple[str, ...] | None = None

    def __post_init__(self):
        names = None if self.feature_names is None else tuple(self.feature_names)
        if not isinstance(self.features, cx.Field):
            raise TypeError("features must be a coordax.Field with named row, column, feature axes")
        if self.features.ndim != 3 or set(self.features.dims) != {"row", "column", "feature"}:
            raise ValueError("Labeled features require named axes row, column, feature")
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
        ids = tuple(self.sampling_unit_ids)
        species = None if self.species is None else tuple(self.species)
        if species is not None and (
            len(species) != len(ids)
            or any(not isinstance(name, str) or not name for name in species)
        ):
            raise ValueError("species must label each sampling unit with a nonempty name")
        identities = ids if species is None else tuple(zip(species, ids, strict=True))
        positions = jnp.asarray(self.grid_positions)
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Region name must be a nonempty string")
        if min(features.shape) < 1 or not bool(jnp.isfinite(features).all()):
            raise ValueError("features must contain finite nonempty row, column, feature data")
        if (
            len(ids) < 2
            or any(not isinstance(x, str) or not x for x in ids)
            or len(set(identities)) != len(ids)
        ):
            raise ValueError("Sampling-unit ids must be nonempty and unique within each species")
        if positions.shape != (len(ids), 2) or not jnp.issubdtype(positions.dtype, jnp.integer):
            raise ValueError(
                "grid_positions must be an integer (sampling units, 2) row/column array"
            )
        if (positions < 0).any() or (positions >= jnp.asarray(features.shape[:2])).any():
            raise ValueError("grid_positions fall outside the raster")
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
        if names is not None:
            axes["feature"] = cx.LabeledAxis("feature", np.asarray(names))
        object.__setattr__(
            self, "features", cx.Field(features, dims=("row", "column", "feature"), axes=axes)
        )
        object.__setattr__(self, "sampling_unit_ids", ids)
        object.__setattr__(self, "species", species)
        object.__setattr__(self, "grid_positions", positions.astype(jnp.int32))

    @property
    def feature_array(self) -> jax.Array:
        """Landscape features as a JAX array with row, column, and feature dimensions."""
        return cast(jax.Array, self.features.data)

    @property
    def pixel_nodes(self) -> jax.Array:
        """Sampling-unit locations as row-major raster pixel indices."""
        positions = cast(jax.Array, self.grid_positions)
        return positions[:, 0] * self.features.shape[1] + positions[:, 1]


def _validate_pairs(pairs) -> tuple[tuple[str, str], ...]:
    """Validate unordered identities while preserving requested orientation and order."""
    try:
        raw = tuple(pairs)
        if any(isinstance(pair, (str, bytes)) for pair in raw):
            raise ValueError("Each pair must contain two sampling-unit labels")
        labels = tuple(tuple(pair) for pair in raw)
    except (TypeError, ValueError) as error:
        raise ValueError("Pairs must contain two sampling-unit labels") from error
    if not labels or any(
        len(pair) != 2
        or any(not isinstance(label, str) or not label for label in pair)
        or pair[0] == pair[1]
        for pair in labels
    ):
        raise ValueError("Pairs require two distinct nonempty sampling-unit labels")
    if len({tuple(sorted(pair)) for pair in labels}) != len(labels):
        raise ValueError("Pairs must be unique, including reversed duplicates")
    return labels


@dataclass(frozen=True)
class PairwiseObservations:
    """Labeled pairwise genetic observations with not-a-number (NaN) entries for absent pairs.

    Use ``from_pairs`` for incomplete observations or ``from_matrix`` for a
    labeled symmetric matrix.
    Floating input precision is retained; array-like conversion uses the current
    JAX precision, so enable float64 before creating precise measurements.
    """

    sampling_unit_ids: tuple[str, ...]
    values: jax.Array
    target: TargetSpec
    species: str | None = None

    def __post_init__(self):
        if self.species is not None and (not isinstance(self.species, str) or not self.species):
            raise ValueError("Observation species must be a nonempty name or None")
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
    def from_matrix(cls, sampling_unit_ids, values, *, target, species=None):
        """Pairwise genetic observations from a labeled matrix for one species.

        Symmetric off-diagonal NaNs identify absent observations.
        """
        return cls(tuple(sampling_unit_ids), values, target, species)

    @classmethod
    def from_pairs(cls, pairs, values, *, target, sampling_unit_ids=None, species=None):
        """Pairwise genetic observations from labeled pairs and finite measurements.

        Repeated pairs, including reversed duplicates, are rejected. Optional
        ``sampling_unit_ids`` declares a larger set of sampling units containing
        sampling units with no observed pairs.
        """
        pairs = _validate_pairs(pairs)
        values = jnp.asarray(values)
        if not pairs or values.shape != (len(pairs),) or not jnp.isfinite(values).all():
            raise ValueError("Observed pairs must be nonempty and have one finite value each")
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
                raise ValueError("Observed pairs must have distinct sampling units")
            if any(endpoint not in lookup for endpoint in pair):
                raise ValueError("An observed sampling unit is absent from sampling_unit_ids")
            canonical = tuple(sorted(pair))
            if canonical in seen:
                raise ValueError("Observed pairs must be unique, including reversed pairs")
            seen.add(canonical)
            left, right = (lookup[endpoint] for endpoint in pair)
            matrix = matrix.at[left, right].set(value).at[right, left].set(value)
        return cls(ids, matrix, target, species)

    @property
    def observed_pairs(self) -> tuple[tuple[str, str], ...]:
        """Measured unordered sampling-unit pairs in deterministic matrix order."""
        left, right = jnp.triu_indices(len(self.sampling_unit_ids), 1)
        present = jnp.isfinite(self.values[left, right])
        return tuple(
            (self.sampling_unit_ids[i], self.sampling_unit_ids[j])
            for i, j in zip(left[present], right[present], strict=True)
        )

    @property
    def observed_values(self) -> jax.Array:
        """Original-scale measurements aligned with ``observed_pairs``."""
        upper = self.values[jnp.triu_indices(len(self.sampling_unit_ids), 1)]
        return upper[jnp.isfinite(upper)]

    def aligned_values(self, region: RegionBatch) -> jax.Array:
        """Observations aligned with a region, retaining NaNs for absent pairs."""
        lookup = {
            label: index
            for index, label in enumerate(region.sampling_unit_ids)
            if region.species is None or region.species[index] == self.species
        }
        if not set(self.sampling_unit_ids).issubset(lookup):
            raise ValueError("Observation labels must belong to their species in the region")
        dtype = (
            self.values.dtype
            if jnp.issubdtype(self.values.dtype, jnp.inexact)
            else jnp.result_type(self.values, 0.0)
        )
        matrix = jnp.full(
            (len(region.sampling_unit_ids), len(region.sampling_unit_ids)), jnp.nan, dtype=dtype
        )
        matrix = matrix.at[jnp.diag_indices(len(region.sampling_unit_ids))].set(0)
        order = jnp.asarray([lookup[label] for label in self.sampling_unit_ids])
        return matrix.at[jnp.ix_(order, order)].set(self.values)

    def select_pairs(self, pairs) -> "PairwiseObservations":
        """Select measured pairs, preserving target, species, precision, and label order.

        Pair identities are unordered. The returned ``observed_pairs`` and
        ``observed_values`` follow the original sampling-unit matrix order,
        independently of selection order. Empty selections, duplicate identities,
        and unobserved pairs are rejected.

        Examples:
            >>> observations = PairwiseObservations.from_pairs(
            ...     [("a", "b"), ("b", "c")], [0.2, 0.4], target=TargetSpec("Fst")
            ... )
            >>> observations.select_pairs([("c", "b")]).observed_pairs
            (("b", "c"),)
        """
        selected = {tuple(sorted(pair)) for pair in _validate_pairs(pairs)}
        observed = self.observed_pairs
        if not selected.issubset({tuple(sorted(pair)) for pair in observed}):
            raise ValueError("Selection contains an unobserved pair")
        mask = jnp.asarray([tuple(sorted(pair)) in selected for pair in observed])
        return type(self).from_pairs(
            [pair for pair, keep in zip(observed, mask, strict=True) if keep],
            self.observed_values[mask],
            target=self.target,
            species=self.species,
            sampling_unit_ids=self.sampling_unit_ids,
        )

    def aligned_pairs(self, region: RegionBatch):
        """Observed index pairs and measurements in the region batch's sampling-unit order."""
        values = self.aligned_values(region)
        left, right = jnp.triu_indices(len(region.sampling_unit_ids), 1)
        present = jnp.isfinite(values[left, right])
        pairs = left[present], right[present]
        if not len(pairs[0]):
            raise ValueError("Selected observations must contain at least one observed pair")
        return pairs, values[pairs]


# Inputs accepted by training and calibration: one region batch, a sequence, or a
# mapping keyed by region name, each with one or several species' observations.
RegionCollection = RegionBatch | Sequence[RegionBatch] | Mapping[str, RegionBatch]
RegionObservations = PairwiseObservations | Sequence[PairwiseObservations]
ObservationCollection = (
    RegionObservations | Sequence[RegionObservations] | Mapping[str, RegionObservations]
)
