"""Target contracts, region batches and pairwise genetic observations."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast, NamedTuple

import coordax as cx
import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike


class _TargetTransform(NamedTuple):
    """Maps between the original and model scales, with the values each accepts."""

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
    """Target contract of pairwise genetic observations: meaning, units and transform.

    ``transform`` maps measurements from the original scale to the model scale,
    and genetic predictions apply its inverse:

    - ``"identity"``: ``x``.
    - ``"log1p"``: ``log(1 + x)``, for nonnegative values.
    - ``"sqrt"``: ``sqrt(x)``, for nonnegative values.
    - ``"odds"``: ``x / (1 - x)``, for values in [0, 1), with inverse ``y / (1 + y)``.
      It linearizes dissimilarities bounded by one, such as Rousset's linearized
      Fst, ``Fst / (1 - Fst)``. Negative values raise an error; remove or clip
      negative Fst estimates before constructing observations.

    Args:
        name: Name of the measured genetic quantity, such as ``"Fst"``.
        units: Units of the original scale.
        kind: ``"dissimilarity"``, ``"relatedness"`` or ``"similarity"``.
        transform: ``"identity"``, ``"log1p"``, ``"sqrt"`` or ``"odds"``.

    Raises:
        ValueError: If a field is empty or names an unknown kind or transform.
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

    def forward(self, values: ArrayLike) -> jax.Array:
        """Transforms original-scale measurements to the model scale.

        Args:
            values: Finite measurements on the original scale.

        Returns:
            The measurements on the model scale.

        Raises:
            ValueError: If a value is nonfinite or outside the transform's domain.
        """
        values = jnp.asarray(values)
        if not bool(jnp.isfinite(values).all()):
            raise ValueError("Observed target values must be finite")
        transform = _TARGET_TRANSFORMS[self.transform]
        if transform.domain is not None:
            accepts, description = transform.domain
            if not bool(accepts(values).all()):
                raise ValueError(f"Target transform {self.transform} requires {description}")
        return transform.forward(values)

    def inverse(self, values: ArrayLike) -> jax.Array:
        """Transforms model-scale predictions to the original scale.

        Args:
            values: Predictions on the model scale.

        Returns:
            The predictions in the target's original units.

        Raises:
            ValueError: If a value is negative under the ``"sqrt"`` or ``"odds"``
                transform.
            FloatingPointError: If the inverse transform produces a nonfinite value.
        """
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


def _feature_names(names) -> tuple[str, ...]:
    """Returns feature names as a tuple after checking they are unique and nonempty."""
    names = tuple(names)
    if (
        not names
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("Feature names must be unique nonempty strings, one per channel")
    return names


def _cell_indices(axis: cx.Coordinate | None, dimension: str, values: np.ndarray) -> np.ndarray:
    """Returns the index of the raster cell containing each coordinate, or -1 outside it.

    Cell edges lie halfway between neighboring cell centers, and the outer edges half a
    cell beyond the first and last centers. Like GDAL, a shared edge belongs to the cell
    with the larger index. The outer edges, widened by a millionth of the smallest cell,
    belong to the raster.
    """
    if not isinstance(axis, cx.LabeledAxis) or not np.issubdtype(axis.ticks.dtype, np.number):
        raise ValueError(
            f"Building from points requires numeric {dimension} coordinates, "
            "such as those cx.from_xarray keeps"
        )
    centers = axis.ticks.astype(np.float64)
    steps = np.diff(centers)
    if (
        centers.size < 2
        or not np.isfinite(centers).all()
        or not ((steps > 0).all() or (steps < 0).all())
    ):
        raise ValueError(
            f"{dimension} coordinates must be finite and strictly monotonic over at least two cells"
        )
    # Orient coordinates along increasing index so that edges increase.
    direction = np.sign(steps[0])
    centers, values = direction * centers, direction * values
    edges = np.concatenate(
        (
            [1.5 * centers[0] - 0.5 * centers[1]],
            (centers[1:] + centers[:-1]) / 2,
            [1.5 * centers[-1] - 0.5 * centers[-2]],
        )
    )
    tolerance = 1e-6 * np.abs(steps).min()
    inside = (values >= edges[0] - tolerance) & (values <= edges[-1] + tolerance)
    indices = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, centers.size - 1)
    return np.where(inside, indices, -1)


@dataclass(frozen=True)
class RegionBatch:
    """Region batch: the landscape features and sampling-unit grid positions of one region.

    Features are a coordax field with ``row``, ``column`` and ``feature``
    dimensions. A ``cx.LabeledAxis`` on the feature dimension names each layer,
    and the batch keeps row and column coordinates, such as those from
    ``cx.from_xarray``. Synthetic studies give each sampling unit's grid position
    directly; [`from_points`][ilg_toolkit.RegionBatch.from_points] snaps projected
    coordinates to raster cells.

    Args:
        name: Region name.
        features: Landscape features with ``row``, ``column`` and ``feature``
            dimensions and a labeled feature axis.
        sampling_unit_ids: Sampling-unit identifiers, unique within each species.
        grid_positions: Integer ``(row, column)`` raster cell of each sampling unit,
            with shape ``(sampling units, 2)``.
        species: Species of each sampling unit, for sampling units of several
            species; ``None`` for one species.
        individual_based: Whether the sampling units are individuals. MLPE losses
            and MLPE calibration heads reject individual-based region batches.

    Raises:
        TypeError: If ``features`` is not a ``coordax.Field`` or
            ``individual_based`` is not a bool.
        ValueError: If the features, identifiers, species or grid positions are
            invalid.

    Examples:
        A synthetic landscape with two named features::

            names = cx.LabeledAxis("feature", np.array(["elevation", "canopy"]))
            features = cx.field(values, "row", "column", names)
            region = RegionBatch("valley", features, ("a", "b"), [[0, 0], [3, 4]])
    """

    name: str
    features: cx.Field
    sampling_unit_ids: tuple[str, ...]
    grid_positions: ArrayLike
    species: tuple[str, ...] | None = None
    individual_based: bool = False

    def __post_init__(self):
        if not isinstance(self.features, cx.Field):
            raise TypeError("features must be a coordax.Field with named row, column, feature axes")
        if self.features.ndim != 3 or set(self.features.dims) != {"row", "column", "feature"}:
            raise ValueError("Labeled features require named axes row, column, feature")
        field = self.features.order_as("row", "column", "feature")
        axis = field.axes.get("feature")
        if not isinstance(axis, cx.LabeledAxis):
            raise ValueError(
                "features require a labeled feature axis naming each channel, "
                'such as cx.LabeledAxis("feature", names)'
            )
        _feature_names(axis.ticks.tolist())
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
        if not isinstance(self.individual_based, bool):
            raise TypeError("individual_based must be a bool")
        object.__setattr__(
            self,
            "features",
            cx.Field(features, dims=("row", "column", "feature"), axes=dict(field.axes)),
        )
        object.__setattr__(self, "sampling_unit_ids", ids)
        object.__setattr__(self, "species", species)
        object.__setattr__(self, "grid_positions", positions.astype(jnp.int32))

    @classmethod
    def from_points(
        cls,
        name: str,
        features: cx.Field,
        sampling_unit_ids: Sequence[str],
        coordinates: ArrayLike,
        *,
        species: Sequence[str] | None = None,
        individual_based: bool = False,
    ) -> "RegionBatch":
        """Builds a region batch by snapping projected coordinates to raster cells.

        Row and column coordinates are cell centers, as ``cx.from_xarray`` keeps
        them from rioxarray rasters, and cell edges lie halfway between neighboring
        centers, so each dimension needs at least two cells. The outer edges lie
        half a cell beyond the outer centers, and points up to a millionth of a cell
        beyond them belong to the outer cells. Rows may follow decreasing y, as in
        most GeoTIFFs. A point on an edge shared by two cells belongs to the cell
        with the larger row or column index, as in GDAL.

        Args:
            name: Region name.
            features: Landscape features with labeled row, column and feature axes,
                for instance
                ``cx.from_xarray(raster.rename(y="row", x="column", band="feature"))``.
            sampling_unit_ids: Sampling-unit identifiers, aligned with ``coordinates``.
            coordinates: Projected ``(x, y)`` coordinates of each sampling unit, in
                the raster's coordinate reference system.
            species: Species of each sampling unit, for sampling units of several
                species; ``None`` for one species.
            individual_based: Whether the sampling units are individuals.

        Returns:
            The region batch with each sampling unit's grid position.

        Raises:
            TypeError: If ``features`` is not a ``coordax.Field``.
            ValueError: If the features lack numeric, strictly monotonic row and
                column coordinates over at least two cells each, or a point lies
                outside the raster extent; the message names those sampling units.
        """
        if not isinstance(features, cx.Field):
            raise TypeError("features must be a coordax.Field with named row, column, feature axes")
        ids = tuple(sampling_unit_ids)
        points = np.asarray(coordinates, dtype=np.float64)
        if points.shape != (len(ids), 2) or not np.isfinite(points).all():
            raise ValueError(
                "coordinates must be a finite (sampling units, 2) array of projected x, y"
            )
        rows = _cell_indices(features.axes.get("row"), "row", points[:, 1])
        columns = _cell_indices(features.axes.get("column"), "column", points[:, 0])
        outside = np.flatnonzero((rows < 0) | (columns < 0))
        if outside.size:
            raise ValueError(
                "Sampling units lie outside the raster extent: "
                + ", ".join(str(ids[index]) for index in outside)
            )
        return cls(
            name,
            features,
            ids,
            np.stack((rows, columns), axis=1),
            species=None if species is None else tuple(species),
            individual_based=individual_based,
        )

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Feature names in channel order, read from the labeled feature axis."""
        axis = cast(cx.LabeledAxis, self.features.axes["feature"])
        return tuple(axis.ticks.tolist())

    @property
    def feature_array(self) -> jax.Array:
        """Landscape features as a JAX array with row, column and feature dimensions."""
        return cast(jax.Array, self.features.data)

    @property
    def pixel_nodes(self) -> jax.Array:
        """Row-major raster cell index of each sampling unit."""
        positions = cast(jax.Array, self.grid_positions)
        return positions[:, 0] * self.features.shape[1] + positions[:, 1]


def _validate_pairs(pairs) -> tuple[tuple[str, str], ...]:
    """Returns pairs as label tuples after checking they are distinct and unique.

    Pair identities are unordered; the returned pairs keep their order and orientation.
    """
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


@dataclass(frozen=True, init=False)
class PairwiseObservations:
    """Pairwise genetic observations of one species, as a labeled symmetric matrix.

    Unobserved pairs are not-a-number (NaN) entries.
    [`from_pairs`][ilg_toolkit.PairwiseObservations.from_pairs] builds the matrix
    from labeled pairs. Floating-point values keep their precision; other values
    convert at the current JAX precision.

    Args:
        sampling_unit_ids: Unique sampling-unit identifiers ordering the matrix rows
            and columns.
        values: Symmetric matrix with a zero diagonal, NaN entries for unobserved
            pairs and at least one observed pair.
        target: Target contract of the measurements.
        species: Species of the sampling units, or ``None`` for one unnamed species.

    Raises:
        ValueError: If the identifiers, the matrix or the species is invalid.
    """

    sampling_unit_ids: tuple[str, ...]
    values: jax.Array
    target: TargetSpec
    species: str | None = None

    def __init__(
        self,
        sampling_unit_ids: Sequence[str],
        values: ArrayLike,
        target: TargetSpec,
        species: str | None = None,
    ):
        if species is not None and (not isinstance(species, str) or not species):
            raise ValueError("Observation species must be a nonempty name or None")
        ids = tuple(sampling_unit_ids)
        values = jnp.asarray(values)
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
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "species", species)

    @classmethod
    def from_pairs(
        cls,
        pairs: Sequence[tuple[str, str]],
        values: ArrayLike,
        *,
        target: TargetSpec,
        sampling_unit_ids: Sequence[str] | None = None,
        species: str | None = None,
    ) -> "PairwiseObservations":
        """Builds observations from labeled pairs and their measurements.

        Args:
            pairs: Unique unordered sampling-unit pairs; a pair and its reversal
                are the same pair.
            values: Finite measurement of each pair, on the original scale.
            target: Target contract of the measurements.
            sampling_unit_ids: Sampling-unit identifiers ordering the matrix,
                including sampling units without observed pairs; ``None`` uses the
                sorted identifiers of the pairs.
            species: Species of the sampling units, or ``None`` for one unnamed
                species.

        Returns:
            The observations, with NaN entries for pairs not in ``pairs``.

        Raises:
            ValueError: If a pair or measurement is invalid, or a pair names a
                sampling unit absent from ``sampling_unit_ids``.
        """
        pairs = _validate_pairs(pairs)
        values = jnp.asarray(values)
        if values.shape != (len(pairs),) or not jnp.isfinite(values).all():
            raise ValueError("Observed pairs must be nonempty and have one finite value each")
        ids = (
            tuple(sampling_unit_ids)
            if sampling_unit_ids is not None
            else tuple(sorted({endpoint for pair in pairs for endpoint in pair}))
        )
        lookup = {label: index for index, label in enumerate(ids)}
        if any(endpoint not in lookup for pair in pairs for endpoint in pair):
            raise ValueError("An observed sampling unit is absent from sampling_unit_ids")
        left, right = (jnp.asarray([lookup[pair[end]] for pair in pairs]) for end in (0, 1))
        dtype = (
            values.dtype
            if jnp.issubdtype(values.dtype, jnp.inexact)
            else jnp.result_type(values, 0.0)
        )
        matrix = jnp.full((len(ids), len(ids)), jnp.nan, dtype=dtype)
        matrix = matrix.at[jnp.diag_indices(len(ids))].set(0)
        matrix = matrix.at[left, right].set(values).at[right, left].set(values)
        return cls(ids, matrix, target, species)

    @property
    def observed_pairs(self) -> tuple[tuple[str, str], ...]:
        """Observed unordered sampling-unit pairs, in matrix order."""
        left, right = jnp.triu_indices(len(self.sampling_unit_ids), 1)
        present = jnp.isfinite(self.values[left, right])
        return tuple(
            (self.sampling_unit_ids[i], self.sampling_unit_ids[j])
            for i, j in zip(left[present], right[present], strict=True)
        )

    @property
    def observed_values(self) -> jax.Array:
        """Original-scale measurements, aligned with ``observed_pairs``."""
        upper = self.values[jnp.triu_indices(len(self.sampling_unit_ids), 1)]
        return upper[jnp.isfinite(upper)]

    def aligned_values(self, region: RegionBatch) -> jax.Array:
        """Returns the observation matrix in a region batch's sampling-unit order.

        Args:
            region: Region batch holding the observed sampling units of this species.

        Returns:
            Matrix over the region batch's sampling units, with a zero diagonal and
            NaN entries for unobserved pairs.

        Raises:
            ValueError: If an observed sampling unit is absent from the region batch
                or belongs to another species.
        """
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

    def select_pairs(self, pairs: Sequence[tuple[str, str]]) -> "PairwiseObservations":
        """Returns the observations of the selected pairs.

        Args:
            pairs: Observed unordered pairs to keep, in any order and orientation.

        Returns:
            Observations with the same target, species, precision and sampling-unit
            identifiers, whose ``observed_pairs`` and ``observed_values`` follow the
            matrix order.

        Raises:
            ValueError: If the selection is empty, repeats a pair or contains an
                unobserved pair.

        Examples:
            >>> observations = PairwiseObservations.from_pairs(
            ...     [("a", "b"), ("b", "c")], [0.2, 0.4], target=TargetSpec("Fst")
            ... )
            >>> observations.select_pairs([("c", "b")]).observed_pairs
            (("b", "c"),)
        """
        selected = {tuple(sorted(pair)) for pair in _validate_pairs(pairs)}
        if not selected.issubset({tuple(sorted(pair)) for pair in self.observed_pairs}):
            raise ValueError("Selection contains an unobserved pair")
        lookup = {label: index for index, label in enumerate(self.sampling_unit_ids)}
        keep = np.eye(len(self.sampling_unit_ids), dtype=bool)
        for left, right in selected:
            keep[lookup[left], lookup[right]] = keep[lookup[right], lookup[left]] = True
        values = jnp.where(jnp.asarray(keep), self.values, jnp.nan)
        return type(self)(self.sampling_unit_ids, values, self.target, self.species)

    def aligned_pairs(self, region: RegionBatch) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        """Returns the observed pairs as sampling-unit indices of a region batch.

        Args:
            region: Region batch holding the observed sampling units of this species.

        Returns:
            Row and column indices of each observed pair in the region batch's
            sampling-unit order, and the original-scale measurements of those pairs.

        Raises:
            ValueError: If an observed sampling unit is absent from the region batch,
                or the region batch holds no observed pair.
        """
        values = self.aligned_values(region)
        left, right = jnp.triu_indices(len(region.sampling_unit_ids), 1)
        present = jnp.isfinite(values[left, right])
        pairs = left[present], right[present]
        if not len(pairs[0]):
            raise ValueError("Selected observations must contain at least one observed pair")
        return pairs, values[pairs]


# Regions and observations accepted by training and calibration fitting: one region
# batch, a sequence, or a mapping keyed by region name, with one or several species'
# observations per region.
_RegionCollection = RegionBatch | Sequence[RegionBatch] | Mapping[str, RegionBatch]
_RegionObservations = PairwiseObservations | Sequence[PairwiseObservations]
_ObservationCollection = (
    _RegionObservations | Sequence[_RegionObservations] | Mapping[str, _RegionObservations]
)
