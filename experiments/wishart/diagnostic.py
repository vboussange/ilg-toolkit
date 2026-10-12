"""Strict Gaussian-marker Wishart references; this module enables no training.

The contract and conditional-score derivation live in
``docs/statistics/wishart.md``. Inputs declare known-zero-mean,
independent Gaussian marker contrasts; structural validation cannot establish
that scientific assumption for real genetic observations.
"""

import math
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from ilg_toolkit.config import ResistanceSolverConfig
from ilg_toolkit.data import RegionBatch
from ilg_toolkit.models import ConductanceModel
from ilg_toolkit.resistance import build_resistance_context


def _helmert(populations):
    basis = np.zeros((populations - 1, populations), dtype=np.float64)
    for row in range(populations - 1):
        count = row + 1
        denominator = math.sqrt(count * (count + 1))
        basis[row, :count] = 1 / denominator
        basis[row, count] = -count / denominator
    return basis


def _positive_definite(matrix, name, *, shape=None):
    if np.iscomplexobj(matrix):
        raise ValueError(f"{name} must be real")
    matrix = np.asarray(matrix, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or not matrix.size
        or (shape is not None and matrix.shape != shape)
        or not np.isfinite(matrix).all()
    ):
        raise ValueError(f"{name} must be a finite square matrix of the required dimension")
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    eigenvalues = np.linalg.eigvalsh(matrix)
    numerical_rank_floor = np.finfo(np.float64).eps * matrix.shape[0] * np.max(np.abs(eigenvalues))
    if eigenvalues[0] <= numerical_rank_floor:
        raise ValueError(
            f"{name} must be strictly positive definite at float64 precision; "
            "singular/indefinite matrices are unsupported and are never repaired"
        )
    try:
        factor = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            f"{name} must be strictly positive definite; singular/indefinite matrices "
            "are unsupported and are never repaired"
        ) from error
    return matrix, factor


@dataclass(frozen=True)
class GaussianMarkerDistances:
    """Complete squared Gaussian-marker distances with explicit scatter scaling.

    ``marker_count`` counts independent, known-zero-mean Gaussian marker contrast
    replicates, not sampling units. ``marker_scaling`` is explicitly ``average``
    or ``scatter``. The only admitted ``interpretation`` is
    ``squared_gaussian_marker_distance``. These declarations are assumptions,
    not statistical validation of Gaussianity or marker independence.
    """

    region_name: str
    sampling_unit_ids: tuple[str, ...]
    distances: np.ndarray
    marker_count: int
    marker_scaling: str
    interpretation: str

    def __post_init__(self):
        labels = tuple(self.sampling_unit_ids)
        if (
            not isinstance(self.region_name, str)
            or not self.region_name
            or len(labels) < 2
            or any(not isinstance(label, str) or not label for label in labels)
            or len(set(labels)) != len(labels)
        ):
            raise ValueError(
                "A named region and at least two unique sampling-unit labels are required"
            )
        if self.interpretation != "squared_gaussian_marker_distance":
            raise ValueError(
                "Wishart diagnostics support only explicitly declared "
                "squared_gaussian_marker_distance; "
                "FST, relatedness and generic dissimilarities are unsupported"
            )
        if self.marker_scaling not in {"average", "scatter"}:
            raise ValueError("marker_scaling must explicitly declare average or scatter")
        if (
            isinstance(self.marker_count, bool)
            or not isinstance(self.marker_count, Integral)
            or self.marker_count < len(labels) - 1
        ):
            raise ValueError(
                "marker_count must be an explicit integer >= population count - 1; "
                "it counts independent Gaussian markers and is never inferred from populations"
            )
        if np.iscomplexobj(self.distances):
            raise ValueError("Squared Gaussian-marker distances must be real")
        distances = np.array(self.distances, dtype=np.float64, copy=True)
        if distances.shape != (len(labels), len(labels)) or not np.isfinite(distances).all():
            raise ValueError(
                "A complete finite distance matrix is required; missing pairs are unsupported"
            )
        if (
            np.any(distances < 0)
            or not np.array_equal(distances, distances.T)
            or np.any(np.diag(distances) != 0)
        ):
            raise ValueError("Squared distances must be nonnegative, symmetric, with zero diagonal")
        basis = _helmert(len(labels))
        centered = -0.5 * basis @ distances @ basis.T
        _positive_definite(centered, "Centered empirical covariance")
        distances.setflags(write=False)
        object.__setattr__(self, "sampling_unit_ids", labels)
        object.__setattr__(self, "distances", distances)

    @property
    def contrast_basis(self):
        """Return row-orthonormal Helmert population contrasts in labelled order."""
        return _helmert(len(self.sampling_unit_ids))

    @property
    def centered_scatter(self):
        """Return scatter in the declared orthonormal contrast coordinates."""
        basis = self.contrast_basis
        scatter = -0.5 * basis @ self.distances @ basis.T
        return scatter * self.marker_count if self.marker_scaling == "average" else scatter


def _wishart_logpdf(scatter, covariance, marker_count):
    scatter, scatter_factor = _positive_definite(scatter, "Empirical scatter")
    covariance, covariance_factor = _positive_definite(
        covariance, "Model covariance", shape=scatter.shape
    )
    dimension = scatter.shape[0]
    if marker_count <= dimension - 1:
        raise ValueError("Wishart degrees of freedom must exceed contrast dimension - 1")
    logdet_scatter = 2 * np.log(np.diag(scatter_factor)).sum()
    logdet_covariance = 2 * np.log(np.diag(covariance_factor)).sum()
    multivariate_gamma = dimension * (dimension - 1) / 4 * math.log(math.pi) + sum(
        math.lgamma((marker_count + 1 - index) / 2) for index in range(1, dimension + 1)
    )
    result = (
        (marker_count - dimension - 1) / 2 * logdet_scatter
        - 0.5 * np.trace(np.linalg.solve(covariance, scatter))
        - marker_count * dimension / 2 * math.log(2)
        - marker_count / 2 * logdet_covariance
        - multivariate_gamma
    )
    if not math.isfinite(result):
        raise FloatingPointError("Nonfinite reference Wishart log density")
    return float(result)


def _require_marker_observations(observations):
    if not isinstance(observations, GaussianMarkerDistances):
        raise TypeError(
            "Expected GaussianMarkerDistances with explicit interpretation, "
            "marker count and scaling; "
            "generic genetic targets are not automatically Wishart observations"
        )


def wishart_log_likelihood(observations: GaussianMarkerDistances, covariance) -> float:
    """Normalized log density in the input's declared Helmert-coordinate scale.

    ``covariance`` is the per-marker covariance in ``observations.contrast_basis``.
    Average inputs include the Jacobian from scatter to marker-average covariance;
    constants are retained so dimensions and marker scaling remain meaningful.
    This is a dense NumPy diagnostic, without gradients or optimizer integration.
    """
    _require_marker_observations(observations)
    result = _wishart_logpdf(observations.centered_scatter, covariance, observations.marker_count)
    if observations.marker_scaling == "average":
        dimension = len(observations.sampling_unit_ids) - 1
        result += dimension * (dimension + 1) / 2 * math.log(observations.marker_count)
    return result


@dataclass(frozen=True)
class HeldoutWishartScore:
    """Conditional density of held-out scatter blocks in explicit anchor coordinates.

    The score includes cross-population and held-out/held-out blocks, given the
    training/training block. It is not a mean pairwise loss or a density of raw
    marker measurements. ``parameter_source`` records the caller's declaration;
    this reference operation performs no fitting or historical leakage audit.
    """

    log_likelihood: float
    training_log_likelihood: float
    full_log_likelihood: float
    training_unit_ids: tuple[str, ...]
    heldout_unit_ids: tuple[str, ...]
    anchor_id: str
    marker_scaling: str
    parameter_source: str
    coordinate_measure: str


def heldout_log_likelihood(
    observations: GaussianMarkerDistances,
    covariance,
    *,
    training_unit_ids,
    anchor_id: str,
    parameter_source: str,
) -> HeldoutWishartScore:
    """Score held-out populations using a normalized joint/marginal density ratio.

    Choose a training anchor before fitting. Fix ``covariance`` from training
    observations only, or supply known fixed parameters. The training block uses
    training distances only; all remaining blocks are scored jointly. Shared
    marker count, scaling and anchor coordinates are preserved. Arbitrary pair
    masks, fitting with held-out targets, and changing markers by block are not
    supported protocols.
    """
    _require_marker_observations(observations)
    if parameter_source not in {"fixed", "training_only"}:
        raise ValueError("parameter_source must declare fixed or training_only parameters")
    training = tuple(training_unit_ids)
    labels = observations.sampling_unit_ids
    if (
        len(training) < 2
        or len(training) >= len(labels)
        or any(not isinstance(label, str) or label not in labels for label in training)
        or len(set(training)) != len(training)
        or anchor_id not in training
    ):
        raise ValueError(
            "Holdout requires >=2 unique known training units including the anchor, "
            "and at least one held-out population"
        )
    covariance, _ = _positive_definite(
        covariance, "Model covariance", shape=(len(labels) - 1, len(labels) - 1)
    )
    heldout = tuple(label for label in labels if label not in training)
    order = tuple(label for label in training if label != anchor_id) + heldout
    anchor_index = labels.index(anchor_id)
    contrast = np.eye(len(labels))[[labels.index(label) for label in order]]
    contrast[:, anchor_index] -= 1
    # Direct anchored contrasts ensure the training principal block reads only
    # training distances, rather than recentering all observations together.
    full_scatter = -0.5 * contrast @ observations.distances @ contrast.T
    if observations.marker_scaling == "average":
        full_scatter *= observations.marker_count
    transform = contrast @ observations.contrast_basis.T
    full_covariance = transform @ covariance @ transform.T
    train_dimension = len(training) - 1
    full_logpdf = _wishart_logpdf(full_scatter, full_covariance, observations.marker_count)
    training_logpdf = _wishart_logpdf(
        full_scatter[:train_dimension, :train_dimension],
        full_covariance[:train_dimension, :train_dimension],
        observations.marker_count,
    )
    if observations.marker_scaling == "average":
        full_dimension = len(labels) - 1
        full_logpdf += (
            full_dimension * (full_dimension + 1) / 2 * math.log(observations.marker_count)
        )
        training_logpdf += (
            train_dimension * (train_dimension + 1) / 2 * math.log(observations.marker_count)
        )
    return HeldoutWishartScore(
        full_logpdf - training_logpdf,
        training_logpdf,
        full_logpdf,
        training,
        heldout,
        anchor_id,
        observations.marker_scaling,
        parameter_source,
        "anchored_marker_average_covariance_entries"
        if observations.marker_scaling == "average"
        else "anchored_scatter_entries",
    )


@dataclass(frozen=True)
class WishartDiagnostic:
    """Fixed-parameter reference evidence, including an explicit training gate."""

    region_name: str
    resistance_scores: np.ndarray
    model_covariance: np.ndarray
    log_likelihood: float
    heldout_score: HeldoutWishartScore
    scale_nugget_rank: int
    scale_nugget_condition_number: float
    training_gate: str
    gate_reasons: tuple[str, ...]


def diagnose_wishart(
    region: RegionBatch,
    observations: GaussianMarkerDistances,
    *,
    encoder: ConductanceModel,
    scale: float,
    nugget: float,
    training_unit_ids,
    anchor_id: str,
    parameter_source: str,
    solver_config: ResistanceSolverConfig | None = None,
) -> WishartDiagnostic:
    """Connect an actual resistance graph to a frozen Gaussian-marker covariance.

    In orthonormal population contrasts H, Sigma = scale*(-H R H.T/2)+nugget*I.
    Scale is positive; nugget is nonnegative scientific variance, not numerical
    jitter. No target repair, parameter estimation, gradient or training switch
    is provided. The scale/nugget rank describes fixed-encoder algebra only;
    shared scale identification during encoder training remains unresolved.
    """
    _require_marker_observations(observations)
    if not isinstance(encoder, ConductanceModel):
        raise TypeError("Wishart graph diagnostics require a conductance encoder")
    if region.name != observations.region_name or set(region.sampling_unit_ids) != set(
        observations.sampling_unit_ids
    ):
        raise ValueError("Region identity and sampling-unit labels must match the observations")
    kinds = region.sampling_unit_kinds
    assert kinds is not None  # RegionBatch normalizes this optional constructor input.
    if any(kind != "population" for kind in kinds):
        raise ValueError(
            "This experimental contract supports declared population sampling units only"
        )
    for name, value, lower_inclusive in (("scale", scale, False), ("nugget", nugget, True)):
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or (value < 0 if lower_inclusive else value <= 0)
        ):
            raise ValueError(
                f"{name} must be finite and {'nonnegative' if lower_inclusive else 'positive'}"
            )
    height, width = region.feature_array.shape[:2]
    if height % encoder.patch_size or width % encoder.patch_size:
        raise ValueError("Raster dimensions must be divisible by encoder patch_size")
    context = build_resistance_context(
        (height // encoder.patch_size, width // encoder.patch_size), solver_config
    )
    scores = np.asarray(
        encoder.predict_distances(region.feature_array, region.pixel_nodes, context=context)
    )
    order = [region.sampling_unit_ids.index(label) for label in observations.sampling_unit_ids]
    scores = np.array(scores[np.ix_(order, order)], dtype=np.float64)
    if not np.isfinite(scores).all() or np.any(scores < 0):
        raise FloatingPointError("The graph produced nonfinite or negative resistance scores")
    basis = observations.contrast_basis
    kernel = -0.5 * basis @ scores @ basis.T
    covariance = scale * kernel + nugget * np.eye(kernel.shape[0])
    _positive_definite(covariance, "Model covariance")
    design = np.column_stack((kernel.ravel(), np.eye(kernel.shape[0]).ravel()))
    rank = int(np.linalg.matrix_rank(design))
    condition = float(np.linalg.cond(design)) if rank == 2 else float("inf")
    heldout = heldout_log_likelihood(
        observations,
        covariance,
        training_unit_ids=training_unit_ids,
        anchor_id=anchor_id,
        parameter_source=parameter_source,
    )
    scores.setflags(write=False)
    covariance.setflags(write=False)
    return WishartDiagnostic(
        region.name,
        scores,
        covariance,
        wishart_log_likelihood(observations, covariance),
        heldout,
        rank,
        condition,
        "no_go",
        (
            "Joint encoder/regional scale fitting requires an explicit "
            "normalization or fixed scale.",
            "Empirical marker validity and effective counts for "
            "dependent/non-Gaussian markers are unresolved.",
            "A reviewed go contract for experimental training and provenance "
            "has not been established.",
        ),
    )
