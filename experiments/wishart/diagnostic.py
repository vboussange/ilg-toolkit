"""Wishart log likelihoods of Gaussian-marker distances under fixed graph covariances.

Inputs declare complete squared distances between independent, zero-mean Gaussian
marker contrasts. ``README.md`` beside this module lists the assumptions.
"""

import math
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from ilg_toolkit import RegionBatch
from ilg_toolkit.models import ConductanceModel
from scipy.linalg import helmert


def _positive_definite(matrix, name, *, shape=None):
    """Checks that a matrix is real, symmetric and positive definite, and returns its factor."""
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
        raise ValueError(f"{name} must be strictly positive definite at float64 precision")
    try:
        factor = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} must be strictly positive definite") from error
    return matrix, factor


@dataclass(frozen=True)
class GaussianMarkerDistances:
    """Complete squared Gaussian-marker distances with declared marker scaling.

    ``marker_count`` counts independent, zero-mean Gaussian marker contrast
    replicates. ``marker_scaling`` is ``"average"`` for marker-averaged squared
    distances or ``"scatter"`` for summed ones. ``interpretation`` is
    ``"squared_gaussian_marker_distance"``.
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
                "Wishart diagnostics require the interpretation "
                "squared_gaussian_marker_distance; "
                "FST, relatedness and other dissimilarities are unsupported"
            )
        if self.marker_scaling not in {"average", "scatter"}:
            raise ValueError("marker_scaling must be average or scatter")
        if (
            isinstance(self.marker_count, bool)
            or not isinstance(self.marker_count, Integral)
            or self.marker_count < len(labels) - 1
        ):
            raise ValueError(
                "marker_count must be an integer of at least the population count - 1, "
                "counting independent Gaussian markers"
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
        basis = helmert(len(labels))
        centered = -0.5 * basis @ distances @ basis.T
        _positive_definite(centered, "Centered empirical covariance")
        distances.setflags(write=False)
        object.__setattr__(self, "sampling_unit_ids", labels)
        object.__setattr__(self, "distances", distances)

    @property
    def contrast_basis(self):
        """The row-orthonormal Helmert population contrasts, in sampling-unit order."""
        return helmert(len(self.sampling_unit_ids))

    @property
    def centered_scatter(self):
        """The marker scatter in the orthonormal contrast coordinates."""
        basis = self.contrast_basis
        scatter = -0.5 * basis @ self.distances @ basis.T
        return scatter * self.marker_count if self.marker_scaling == "average" else scatter


def _wishart_logpdf(scatter, covariance, marker_count):
    """Returns the normalized Wishart log density of a scatter matrix."""
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
    """Raises a ``TypeError`` unless the observations are Gaussian-marker distances."""
    if not isinstance(observations, GaussianMarkerDistances):
        raise TypeError(
            "Expected GaussianMarkerDistances with a declared interpretation, "
            "marker count and marker scaling"
        )


def wishart_log_likelihood(observations: GaussianMarkerDistances, covariance) -> float:
    """Computes the normalized Wishart log density of the observations' contrast scatter.

    Marker-averaged observations include the Jacobian from scatter to
    marker-average covariance entries. The computation uses dense NumPy algebra.

    Args:
        observations: Gaussian-marker distances.
        covariance: Per-marker covariance in ``observations.contrast_basis``.

    Returns:
        The log density, in the coordinates of the declared marker scaling.
    """
    _require_marker_observations(observations)
    result = _wishart_logpdf(observations.centered_scatter, covariance, observations.marker_count)
    if observations.marker_scaling == "average":
        dimension = len(observations.sampling_unit_ids) - 1
        result += dimension * (dimension + 1) / 2 * math.log(observations.marker_count)
    return result


@dataclass(frozen=True)
class HeldoutWishartScore:
    """A conditional log density of held-out scatter blocks in anchored contrast coordinates.

    The score is the log density of the training–held-out and held-out–held-out
    blocks of the anchored scatter, given its training–training block.
    ``parameter_source`` records the caller's declaration of how the covariance
    was obtained.
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
    """Scores held-out populations by the ratio of the joint to the training density.

    Contrasts against the anchor define the coordinates. The training block reads
    training distances only, and every remaining block is scored jointly, with
    the same marker count and marker scaling.

    Args:
        observations: Gaussian-marker distances of every population.
        covariance: Per-marker covariance in ``observations.contrast_basis``, known
            or estimated from training observations only.
        training_unit_ids: At least two training sampling units, leaving at least
            one held out.
        anchor_id: The training sampling unit whose contrasts define the
            coordinates.
        parameter_source: ``"fixed"`` for known parameters, or ``"training_only"``
            for parameters estimated from training observations.

    Returns:
        The held-out score with its training and full log densities.

    Raises:
        ValueError: The partition, anchor, covariance or parameter source is invalid.
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
            "Holdout requires at least two unique known training units, including the "
            "anchor, and at least one held-out population"
        )
    covariance, _ = _positive_definite(
        covariance, "Model covariance", shape=(len(labels) - 1, len(labels) - 1)
    )
    heldout = tuple(label for label in labels if label not in training)
    order = tuple(label for label in training if label != anchor_id) + heldout
    anchor_index = labels.index(anchor_id)
    contrast = np.eye(len(labels))[[labels.index(label) for label in order]]
    contrast[:, anchor_index] -= 1
    # Contrasts against the anchor make the training block depend on training
    # distances only.
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
    """A fixed-parameter Wishart diagnostic of a conductance grid graph, with its training gate."""

    region_name: str
    resistances: np.ndarray
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
    patch_encoder: ConductanceModel,
    scale: float,
    nugget: float,
    training_unit_ids,
    anchor_id: str,
    parameter_source: str,
) -> WishartDiagnostic:
    """Scores a Gaussian-marker covariance built from a patch encoder's effective resistances.

    In orthonormal population contrasts H, the per-marker covariance is
    Sigma = scale * (-H R H.T / 2) + nugget * I, where R holds the effective
    resistances between sampling units. The rank and condition number of the
    scale–nugget design tell whether the fixed patch encoder separates scale and
    nugget.

    Args:
        region: Region batch of population sampling units.
        observations: Gaussian-marker distances of the region's sampling units.
        patch_encoder: Single-species patch encoder giving the effective
            resistances.
        scale: Positive scale of the resistance kernel.
        nugget: Nonnegative marker variance added to every contrast.
        training_unit_ids: Training sampling units of the held-out score.
        anchor_id: Training sampling unit whose contrasts define the coordinates
            of the held-out score.
        parameter_source: ``"fixed"`` or ``"training_only"``, as in
            ``heldout_log_likelihood``.

    Returns:
        The resistances, covariance, log likelihood, held-out score, scale–nugget
        design diagnostics and the ``"no_go"`` training gate with its reasons.
    """
    _require_marker_observations(observations)
    if not isinstance(patch_encoder, ConductanceModel):
        raise TypeError("Wishart graph diagnostics require a patch encoder")
    if region.name != observations.region_name or set(region.sampling_unit_ids) != set(
        observations.sampling_unit_ids
    ):
        raise ValueError("Region identity and sampling-unit labels must match the observations")
    if region.individual_based:
        raise ValueError("Wishart diagnostics require population sampling units")
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
    prediction = patch_encoder.predict(region)
    resistances = np.array(prediction.as_matrix(), dtype=np.float64)
    np.fill_diagonal(resistances, 0)
    order = [prediction.sampling_unit_ids.index(label) for label in observations.sampling_unit_ids]
    resistances = resistances[np.ix_(order, order)]
    if not np.isfinite(resistances).all() or np.any(resistances < 0):
        raise FloatingPointError("The graph produced nonfinite or negative effective resistances")
    basis = observations.contrast_basis
    kernel = -0.5 * basis @ resistances @ basis.T
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
    resistances.setflags(write=False)
    covariance.setflags(write=False)
    return WishartDiagnostic(
        region.name,
        resistances,
        covariance,
        wishart_log_likelihood(observations, covariance),
        heldout,
        rank,
        condition,
        "no_go",
        (
            "Joint training of a patch encoder and regional scales requires an "
            "explicit normalization or a fixed scale.",
            "Training requires validated marker independence, Gaussianity and "
            "effective marker counts.",
            "Training requires a reviewed contract for experimental training and provenance.",
        ),
    )
