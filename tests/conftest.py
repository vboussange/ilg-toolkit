"""Shared scientific data and numerical reference fixtures."""

import itertools
from collections.abc import Callable
from dataclasses import replace
from types import SimpleNamespace
from typing import final

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ilg_toolkit import (
    DirectCalibrationModule,
    Ensemble,
    LandscapeGeneticsModel,
    MLPELoss,
    PairwiseObservations,
    RegionBatch,
    ResistanceSolverConfig,
    TargetSpec,
)
from ilg_toolkit.mlpe import calibrate_mlpe
from ilg_toolkit.models import (
    AbstractDistanceModel,
    ConductanceModel,
    EmbeddingDistanceModel,
)


# The suite runs in float64; float32 behavior is tested inside `jax.enable_x64(False)`.
jax.config.update("jax_enable_x64", True)

_R_DISTANCES = np.array(
    [
        -1.0847845325555074,
        -0.7856562765597935,
        1.4076148426443598,
        -0.005894023710191141,
        -1.375475831778455,
        0.8982644456212491,
        -1.2894979474794355,
        1.5465626070493472,
        -1.0016534312119365,
        -0.07602355302299327,
        -0.9160524297215114,
        0.38800235096599806,
        -1.378941894269037,
        0.5617899637344953,
        -1.606879982245179,
        0.7412399591855078,
        -0.9872195123319644,
        1.9954955626370834,
        -0.2919254085375479,
        -0.9038371900227332,
        -0.9801366442736841,
        -0.21221830555519952,
        -0.3569980434650668,
        -0.5131156917245834,
        1.5220664939667614,
        -0.4262339398597622,
        0.4617831441001972,
        0.6577818753941715,
    ]
)


_R_TARGETS = np.array(
    [
        0.7867947762864243,
        0.5844969351554589,
        -0.19944738061299205,
        0.7820143760316058,
        1.4691315076235152,
        0.4817642650020033,
        0.9965003483084469,
        -0.6201612100865199,
        1.7103092068446286,
        1.1655587443537478,
        1.7845260919265316,
        1.0164728564848375,
        1.003964815178125,
        0.6983581279293211,
        2.5398922377512445,
        1.029933745907628,
        2.599668270650879,
        -0.6011444799317919,
        1.9785807335604428,
        2.513022681996845,
        2.4701382843858752,
        0.9982332167064097,
        2.087895648701189,
        2.5344018288602435,
        0.1677412867636987,
        2.8312101442620357,
        0.7609805324512989,
        1.2129905800176382,
    ]
)


_R_FIXED = np.array([1.2422795775895932, -0.6653657702662492])


_R_VARIANCES = np.array([0.23562746377731067, 0.02879401793899513])


_R_DISTANCE_CENTER = -0.14328369260805038
_R_DISTANCE_SCALE = 1.012896471531856


_R_LOG_LIKELIHOOD = -6.137989809270536


_R_BLUPS = np.array(
    [
        -0.7520639701118881,
        -0.46501385425700786,
        -0.13777554492600885,
        0.1958548966482916,
        0.3832154152413225,
        0.4273528598272225,
        0.7313024458218655,
        -0.3828722482437748,
    ]
)


def _dense_resistance(surface, nodes):
    """Dense Laplacian reference for effective resistance, constructed with NumPy."""
    rows, columns = surface.shape
    laplacian = np.zeros((surface.size, surface.size))
    for row in range(rows):
        for column in range(columns):
            first = row * columns + column
            for next_row, next_column in ((row + 1, column), (row, column + 1)):
                if next_row < rows and next_column < columns:
                    second = next_row * columns + next_column
                    weight = (surface[row, column] + surface[next_row, next_column]) / 2
                    incidence = np.zeros(surface.size)
                    incidence[first], incidence[second] = 1, -1
                    laplacian += weight * np.outer(incidence, incidence)
    inverse = np.linalg.pinv(laplacian, hermitian=True)
    diagonal = np.diag(inverse)
    matrix = diagonal[:, None] + diagonal[None, :] - 2 * inverse
    return matrix[nodes[:, None], nodes[None, :]]


@final
class _PointConductance(ConductanceModel):
    """Distance model that scales one landscape feature into a conductance surface."""

    log_scale: jax.Array
    num_species: int = eqx.field(static=True, default=1)
    patch_size: int = eqx.field(static=True, default=1)
    solver_config: ResistanceSolverConfig = eqx.field(
        static=True, kw_only=True, default=ResistanceSolverConfig()
    )

    def __init__(self, log_scale: jax.Array, *, solver_config=None, num_species=1):
        self.log_scale = log_scale
        self.num_species = num_species
        self.solver_config = ResistanceSolverConfig() if solver_config is None else solver_config

    def backbone(self, features):
        return features

    def conductance(self, latent, species_index):
        scale = self.log_scale if self.log_scale.ndim == 0 else self.log_scale[species_index]
        return jnp.exp(scale) * latent[..., 0]


@final
class _CustomDistance(AbstractDistanceModel):
    scale: jax.Array
    num_species: int = eqx.field(static=True, default=1)
    patch_size: int = eqx.field(static=True, default=1)

    def backbone(self, features):
        return features.reshape(-1) * self.scale

    def ecological_distances(self, latent, pairs, species_index):
        nodes, (left, right) = pairs
        values = latent[nodes]
        return (values[left] - values[right]) ** 2


@final
class _DropoutConductance(ConductanceModel):
    dropout: eqx.nn.Dropout = eqx.field(default_factory=lambda: eqx.nn.Dropout(0.5))
    conductance_dropout: eqx.nn.Dropout = eqx.field(default_factory=lambda: eqx.nn.Dropout(0.5))
    num_species: int = eqx.field(static=True, default=1)
    patch_size: int = eqx.field(static=True, default=1)
    solver_config: ResistanceSolverConfig = eqx.field(static=True, default=ResistanceSolverConfig())

    def backbone(self, features):
        return self.dropout(features)

    def conductance(self, latent, species_index):
        return jnp.exp(self.conductance_dropout(latent)[..., 0])


@pytest.fixture(name="DropoutConductance", scope="session")
def dropout_conductance_class():
    return _DropoutConductance


def _known_ensemble(*, num_species=1) -> tuple[RegionBatch, Ensemble, TargetSpec]:
    region = RegionBatch(
        "alpine",
        cx.field(np.arange(4).reshape(1, 4, 1), "row", "column", "feature"),
        ("a", "b", "c", "d"),
        np.array([[0, 0], [0, 1], [0, 2], [0, 3]]),
    )
    target = TargetSpec("divergence", units="index")
    if num_species == 2:
        region = replace(
            region,
            sampling_unit_ids=region.sampling_unit_ids * 2,
            grid_positions=np.tile(region.grid_positions, (2, 1)),
            sampling_unit_kinds=None,
            species=("frog",) * 4 + ("newt",) * 4,
        )
    members = [
        LandscapeGeneticsModel(
            _ScalarEmbedding(
                jnp.asarray(weight if num_species == 1 else [weight, weight + 2]),
                num_species=num_species,
            ),
            DirectCalibrationModule(target),
            1,
            species=None if num_species == 1 else ("frog", "newt"),
        )
        for weight in (1.0, 2.0)
    ]
    return region, Ensemble(members), target


def _reference_observations():
    ids = tuple(f"p{index:02d}" for index in range(1, 9))
    pairs = tuple(itertools.combinations(ids, 2))
    return PairwiseObservations.from_pairs(
        pairs, _R_TARGETS, target=TargetSpec("synthetic dissimilarity", units="index")
    )


def _calibrated():
    observations = _reference_observations()
    head = calibrate_mlpe(
        _R_DISTANCES, observations, region_name="alpine", loss=MLPELoss(jitter=1e-7)
    )
    return head, observations


def _joint_gaussian_reference(
    head, observed_pairs, observed_distances, targets, query_pairs, distances
):
    """Gaussian conditioning reference built from the joint observation covariance."""
    all_pairs = tuple(observed_pairs) + tuple(query_pairs)
    labels = sorted({label for pair in all_pairs for label in pair})
    incidence = np.array([[float(label in pair) for label in labels] for pair in all_pairs])
    covariance = head.unit_variance * incidence @ incidence.T
    covariance += (head.residual_variance + head.loss.jitter) * np.eye(len(all_pairs))
    all_distances = np.r_[observed_distances, distances]
    fixed = (
        head.intercept + head.slope * (all_distances - head.distance_center) / head.distance_scale
    )
    n = len(observed_pairs)
    cross = covariance[n:, :n]
    mean = fixed[n:] + cross @ np.linalg.solve(covariance[:n, :n], targets - fixed[:n])
    conditional = covariance[n:, n:] - cross @ np.linalg.solve(covariance[:n, :n], cross.T)
    return mean, np.diag(conditional)


def _frozen_r_conditional_reference(
    observed_pairs, observed_distances, targets, query_pairs, distances, *, jitter
):
    """NumPy conditional expectations derived from frozen R calibration results.

    Coefficients, variance components and distance moments come from the R
    reference; conditioning uses the supplied observation inputs. Residual
    variance plus jitter enters observation covariance and model variance.
    """
    labels = sorted({label for pair in (*observed_pairs, *query_pairs) for label in pair})
    observed = np.array([[float(label in pair) for label in labels] for pair in observed_pairs])
    query = np.array([[float(label in pair) for label in labels] for pair in query_pairs])
    noise = _R_VARIANCES[1] + jitter
    covariance = _R_VARIANCES[0] * observed @ observed.T + noise * np.eye(len(observed_pairs))
    cross = _R_VARIANCES[0] * query @ observed.T
    fixed_observed = (
        _R_FIXED[0] + _R_FIXED[1] * (observed_distances - _R_DISTANCE_CENTER) / _R_DISTANCE_SCALE
    )
    fixed_query = _R_FIXED[0] + _R_FIXED[1] * (distances - _R_DISTANCE_CENTER) / _R_DISTANCE_SCALE
    mean = fixed_query + cross @ np.linalg.solve(covariance, targets - fixed_observed)
    effect_covariance = _R_VARIANCES[0] * query @ query.T - cross @ np.linalg.solve(
        covariance, cross.T
    )
    effect_variance = np.diag(effect_covariance)
    return {
        "model_values": mean,
        "model_variance": effect_variance + noise,
        "effect_variance": effect_variance,
    }


@final
class _WeightedEmbedding(EmbeddingDistanceModel):
    """Distance model with a learned scale for each embedding feature."""

    log_weights: jax.Array
    num_species: int = eqx.field(static=True, default=1)
    patch_size: int = eqx.field(static=True, default=1)

    def __init__(self, log_weights: jax.Array, *, num_species=1):
        self.log_weights = log_weights
        self.num_species = num_species

    def backbone(self, features):
        return features

    def embedding(self, latent, species_index):
        weights = (
            self.log_weights if self.log_weights.ndim == 1 else self.log_weights[species_index]
        )
        return latent * jnp.exp(weights)


def _embedding_problem(name="alpine", offset=0.0, seed=14, *, columns=3):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(2, columns, 2)).astype(np.float32)
    unit_count = 2 * columns
    ids = tuple(f"unit-{i}" for i in range(unit_count))
    region = RegionBatch(
        name,
        cx.field(features, "row", "column", "feature"),
        ids,
        np.array(list(np.ndindex(2, columns))),
        feature_names=("elevation", "canopy"),
    )
    embedding = features.reshape(unit_count, 2) * np.array([0.3, 1.5])
    matrix = ((embedding[:, None] - embedding[None, :]) ** 2).sum(axis=-1)
    left, right = np.triu_indices(unit_count, 1)
    distances = matrix[left, right]
    effects = rng.normal(scale=0.04, size=unit_count)
    values = (
        1.4
        + offset
        - 0.4 * (distances - distances.mean()) / distances.std(ddof=1)
        + effects[left]
        + effects[right]
        + rng.normal(scale=0.02, size=len(left))
    )
    observations = PairwiseObservations.from_pairs(
        [(ids[a], ids[b]) for a, b in zip(left, right, strict=True)],
        values,
        target=TargetSpec("synthetic divergence", units="index"),
        sampling_unit_ids=ids,
    )
    return region, observations


def _dense_profile(distances, targets, left, right, n_populations, variances, *, jitter: float = 0):
    """Dense Gaussian likelihood reference for fitted calibration parameters."""
    distances = np.asarray(distances, dtype=np.float64)
    z = np.zeros((len(distances), n_populations))
    for row, (a, b) in enumerate(zip(left, right, strict=True)):
        z[row, a] = z[row, b] = 1
    x = np.column_stack(
        [np.ones(len(distances)), (distances - distances.mean()) / distances.std(ddof=1)]
    )
    v = variances[0] * (z @ z.T) + (variances[1] + jitter) * np.eye(len(distances))
    beta = np.linalg.lstsq(
        x.T @ np.linalg.solve(v, x), x.T @ np.linalg.solve(v, targets), rcond=None
    )[0]
    r = np.asarray(targets) - x @ beta
    nll = 0.5 * (
        len(distances) * np.log(2 * np.pi) + np.linalg.slogdet(v)[1] + r @ np.linalg.solve(v, r)
    )
    effect_mean = variances[0] * z.T @ np.linalg.solve(v, r)
    effect_covariance = variances[0] * np.eye(n_populations) - variances[
        0
    ] ** 2 * z.T @ np.linalg.solve(v, z)
    return nll, beta, effect_mean, effect_covariance


@final
class _CovariateConductance(ConductanceModel):
    weight: jax.Array
    num_species: int = eqx.field(static=True, default=1)
    patch_size: int = eqx.field(static=True, default=1)
    solver_config: ResistanceSolverConfig = eqx.field(
        static=True, kw_only=True, default=ResistanceSolverConfig()
    )

    def __init__(self, weight: jax.Array):
        self.weight = weight

    def backbone(self, features):
        return features

    def conductance(self, latent, species_index):
        return jnp.exp(self.weight * latent[..., 0])


@final
class _ScalarEmbedding(EmbeddingDistanceModel):
    weight: jax.Array
    num_species: int = eqx.field(static=True, default=1)
    patch_size: int = eqx.field(static=True, default=1)

    def __init__(self, weight: jax.Array, *, num_species=1):
        self.weight = weight
        self.num_species = num_species

    def backbone(self, features):
        return features

    def embedding(self, latent, species_index):
        weight = self.weight if self.weight.ndim == 0 else self.weight[species_index]
        return latent[..., :1] * weight


@final
class _SharedParameterEmbedding(EmbeddingDistanceModel):
    weights: jax.Array
    num_species: int = eqx.field(static=True)
    patch_size: int = eqx.field(static=True, default=1)

    on_embedding: Callable[[], None] | None = eqx.field(static=True, default=None)

    def __init__(self, weights, *, on_embedding=None):
        self.weights = weights
        self.num_species = len(weights)
        self.on_embedding = on_embedding

    def backbone(self, features):
        return features * self.weights.sum()

    def embedding(self, latent, species_index):
        if self.on_embedding is not None:
            self.on_embedding()
        return latent * self.weights[species_index]


def _regions_and_observations() -> tuple[list[RegionBatch], list[PairwiseObservations]]:
    regions = [
        RegionBatch(
            "wide",
            cx.field(np.array([[[0.0], [1.0], [2.0]]]), "row", "column", "feature"),
            ("a", "b", "c"),
            np.array([[0, 0], [0, 1], [0, 2]]),
            feature_names=("elevation",),
        ),
        RegionBatch(
            "short",
            cx.field(np.array([[[0.0], [2.0]]]), "row", "column", "feature"),
            ("x", "y"),
            np.array([[0, 0], [0, 1]]),
            feature_names=("elevation",),
        ),
    ]
    target = TargetSpec("synthetic divergence", units="index")
    observations = [
        PairwiseObservations.from_matrix(
            ("a", "b", "c"), [[0, 0.1, 0.1], [0.1, 0, 0.1], [0.1, 0.1, 0]], target=target
        ),
        PairwiseObservations.from_pairs([("x", "y")], [15], target=target),
    ]
    return regions, observations


@final
class _StochasticEmbedding(EmbeddingDistanceModel):
    distance_model: _ScalarEmbedding
    dropout: eqx.nn.Dropout
    num_species: int = eqx.field(static=True, default=1)
    patch_size: int = eqx.field(static=True, default=1)

    def __init__(self, weight: jax.Array):
        self.distance_model = _ScalarEmbedding(weight)
        self.dropout = eqx.nn.Dropout(0.3, inference=True)

    @property
    def weight(self) -> jax.Array:
        return self.distance_model.weight

    def backbone(self, features):
        return self.dropout(features)

    def embedding(self, latent, species_index):
        return self.distance_model.embedding(latent, species_index)


def _observation_problem() -> tuple[RegionBatch, TargetSpec, _ScalarEmbedding]:
    region = RegionBatch(
        "anywhere",
        cx.field(np.array([[[0.0, 2.0], [1.0, 3.0], [2.0, 4.0]]]), "row", "column", "feature"),
        ("unit/7", "unit/20", "unit/99"),
        np.array([[0, 0], [0, 1], [0, 2]]),
    )
    target = TargetSpec("pair dissimilarity", units="index")
    return region, target, _ScalarEmbedding(jnp.asarray(0.7))


def _population_effect_problem(
    name: str, target: TargetSpec
) -> tuple[RegionBatch, PairwiseObservations]:
    coordinates = np.array([0, 0.2, 0.8, 1.1, 1.9, 2.8])
    features = np.stack((coordinates, coordinates * 0.1 + 1), axis=-1).reshape(2, 3, 2)
    labels = tuple(f"population-{index}" for index in range(6))
    region = RegionBatch(
        name,
        cx.Field(jnp.asarray(features), dims=("row", "column", "feature")),
        labels,
        np.array(list(np.ndindex(2, 3))),
        feature_names=("east_covariate", "canopy_covariate"),
    )
    distances = (coordinates[:, None] - coordinates[None, :]) ** 2
    effects = np.array([0.2, -0.15, 0.1, 0.3, -0.2, 0.05])
    observed = 2 + 0.7 * distances + effects[:, None] + effects[None, :]
    noise = np.random.default_rng(17).normal(scale=0.12, size=(6, 6))
    observed += (noise + noise.T) / 2
    np.fill_diagonal(observed, 0)
    observations = PairwiseObservations.from_matrix(labels, observed, target=target)
    return region, observations


def _persistence_problem(name="inference"):
    return _population_effect_problem(
        name, TargetSpec("synthetic divergence", units="index", transform="log1p")
    )


def _calibration_problem(
    name: str = "original",
) -> tuple[RegionBatch, PairwiseObservations, PairwiseObservations, PairwiseObservations]:
    region, observations = _population_effect_problem(
        name, TargetSpec("synthetic divergence", units="index")
    )
    training = observations.select_pairs(observations.observed_pairs[:10])
    validation = observations.select_pairs(observations.observed_pairs[10:])
    return region, observations, training, validation


def _dense_jax_profile(distances, targets, left, right, n_populations, raw_variances):
    """Dense Gaussian likelihood and gradient reference using automatic differentiation."""
    import jax
    import jax.numpy as jnp
    from jax.scipy.linalg import cho_solve

    z = np.zeros((len(left), n_populations))
    for row, (a, b) in enumerate(zip(left, right, strict=True)):
        z[row, a] = z[row, b] = 1
    z = jnp.asarray(z, dtype=distances.dtype)
    variances = jax.nn.softplus(raw_variances) + 1e-10
    x = jnp.column_stack(
        (jnp.ones_like(distances), (distances - distances.mean()) / distances.std(ddof=1))
    )
    v = variances[0] * (z @ z.T) + variances[1] * jnp.eye(len(distances))
    factor = jnp.linalg.cholesky(v)
    beta = jnp.linalg.solve(
        x.T @ cho_solve((factor, True), x), x.T @ cho_solve((factor, True), targets)
    )
    r = targets - x @ beta
    return 0.5 * (
        len(distances) * np.log(2 * np.pi)
        + 2 * jnp.log(jnp.diag(factor)).sum()
        + r @ cho_solve((factor, True), r)
    ), beta


def _decimal_dense_profile_and_gradients(distances, targets, left, right, raw_variances):
    """60-digit dense Gaussian reference for covariance gradients.

    Uses standard-library Decimal Gaussian elimination.
    """
    import math
    from collections.abc import Iterable
    from decimal import Decimal, localcontext

    with localcontext() as context:
        context.prec = 60

        def d(value) -> Decimal:
            return Decimal(str(float(value)))

        def decimal_sum(values: Iterable[Decimal]) -> Decimal:
            return sum(values, Decimal(0))

        size = len(distances)
        s, y = list(map(d, distances)), list(map(d, targets))
        raw = list(map(d, raw_variances))
        u, e = [(Decimal(1) + value.exp()).ln() + Decimal("1e-10") for value in raw]
        overlap = [
            [len({left[i], right[i]} & {left[j], right[j]}) for j in range(size)]
            for i in range(size)
        ]
        # Gauss-Jordan inversion of the full pair covariance, with determinant.
        a = [
            [u * overlap[i][j] + (e if i == j else 0) for j in range(size)]
            + [Decimal(i == j) for j in range(size)]
            for i in range(size)
        ]
        determinant = Decimal(1)
        for i in range(size):
            pivot = max(range(i, size), key=lambda row: abs(a[row][i]))
            if pivot != i:
                a[i], a[pivot] = a[pivot], a[i]
                determinant = -determinant
            value = a[i][i]
            determinant *= value
            a[i] = [entry / value for entry in a[i]]
            for j in range(size):
                if j != i:
                    coefficient = a[j][i]
                    a[j] = [v - coefficient * b for v, b in zip(a[j], a[i], strict=True)]
        inverse = [row[size:] for row in a]
        center = decimal_sum(s) / size
        scale = (decimal_sum((value - center) ** 2 for value in s) / (size - 1)).sqrt()
        z = [(value - center) / scale for value in s]
        x = [[Decimal(1), value] for value in z]
        vinv_y = [
            decimal_sum(v * value for v, value in zip(row, y, strict=True)) for row in inverse
        ]
        vinv_x = [
            [decimal_sum(inverse[i][j] * x[j][k] for j in range(size)) for k in range(2)]
            for i in range(size)
        ]
        normal = [
            [decimal_sum(x[i][j] * vinv_x[i][k] for i in range(size)) for k in range(2)]
            for j in range(2)
        ]
        rhs = [decimal_sum(x[i][j] * vinv_y[i] for i in range(size)) for j in range(2)]
        denominator = normal[0][0] * normal[1][1] - normal[0][1] * normal[1][0]
        beta = [
            (rhs[0] * normal[1][1] - rhs[1] * normal[0][1]) / denominator,
            (rhs[1] * normal[0][0] - rhs[0] * normal[1][0]) / denominator,
        ]
        residual = [y[i] - beta[0] - beta[1] * z[i] for i in range(size)]
        w = [
            decimal_sum(v * value for v, value in zip(row, residual, strict=True))
            for row in inverse
        ]
        nll = (
            size * d(math.log(2 * math.pi))
            + determinant.ln()
            + decimal_sum(a * b for a, b in zip(residual, w, strict=True))
        ) / 2
        wz = decimal_sum(a * b for a, b in zip(w, z, strict=True))
        distance_gradient = [
            -beta[1] * (w[i] - decimal_sum(w) / size - z[i] * wz / (size - 1)) / scale
            for i in range(size)
        ]
        trace_unit = decimal_sum(
            inverse[i][j] * overlap[i][j] for i in range(size) for j in range(size)
        )
        quadratic_unit = decimal_sum(
            w[i] * w[j] * overlap[i][j] for i in range(size) for j in range(size)
        )
        variance_gradient = [
            (trace_unit - quadratic_unit) / (2 * (1 + (-raw[0]).exp())),
            (decimal_sum(inverse[i][i] for i in range(size)) - decimal_sum(value**2 for value in w))
            / (2 * (1 + (-raw[1]).exp())),
        ]
        return (
            float(nll),
            np.array(list(map(float, beta))),
            (
                np.array(list(map(float, distance_gradient))),
                np.array(list(map(float, variance_gradient))),
            ),
        )


@pytest.fixture(name="dense_resistance", scope="session")
def dense_resistance_fixture():
    return _dense_resistance


@pytest.fixture(name="PointConductance", scope="session")
def point_conductance_fixture():
    return _PointConductance


@pytest.fixture(name="CustomDistance", scope="session")
def custom_distance_fixture():
    return _CustomDistance


@pytest.fixture(name="FeatureEmbedding", scope="session")
def feature_embedding_fixture():
    return _ScalarEmbedding


@pytest.fixture(name="known_ensemble", scope="session")
def known_ensemble_fixture():
    return _known_ensemble


@pytest.fixture(name="reference_observations", scope="session")
def reference_observations_fixture():
    return _reference_observations


@pytest.fixture(name="calibrated", scope="session")
def calibrated_fixture():
    return _calibrated()


@pytest.fixture(name="joint_gaussian_reference", scope="session")
def joint_gaussian_reference_fixture():
    return _joint_gaussian_reference


@pytest.fixture(name="frozen_r_conditional_reference", scope="session")
def frozen_r_conditional_reference_fixture():
    return _frozen_r_conditional_reference


@pytest.fixture(name="WeightedEmbedding", scope="session")
def weighted_embedding_fixture():
    return _WeightedEmbedding


@pytest.fixture(name="embedding_problem", scope="session")
def embedding_problem_fixture():
    return _embedding_problem


@pytest.fixture(name="dense_profile", scope="session")
def dense_profile_fixture():
    return _dense_profile


@pytest.fixture(name="CovariateConductance", scope="session")
def covariate_conductance_fixture():
    return _CovariateConductance


@pytest.fixture(name="ScalarEmbedding", scope="session")
def scalar_embedding_fixture():
    return _ScalarEmbedding


@pytest.fixture(name="SharedParameterEmbedding", scope="session")
def shared_parameter_embedding_fixture():
    return _SharedParameterEmbedding


@pytest.fixture(name="regions_and_observations", scope="session")
def regions_and_observations_fixture():
    return _regions_and_observations


@pytest.fixture(name="StochasticEmbedding", scope="session")
def stochastic_embedding_fixture():
    return _StochasticEmbedding


@pytest.fixture(name="observation_problem", scope="session")
def observation_problem_fixture():
    return _observation_problem


@pytest.fixture(name="persistence_problem", scope="session")
def persistence_problem_fixture():
    return _persistence_problem


@pytest.fixture(name="calibration_problem", scope="session")
def calibration_problem_fixture():
    return _calibration_problem


@pytest.fixture(name="dense_jax_profile", scope="session")
def dense_jax_profile_fixture():
    return _dense_jax_profile


@pytest.fixture(name="decimal_dense_profile_and_gradients", scope="session")
def decimal_dense_profile_and_gradients_fixture():
    return _decimal_dense_profile_and_gradients


@pytest.fixture(scope="session")
def mlpe_reference():
    """Frozen maximum-likelihood population effects (MLPE) reference values.

    Computed with R 4.1.2, lme4 1.1-28 and ResistanceGA2 1.1.20, using full
    maximum likelihood and sample standard deviation scaling.
    """
    return SimpleNamespace(
        distances=_R_DISTANCES,
        targets=_R_TARGETS,
        fixed=_R_FIXED,
        variances=_R_VARIANCES,
        log_likelihood=_R_LOG_LIKELIHOOD,
        blups=_R_BLUPS,
    )


@pytest.fixture(scope="session")
def training_problem():
    """Build the same identifiable observations for each analytic distance-model family.

    Families are ``"embedding"``, ``"conductance"`` and the single-species
    ``"dropout"`` embedding, whose backbone applies dropout during training.
    """

    def build(family, *, name="alpine", num_species=1):
        features = np.random.default_rng(14).uniform(0.8, 1.6, size=(2, 2, 2)).astype(np.float32)
        region = RegionBatch(
            name,
            cx.field(features, "row", "column", "feature"),
            ("south", "north", "west", "east"),
            np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            feature_names=("elevation", "canopy"),
        )
        observations = PairwiseObservations.from_pairs(
            tuple(itertools.combinations(region.sampling_unit_ids, 2)),
            [0.4, 0.7, 1.4, 0.6, 0.9, 0.8],
            target=TargetSpec("synthetic divergence", units="index"),
        )
        if family == "embedding":
            weights = jnp.zeros(2) if num_species == 1 else jnp.array([[0.0, 0.0], [0.2, -0.3]])
            model = _WeightedEmbedding(weights, num_species=num_species)
        elif family == "conductance":
            scales = jnp.array(0.0) if num_species == 1 else jnp.arange(num_species) * 0.2
            model = _PointConductance(scales, num_species=num_species)
        elif family == "dropout" and num_species == 1:
            model = _StochasticEmbedding(jnp.asarray(1.0))
        else:
            raise ValueError(f"Unknown test model family: {family} with {num_species} species")
        if num_species > 1:
            names = tuple(f"species-{index}" for index in range(num_species))
            region = replace(
                region,
                sampling_unit_ids=region.sampling_unit_ids * num_species,
                grid_positions=np.tile(region.grid_positions, (num_species, 1)),
                sampling_unit_kinds=None,
                species=tuple(name for name in names for _ in range(4)),
            )
            observations = tuple(
                replace(observations, species=name, values=observations.values * (index + 1))
                for index, name in enumerate(names)
            )
        return region, observations, model

    return build
