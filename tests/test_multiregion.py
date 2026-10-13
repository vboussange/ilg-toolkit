"""Shared fitting checked at the approved public workflow seam."""

from typing import final

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ilg_toolkit import (
    PairwiseObservations,
    RegionBatch,
    ResistanceSolverConfig,
    SamplingUnits,
    TargetSpec,
    TrainingConfig,
    fit,
)
from ilg_toolkit.models import ConductanceModel, EmbeddingDistanceModel


@final
class ScalarEmbedding(EmbeddingDistanceModel):
    species: tuple[str, ...] = eqx.field(static=True, default=("species",), kw_only=True)
    weight: jax.Array
    patch_size: int = eqx.field(static=True, default=1)

    def __init__(self, weight: jax.Array):
        self.weight = weight

    def _embedding_grid(
        self, features, species_index, *, inference=True, key=None, patch_batch_size=None
    ):
        return features[..., :1] * self.weight


def regions_and_observations() -> tuple[list[RegionBatch], list[PairwiseObservations]]:
    regions = [
        RegionBatch(
            "wide",
            cx.field(np.array([[[0.0], [1.0], [2.0]]]), "row", "column", "feature"),
            {"species": SamplingUnits(("a", "b", "c"), np.array([[0, 0], [0, 1], [0, 2]]))},
            feature_names=("elevation",),
        ),
        RegionBatch(
            "short",
            cx.field(np.array([[[0.0], [2.0]]]), "row", "column", "feature"),
            {"species": SamplingUnits(("x", "y"), np.array([[0, 0], [0, 1]]))},
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
    return (regions, observations)


def test_shared_embedding_update_matches_independent_equal_region_objective():
    with jax.enable_x64():
        regions, observations = regions_and_observations()
        initial = 1.0

        def independent_objective(weight):
            predictions = [np.array([1, 4, 1]) * weight**2, np.array([4]) * weight**2]
            targets = [np.array([0.1, 0.1, 0.1]), np.array([15])]
            return np.mean(
                [
                    np.mean((np.log1p(p) - np.log1p(t)) ** 2)
                    for p, t in zip(predictions, targets, strict=True)
                ]
            )

        step = 1e-05
        gradient = (
            independent_objective(initial + step) - independent_objective(initial - step)
        ) / (2 * step)
        expected = initial - 0.01 * gradient / (abs(gradient) + 1e-08)
        result = fit(
            regions,
            {r.name: {"species": o} for r, o in zip(regions, observations, strict=True)},
            distance_model=ScalarEmbedding(jnp.asarray(initial, dtype=jnp.float64)),
            config=TrainingConfig(objective="direct_log1p", steps=1, learning_rate=0.01),
        )
        assert isinstance(result.model.distance_model, ScalarEmbedding)
        np.testing.assert_allclose(result.model.distance_model.weight, expected, rtol=1e-08)
        np.testing.assert_allclose(
            result.history[0].training_loss, independent_objective(initial), rtol=1e-08
        )
        assert result.region_names == ("short", "wide")
        for region in regions:
            assert np.isfinite(result.model.predict(region)["species"].values).all()


@final
class ScaledConductance(ConductanceModel):
    species: tuple[str, ...] = eqx.field(static=True, default=("species",), kw_only=True)
    log_scale: jax.Array
    patch_size: int = eqx.field(static=True, default=1)
    solver_config: ResistanceSolverConfig = eqx.field(
        static=True, kw_only=True, default=ResistanceSolverConfig()
    )

    def __init__(self, log_scale: jax.Array):
        self.log_scale = log_scale

    def _conductance(self, features, species_index, *, patch_batch_size=None):
        return jnp.exp(self.log_scale) * features[..., 0]


def dense_graph_resistances(surface, nodes):
    rows, columns = surface.shape
    laplacian = np.zeros((surface.size, surface.size))
    for row in range(rows):
        for column in range(columns):
            first = row * columns + column
            for r, c in ((row + 1, column), (row, column + 1)):
                if r < rows and c < columns:
                    second = r * columns + c
                    edge = np.zeros(surface.size)
                    edge[first], edge[second] = (1, -1)
                    laplacian += (surface[row, column] + surface[r, c]) / 2 * np.outer(edge, edge)
    inverse = np.linalg.pinv(laplacian, hermitian=True)
    result = np.diag(inverse)[:, None] + np.diag(inverse)[None, :] - 2 * inverse
    return result[nodes[:, None], nodes[None, :]]


def test_shared_conductance_update_matches_independent_combined_graph_objective():
    with jax.enable_x64():
        target = TargetSpec("synthetic graph divergence", units="index")
        regions = [
            RegionBatch(
                "large",
                cx.field(np.array([[[1.0], [2.0], [3.0]]]), "row", "column", "feature"),
                {"species": SamplingUnits(("a", "b", "c"), np.array([[0, 0], [0, 1], [0, 2]]))},
                feature_names=("covariate",),
            ),
            RegionBatch(
                "small",
                cx.field(np.array([[[1.0], [3.0]]]), "row", "column", "feature"),
                {"species": SamplingUnits(("x", "y"), np.array([[0, 0], [0, 1]]))},
                feature_names=("covariate",),
            ),
        ]
        observations = [
            PairwiseObservations.from_matrix(
                ("a", "b", "c"), [[0, 0.01, 0.01], [0.01, 0, 0.01], [0.01, 0.01, 0]], target=target
            ),
            PairwiseObservations.from_pairs([("x", "y")], [5], target=target),
        ]

        def independent_objective(log_scale):
            losses = []
            for region, observations_for_region in zip(regions, observations, strict=True):
                surface = np.exp(log_scale) * region.feature_array[..., 0].astype(np.float64)
                distances = dense_graph_resistances(surface, region._pixel_nodes("species"))
                pairs = np.triu_indices(len(region.sampling_units["species"].ids), 1)
                losses.append(
                    np.mean(
                        (
                            np.log1p(distances[pairs])
                            - np.log1p(observations_for_region.values[pairs])
                        )
                        ** 2
                    )
                )
            return np.mean(losses)

        initial, step = (0.2, 1e-05)
        gradient = (
            independent_objective(initial + step) - independent_objective(initial - step)
        ) / (2 * step)
        expected = initial - 0.01 * gradient / (abs(gradient) + 1e-08)
        result = fit(
            regions,
            {r.name: {"species": o} for r, o in zip(regions, observations, strict=True)},
            distance_model=ScaledConductance(jnp.asarray(initial, dtype=jnp.float64)),
            config=TrainingConfig(objective="direct_log1p", steps=1, learning_rate=0.01),
        )
        assert isinstance(result.model.distance_model, ScaledConductance)
        np.testing.assert_allclose(result.model.distance_model.log_scale, expected, rtol=1e-08)
        np.testing.assert_allclose(
            result.history[0].training_loss, independent_objective(initial), rtol=1e-08
        )
        for region in regions:
            surface = result.model.conductance_surface(region)["species"]
            np.testing.assert_allclose(
                result.model.predict(region)["species"].as_matrix(),
                dense_graph_resistances(surface, region._pixel_nodes("species")),
                rtol=1e-07,
            )


@final
class StochasticEmbedding(EmbeddingDistanceModel):
    species: tuple[str, ...] = eqx.field(static=True, default=("species",), kw_only=True)
    distance_model: ScalarEmbedding
    patch_size: int = eqx.field(static=True, default=1)

    def __init__(self, weight: jax.Array):
        self.distance_model = ScalarEmbedding(weight)

    @property
    def weight(self) -> jax.Array:
        return self.distance_model.weight

    def _embedding_grid(
        self, features, species_index, *, inference=True, key=None, patch_batch_size=None
    ):
        values = self.distance_model._embedding_grid(
            features, species_index, inference=inference, key=key, patch_batch_size=patch_batch_size
        )
        if inference:
            return values
        assert key is not None
        return values * jax.random.bernoulli(key, 0.7, values.shape) / 0.7


def test_region_order_and_validation_do_not_change_training_random_trajectory():
    from dataclasses import replace

    regions, observations = regions_and_observations()
    config = TrainingConfig(objective="direct_log1p", steps=3, seed=7)
    model = StochasticEmbedding(jnp.asarray(1.0, dtype=jnp.float32))
    baseline = fit(
        regions,
        {r.name: {"species": o} for r, o in zip(regions, observations, strict=True)},
        distance_model=model,
        config=config,
    )
    reversed_regions = {region.name: region for region in reversed(regions)}
    reversed_observations = {
        region.name: observed
        for region, observed in reversed(list(zip(regions, observations, strict=True)))
    }
    validation_region = replace(
        regions[0],
        name="validation-only",
        sampling_units={
            "species": replace(regions[0].sampling_units["species"], ids=("va", "vb", "vc"))
        },
    )
    validation_observations = PairwiseObservations.from_matrix(
        ("va", "vb", "vc"), observations[0].values, target=observations[0].target
    )
    validated = fit(
        reversed_regions,
        {name: {"species": value} for name, value in reversed_observations.items()},
        distance_model=model,
        config=config,
        validation=(
            [validation_region],
            {validation_region.name: {"species": validation_observations}},
        ),
    )
    assert [epoch.training_by_group for epoch in baseline.history] == [
        epoch.training_by_group for epoch in validated.history
    ]
    reordered = fit(
        reversed_regions,
        {name: {"species": value} for name, value in reversed_observations.items()},
        distance_model=model,
        config=config,
    )
    assert isinstance(baseline.model.distance_model, StochasticEmbedding)
    assert isinstance(reordered.model.distance_model, StochasticEmbedding)
    np.testing.assert_array_equal(
        baseline.model.distance_model.weight, reordered.model.distance_model.weight
    )


def test_validation_on_another_region_requires_declared_feature_meanings():
    from dataclasses import replace

    import pytest

    regions, observations = regions_and_observations()
    regions = [
        replace(
            region,
            features=cx.field(region.feature_array, "row", "column", "feature"),
            feature_names=None,
        )
        for region in regions
    ]
    with pytest.raises(ValueError, match="explicit feature_names"):
        fit(
            [regions[0]],
            {regions[0].name: {"species": observations[0]}},
            distance_model=ScalarEmbedding(jnp.asarray(1.0)),
            validation=([regions[1]], {regions[1].name: {"species": observations[1]}}),
            config=TrainingConfig(objective="direct_log1p", steps=0),
        )


def test_incompatible_regional_inputs_fail_before_optimization():
    from dataclasses import replace

    import pytest

    from ilg_toolkit import ObservationPartition

    regions, observations = regions_and_observations()
    model = ScalarEmbedding(jnp.asarray(1.0))
    nested = {r.name: {"species": o} for r, o in zip(regions, observations, strict=True)}
    cases = [
        ([regions[0], regions[0]], nested, None, "unique"),
        (
            [
                replace(
                    r,
                    features=cx.field(r.feature_array, "row", "column", "feature"),
                    feature_names=None,
                )
                for r in regions
            ],
            nested,
            None,
            "explicit feature_names",
        ),
        (
            [
                regions[0],
                replace(
                    regions[1],
                    features=cx.field(regions[1].feature_array, "row", "column", "feature"),
                    feature_names=("canopy",),
                ),
            ],
            nested,
            None,
            "feature contracts",
        ),
        (
            {"wrong-name": regions[0]},
            {"wrong-name": {"species": observations[0]}},
            None,
            "mapping keys",
        ),
        (regions, {regions[0].name: nested[regions[0].name]}, None, "matching region names"),
        (regions, nested, {regions[0].name: {"species": None}}, "matching region names"),
        (
            regions,
            nested,
            {
                regions[0].name: {
                    "species": ObservationPartition("wrong-region", "species", (("a", "b"),))
                },
                regions[1].name: {"species": None},
            },
            "region_name",
        ),
    ]
    for supplied_regions, supplied_observations, supplied_partitions, message in cases:
        with pytest.raises(ValueError, match=message):
            fit(
                supplied_regions,
                supplied_observations,
                distance_model=model,
                partition=supplied_partitions,
                config=TrainingConfig(objective="direct_log1p", steps=0),
            )
