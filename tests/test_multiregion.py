"""Shared fitting checked at the approved public workflow seam."""

from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ilg_toolkit import PairwiseObservations, RegionBatch, TargetSpec, TrainingConfig, fit
from ilg_toolkit.models import ConductanceModel, EmbeddingDistanceModel


@final
class ScalarEmbedding(EmbeddingDistanceModel):
    weight: jax.Array
    patch_size: int = eqx.field(static=True, default=1)

    def __init__(self, weight: jax.Array):
        self.weight = weight

    def embedding_grid(self, features, *, inference=True, key=None, patch_batch_size=None):
        return features[..., :1] * self.weight


def regions_and_observations() -> tuple[list[RegionBatch], list[PairwiseObservations]]:
    regions = [
        RegionBatch(
            "wide",
            np.array([[[0.0], [1.0], [2.0]]]),
            ("a", "b", "c"),
            np.array([[0, 0], [0, 1], [0, 2]]),
            feature_names=("elevation",),
        ),
        RegionBatch(
            "short",
            np.array([[[0.0], [2.0]]]),
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

        step = 1e-5
        gradient = (
            independent_objective(initial + step) - independent_objective(initial - step)
        ) / (2 * step)
        expected = initial - 0.01 * gradient / (abs(gradient) + 1e-8)
        result = fit(
            regions,
            observations,
            model=ScalarEmbedding(jnp.asarray(initial, dtype=jnp.float64)),
            config=TrainingConfig(epochs=1, learning_rate=0.01),
        )
        assert isinstance(result.model.encoder, ScalarEmbedding)
        np.testing.assert_allclose(result.model.encoder.weight, expected, rtol=1e-8)
        np.testing.assert_allclose(
            result.history[0].training_loss, independent_objective(initial), rtol=1e-8
        )
        assert result.region_names == ("short", "wide")
        for region in regions:
            assert np.isfinite(result.model.predict(region).values).all()


@final
class ScaledConductance(ConductanceModel):
    log_scale: jax.Array
    patch_size: int = eqx.field(static=True, default=1)

    def __init__(self, log_scale: jax.Array):
        self.log_scale = log_scale

    def conductance(self, features, *, patch_batch_size=None):
        return jnp.exp(self.log_scale) * features[..., 0]


def dense_graph_scores(surface, nodes):
    # Independent full pseudoinverse, using no production graph/solver utilities.
    rows, columns = surface.shape
    laplacian = np.zeros((surface.size, surface.size))
    for row in range(rows):
        for column in range(columns):
            first = row * columns + column
            for r, c in ((row + 1, column), (row, column + 1)):
                if r < rows and c < columns:
                    second = r * columns + c
                    edge = np.zeros(surface.size)
                    edge[first], edge[second] = 1, -1
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
                np.array([[[1.0], [2.0], [3.0]]]),
                ("a", "b", "c"),
                np.array([[0, 0], [0, 1], [0, 2]]),
                feature_names=("covariate",),
            ),
            RegionBatch(
                "small",
                np.array([[[1.0], [3.0]]]),
                ("x", "y"),
                np.array([[0, 0], [0, 1]]),
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
                scores = dense_graph_scores(surface, region.pixel_nodes)
                pairs = np.triu_indices(len(region.sampling_unit_ids), 1)
                losses.append(
                    np.mean(
                        (np.log1p(scores[pairs]) - np.log1p(observations_for_region.values[pairs]))
                        ** 2
                    )
                )
            return np.mean(losses)

        initial, step = 0.2, 1e-5
        gradient = (
            independent_objective(initial + step) - independent_objective(initial - step)
        ) / (2 * step)
        expected = initial - 0.01 * gradient / (abs(gradient) + 1e-8)
        result = fit(
            regions,
            observations,
            model=ScaledConductance(jnp.asarray(initial, dtype=jnp.float64)),
            config=TrainingConfig(epochs=1, learning_rate=0.01),
        )
        assert isinstance(result.model.encoder, ScaledConductance)
        np.testing.assert_allclose(result.model.encoder.log_scale, expected, rtol=1e-8)
        np.testing.assert_allclose(
            result.history[0].training_loss, independent_objective(initial), rtol=1e-8
        )
        for region in regions:
            surface = result.model.conductance_surface(region)
            np.testing.assert_allclose(
                result.model.predict(region).values,
                dense_graph_scores(surface, region.pixel_nodes),
                rtol=1e-7,
            )


@final
class StochasticEmbedding(EmbeddingDistanceModel):
    encoder: ScalarEmbedding
    patch_size: int = eqx.field(static=True, default=1)

    def __init__(self, weight: jax.Array):
        self.encoder = ScalarEmbedding(weight)

    @property
    def weight(self) -> jax.Array:
        return self.encoder.weight

    def embedding_grid(self, features, *, inference=True, key=None, patch_batch_size=None):
        values = self.encoder.embedding_grid(
            features, inference=inference, key=key, patch_batch_size=patch_batch_size
        )
        if inference:
            return values
        assert key is not None
        return values * jax.random.bernoulli(key, 0.7, values.shape) / 0.7


def test_region_order_and_validation_do_not_change_training_random_trajectory():
    from dataclasses import replace

    regions, observations = regions_and_observations()
    config = TrainingConfig(epochs=3, seed=7)
    model = StochasticEmbedding(jnp.asarray(1.0, dtype=jnp.float32))
    baseline = fit(regions, observations, model=model, config=config)
    reversed_regions = {region.name: region for region in reversed(regions)}
    reversed_observations = {
        region.name: observed
        for region, observed in reversed(list(zip(regions, observations, strict=True)))
    }
    validation_region = replace(regions[0], name="validation-only")
    validated = fit(
        reversed_regions,
        reversed_observations,
        model=model,
        config=config,
        validation=(validation_region, observations[0]),
    )
    assert [epoch.training_by_region for epoch in baseline.history] == [
        epoch.training_by_region for epoch in validated.history
    ]
    reordered = fit(reversed_regions, reversed_observations, model=model, config=config)
    assert isinstance(baseline.model.encoder, StochasticEmbedding)
    assert isinstance(reordered.model.encoder, StochasticEmbedding)
    np.testing.assert_array_equal(baseline.model.encoder.weight, reordered.model.encoder.weight)


def test_incompatible_regional_inputs_fail_before_optimization():
    from dataclasses import replace

    import pytest

    from ilg_toolkit import ObservationPartition

    regions, observations = regions_and_observations()
    model = ScalarEmbedding(jnp.asarray(1.0))
    cases = [
        ([regions[0], regions[0]], observations, None, "unique"),
        (
            [replace(region, feature_names=None) for region in regions],
            observations,
            None,
            "explicit feature_names",
        ),
        (
            [regions[0], replace(regions[1], feature_names=("canopy",))],
            observations,
            None,
            "feature contracts",
        ),
        ({"wrong-name": regions[0]}, {"wrong-name": observations[0]}, None, "mapping keys"),
        (regions, {regions[0].name: observations[0]}, None, "mapping keys"),
        (regions, observations, {regions[0].name: None}, "mapping keys"),
        (
            regions,
            observations,
            [ObservationPartition("wrong-region", (("a", "b"),)), None],
            "region_name",
        ),
    ]
    for supplied_regions, supplied_observations, supplied_partitions, message in cases:
        with pytest.raises(ValueError, match=message):
            fit(supplied_regions, supplied_observations, model=model, partition=supplied_partitions)


def test_validation_on_another_region_requires_declared_feature_meanings():
    from dataclasses import replace

    import pytest

    regions, observations = regions_and_observations()
    regions = [replace(region, feature_names=None) for region in regions]
    with pytest.raises(ValueError, match="explicit feature_names"):
        fit(
            regions[0],
            observations[0],
            model=ScalarEmbedding(jnp.asarray(1.0)),
            validation=(regions[1], observations[1]),
            config=TrainingConfig(epochs=0),
        )
