"""Native domain-tree persistence and continuation through the public workflow."""

from dataclasses import replace
from typing import final

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ilg_toolkit import (
    ObservationPartition,
    PairwiseObservations,
    RegionBatch,
    TargetSpec,
    TrainingConfig,
    fit,
)
from ilg_toolkit.models import EmbeddingDistanceModel


@final
class ScaledEmbedding(EmbeddingDistanceModel):
    log_weights: jax.Array
    patch_size: int = eqx.field(static=True, default=1)

    def __init__(self, log_weights: jax.Array):
        self.log_weights = log_weights

    def embedding_grid(self, features, *, inference=True, key=None, patch_batch_size=None):
        return features * jnp.exp(self.log_weights)


def native_problem():
    features = jnp.asarray([[[0.0, 0.0], [0.2, 0.7], [0.8, -0.1], [1.1, 1.3]]])
    ids = ("a", "b", "c", "d")
    region = RegionBatch(
        "alpine",
        cx.field(features, "row", "column", "feature"),
        ids,
        jnp.asarray([[0, 0], [0, 1], [0, 2], [0, 3]]),
        feature_names=("elevation", "canopy"),
    )
    observations = PairwiseObservations.from_pairs(
        (("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"), ("b", "d"), ("c", "d")),
        jnp.asarray([0.4, 0.7, 1.4, 0.6, 0.9, 0.8]),
        sampling_unit_ids=ids,
        target=TargetSpec("synthetic divergence", units="index"),
    )
    return region, observations


@jax.enable_x64()
def test_native_direct_model_and_training_state_round_trip_and_continue(tmp_path):
    region, observations = native_problem()
    config = TrainingConfig(objective="direct_log1p", epochs=1, seed=17)
    encoder = ScaledEmbedding(jnp.zeros(2))
    validation = (
        replace(region, name="validation"),
        PairwiseObservations.from_matrix(
            region.sampling_unit_ids,
            encoder.predict_distances(region.feature_array, region.pixel_nodes),
            target=observations.target,
        ),
    )
    partial = fit(region, observations, model=encoder, config=config, validation=validation)
    template = fit(
        region,
        observations,
        model=ScaledEmbedding(jnp.ones(2)),
        config=config,
        validation=validation,
    )

    assert partial.history[0].training_loss != template.history[0].training_loss

    model_path, state_path = tmp_path / "model.eqx", tmp_path / "state.eqx"
    eqx.tree_serialise_leaves(model_path, partial.model)
    eqx.tree_serialise_leaves(state_path, partial)
    model = eqx.tree_deserialise_leaves(model_path, template.model)
    state = eqx.tree_deserialise_leaves(state_path, template)
    np.testing.assert_array_equal(
        model.predict(region).values, partial.model.predict(region).values
    )
    assert state.history == partial.history
    assert state.selected_epoch == partial.selected_epoch

    continued = fit(
        region, observations, state=state, config=replace(config, epochs=2), validation=validation
    )
    uninterrupted = fit(
        region,
        observations,
        model=ScaledEmbedding(jnp.zeros(2)),
        config=replace(config, epochs=2),
        validation=validation,
    )
    assert continued.history == uninterrupted.history
    np.testing.assert_array_equal(
        continued.model.predict(region).values, uninterrupted.model.predict(region).values
    )

    assert continued.selected_epoch == 0
    assert not np.array_equal(
        continued.latest_model.predict(region).values, continued.best_model.predict(region).values
    )
    for actual, expected in zip(
        jax.tree.leaves((continued.optimizer_state, continued.rng_key)),
        jax.tree.leaves((uninterrupted.optimizer_state, uninterrupted.rng_key)),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)


@jax.enable_x64()
def test_native_mlpe_state_preserves_posterior_support_predictions_and_continuation(tmp_path):
    region, observations = native_problem()
    partition = ObservationPartition(region.name, observations.observed_pairs[:-2], role="training")
    config = TrainingConfig(objective="mlpe", epochs=1, seed=23)
    partial = fit(
        region,
        observations,
        model=ScaledEmbedding(jnp.zeros(2)),
        config=config,
        partition=partition,
    )
    template = fit(
        region,
        observations,
        model=ScaledEmbedding(jnp.asarray([0.8, -0.4])),
        config=config,
        partition=partition,
    )

    path = tmp_path / "mlpe-state.eqx"
    eqx.tree_serialise_leaves(path, partial)
    state = eqx.tree_deserialise_leaves(path, template)
    query = observations.observed_pairs[-1:]
    expected = partial.model.predict_known_effects(region, query)
    actual = state.best_model.predict_known_effects(region, query)
    np.testing.assert_array_equal(actual.values, expected.values)
    np.testing.assert_array_equal(actual.model_variance, expected.model_variance)
    assert actual.provenance == expected.provenance
    assert state.history == partial.history

    support_pair = observations.observed_pairs[-2:-1]
    support = PairwiseObservations.from_pairs(
        support_pair,
        observations.observed_values[-2:-1],
        target=observations.target,
        sampling_unit_ids=region.sampling_unit_ids,
    )
    support_partition = ObservationPartition(region.name, support_pair, role="support")
    expected = partial.model.predict_with_support(
        region, query, support, support_partition=support_partition
    )
    actual = state.best_model.predict_with_support(
        region, query, support, support_partition=support_partition
    )
    np.testing.assert_array_equal(actual.values, expected.values)
    np.testing.assert_array_equal(actual.model_variance, expected.model_variance)

    continued = fit(
        region, observations, state=state, config=replace(config, epochs=2), partition=partition
    )
    uninterrupted = fit(
        region,
        observations,
        model=ScaledEmbedding(jnp.zeros(2)),
        config=replace(config, epochs=2),
        partition=partition,
    )
    assert continued.history == uninterrupted.history
    np.testing.assert_array_equal(
        continued.model.predict(region).values, uninterrupted.model.predict(region).values
    )
