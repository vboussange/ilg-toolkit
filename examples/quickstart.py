"""Fit a tiny synthetic landscape and predict without query genetic labels."""

import coordax as cx
import jax
import jax.numpy as jnp

from ilg_toolkit import PairwiseObservations, RegionBatch, TargetSpec, TrainingConfig, fit
from ilg_toolkit.models import UNetEmbeddingDistance

region = RegionBatch(
    "synthetic-watershed",
    cx.field(jnp.arange(32, dtype=jnp.float32).reshape(4, 4, 2) / 32, "row", "column", "feature"),
    ("a", "b", "c", "d"),
    jnp.array([[0, 0], [0, 3], [3, 0], [3, 3]]),
)
observations = PairwiseObservations.from_matrix(
    region.sampling_unit_ids,
    [[0, 0.2, 0.4, 0.6], [0.2, 0, 0.2, 0.4], [0.4, 0.2, 0, 0.2], [0.6, 0.4, 0.2, 0]],
    target=TargetSpec("synthetic dissimilarity", units="index"),
)
model = UNetEmbeddingDistance(
    2, patch_size=1, base_channels=2, embedding_dim=2, dropout=0, key=jax.random.key(3)
)
result = fit(
    region,
    observations,
    model=model,
    config=TrainingConfig(objective="direct_log1p", epochs=20, learning_rate=0.01, seed=3),
)
prediction = result.model.predict(region)
print(f"Objective: {result.history[0].training_loss:.5f} -> {result.history[-1].training_loss:.5f}")
print(f"Selected epoch: {result.selected_epoch}; target: {prediction.target}")
print(prediction.values)
