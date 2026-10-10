"""Freeze a tiny encoder, then explicitly calibrate original and unseen regions."""

from dataclasses import replace

import jax
import numpy as np

from ilg_toolkit import (
    TrainingConfig,
    ObservationPartition,
    PairwiseObservations,
    RegionBatch,
    TargetSpec,
    fit,
    recalibrate,
)
from ilg_toolkit.models import UNetEmbeddingDistance

rng = np.random.default_rng(4)
region = RegionBatch(
    "original-catchment",
    rng.normal(size=(4, 4, 2)),
    tuple(f"population-{index}" for index in range(6)),
    np.array([[0, 0], [0, 3], [1, 1], [2, 2], [3, 0], [3, 3]]),
    feature_names=("elevation", "canopy"),
)
model = UNetEmbeddingDistance(
    2, patch_size=1, base_channels=2, embedding_dim=2, dropout=0, key=jax.random.key(4)
)
initial_scores = np.asarray(model.predict_distances(region.feature_array, region.pixel_nodes))
effects = np.array([0.2, -0.1, 0.15, -0.2, 0.25, -0.05])
noise = rng.normal(scale=0.08, size=(6, 6))
targets = 2 + 0.5 * initial_scores + effects[:, None] + effects[None, :] + (noise + noise.T) / 2
np.fill_diagonal(targets, 0)
observations = PairwiseObservations.from_matrix(
    region.sampling_unit_ids, targets, target=TargetSpec("synthetic divergence", units="index")
)
result = fit(
    region, observations, model=model, config=TrainingConfig(epochs=3, learning_rate=0.0001, seed=4)
)
calibrated = recalibrate(result.model, region, observations)
new_region = replace(region, name="new-catchment", features=region.feature_array * 1.1)
print("Label-free scores in new region:\n", calibrated.landscape_scores(new_region))
try:
    calibrated.predict(new_region)
except ValueError as error:
    print("Before regional calibration:", error)
new_observations = PairwiseObservations.from_matrix(
    new_region.sampling_unit_ids, observations.values * 2, target=observations.target
)
development = ObservationPartition(
    new_region.name, new_observations.observed_pairs, role="calibration"
)
transferred = calibrated.recalibrate(new_region, new_observations, partitions=development)
assert transferred.encoder is calibrated.encoder
assert transferred.calibrations[region.name] is calibrated.calibrations[region.name]
print("Calibrated new-region predictions:\n", transferred.predict(new_region).values)
print("Declared data roles:", set(transferred.calibrations[new_region.name].calibration_roles))
