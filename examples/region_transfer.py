"""Maximum-likelihood population effects (MLPE) calibration for a new region."""

from dataclasses import replace

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ilg_toolkit import (
    Log1pLoss,
    MLPECalibrationModule,
    MLPELoss,
    PairwiseObservations,
    RegionBatch,
    TargetSpec,
    TrainingConfig,
    calibrate,
    train,
)
from ilg_toolkit.models import UNetEmbeddingDistance

rng = np.random.default_rng(4)
region = RegionBatch(
    "original-catchment",
    cx.field(rng.normal(size=(4, 4, 2)).astype(np.float32), "row", "column", "feature"),
    tuple(f"population-{index}" for index in range(6)),
    np.array([[0, 0], [0, 3], [1, 1], [2, 2], [3, 0], [3, 3]]),
    feature_names=("elevation", "canopy"),
)
model = UNetEmbeddingDistance(
    2, patch_size=1, base_channels=2, embedding_dim=2, dropout=0, key=jax.random.key(4)
)
initial_distances = model.predict(region).as_matrix()
effects = np.array([0.2, -0.1, 0.15, -0.2, 0.25, -0.05])
noise = rng.normal(scale=0.08, size=(6, 6))
targets = 2 + 0.5 * initial_distances + effects[:, None] + effects[None, :] + (noise + noise.T) / 2
targets = targets.at[jnp.diag_indices(len(region.sampling_unit_ids))].set(0)
observations = PairwiseObservations.from_matrix(
    region.sampling_unit_ids, targets, target=TargetSpec("synthetic divergence", units="index")
)
result = train(
    region,
    observations,
    distance_model=model,
    config=TrainingConfig(loss=Log1pLoss(), steps=3, learning_rate=0.0001, seed=4),
)
calibrated = calibrate(
    region, observations, result.selected_model.distance_model, loss=MLPELoss(), max_iterations=2000
)
new_region = replace(
    region,
    name="new-catchment",
    features=cx.field(region.feature_array * 1.1, "row", "column", "feature"),
)
print(
    "Label-free ecological distances in new region:\n",
    calibrated.distance_model.predict(new_region).as_matrix(),
)
try:
    calibrated.predict(new_region)
except ValueError as error:
    print("Before MLPE calibration:", error)
new_observations = PairwiseObservations.from_matrix(
    new_region.sampling_unit_ids, observations.values * 2, target=observations.target
)
transferred = calibrate(
    new_region, new_observations, calibrated.distance_model, loss=MLPELoss(), max_iterations=2000
)
assert eqx.tree_equal(transferred.distance_model, calibrated.distance_model)
assert isinstance(calibrated.calibration, MLPECalibrationModule)
assert isinstance(transferred.calibration, MLPECalibrationModule)
assert tuple(transferred.calibration.regions) == ((new_region.name, 0),)
print("Calibrated new-region predictions:\n", transferred.predict(new_region).as_matrix())
assert isinstance(transferred.calibration, MLPECalibrationModule)
print("Calibration pairs:", transferred.calibration.regions[new_region.name, 0].calibration_pairs)
