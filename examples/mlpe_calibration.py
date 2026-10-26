"""Maximum-likelihood population effects (MLPE) calibration for a frozen distance model."""

import coordax as cx
import jax
import jax.numpy as jnp
import numpy as np

from ilg_toolkit import (
    MLPECalibrationModule,
    MLPELoss,
    PairwiseObservations,
    RegionBatch,
    TargetSpec,
    calibrate,
)
from ilg_toolkit.models import UNetEmbeddingDistance

rng = np.random.default_rng(5)
region = RegionBatch(
    "synthetic-valley",
    cx.field(rng.normal(size=(4, 4, 2)).astype(np.float32), "row", "column", "feature"),
    tuple(f"population-{i}" for i in range(8)),
    np.array([[0, 0], [0, 1], [0, 3], [1, 1], [1, 3], [2, 0], [3, 1], [3, 3]]),
)
distance_model = UNetEmbeddingDistance(
    2,
    patch_size=1,
    base_channels=2,
    embedding_dim=2,
    dropout=0,
    key=jax.random.key(3),
)
distance_matrix = distance_model.predict(region).as_matrix()
left, right = jnp.triu_indices(len(region.sampling_unit_ids), 1)
distances = distance_matrix[left, right].astype(jnp.float64)
effects = rng.normal(scale=0.04, size=8)
measured = (
    1
    + 0.15 * (distances - distances.mean()) / distances.std(ddof=1)
    + effects[left]
    + effects[right]
    + rng.normal(scale=0.02, size=len(left))
)
pairs = tuple(
    (region.sampling_unit_ids[i], region.sampling_unit_ids[j])
    for i, j in zip(left, right, strict=True)
)
observations = PairwiseObservations.from_pairs(
    pairs,
    measured,
    sampling_unit_ids=region.sampling_unit_ids,
    target=TargetSpec("synthetic divergence", units="index"),
)
model = calibrate(region, observations, distance_model, loss=MLPELoss(), max_iterations=2000)
prediction = model.predict(region, pairs=observations.observed_pairs)
assert isinstance(model.calibration, MLPECalibrationModule)
calibration = model.calibration.for_region(region.name, 0)
print("Ecological distances:", distances[:4])
print(
    "Genetic predictions:",
    prediction.values[:4],
    calibration.target.units,
)
print("Signed fitted slope:", calibration.slope)
