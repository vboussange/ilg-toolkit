"""Calibrate ecological distances separately from the frozen encoder."""

import coordax as cx
import jax
import jax.numpy as jnp
import numpy as np

from ilg_toolkit import PairwiseObservations, RegionBatch, TargetSpec, calibrate_mlpe
from ilg_toolkit.models import UNet

rng = np.random.default_rng(5)
region = RegionBatch(
    "synthetic-valley",
    cx.field(rng.normal(size=(4, 4, 2)).astype(np.float32), "row", "column", "feature"),
    tuple(f"population-{i}" for i in range(8)),
    np.array([[0, 0], [0, 1], [0, 3], [1, 1], [1, 3], [2, 0], [3, 1], [3, 3]]),
)
encoder = UNet(
    2,
    patch_size=1,
    base_channels=2,
    embedding_dim=2,
    dropout=0,
    key=jax.random.key(3),
)
raw_matrix = encoder.predict(region).as_matrix()
left, right = jnp.triu_indices(len(region.sampling_unit_ids), 1)
distances = raw_matrix[left, right].astype(jnp.float64)
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
head = calibrate_mlpe(distances, observations, region_name=region.name)
prediction = head.predict_marginal(distances, observations.observed_pairs)
print("Ecological distances:", distances[:4])
print(
    "Calibrated genetic predictions:",
    prediction.prediction.values[:4],
    head.target.units,
)
print("Signed fitted slope:", head.slope)
