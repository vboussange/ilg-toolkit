"""Share one cleaned U-Net encoder across synthetic regions of different extents."""

import jax
import numpy as np

from ilg_toolkit import FitConfig, PairwiseObservations, PreparedRegion, TargetSpec, fit
from ilg_toolkit.models import UNetEmbeddingDistance


def synthetic_region(name, width, locations):
    rows, columns = np.mgrid[:4, :width]
    features = np.stack((rows / 4, columns / 6), axis=-1).astype(np.float32)
    identifiers = tuple(f"{name}-{index}" for index in range(len(locations)))
    return PreparedRegion(
        name,
        features,
        identifiers,
        np.asarray(locations),
        feature_names=("north_covariate", "east_covariate"),
    )


regions = [
    synthetic_region("headwaters", 4, [[0, 0], [0, 3], [3, 3]]),
    synthetic_region("lowlands", 6, [[0, 0], [3, 5]]),
]
target = TargetSpec("synthetic divergence", units="index")
observations = []
for region in regions:
    endpoint_features = region.features[tuple(region.grid_positions.T)]
    dissimilarities = np.sum(
        (endpoint_features[:, None] - endpoint_features[None, :]) ** 2, axis=-1
    )
    observations.append(
        PairwiseObservations.from_matrix(region.sampling_unit_ids, dissimilarities, target=target)
    )
model = UNetEmbeddingDistance(
    2, patch_size=1, base_channels=2, embedding_dim=2, dropout=0, key=jax.random.key(4)
)
result = fit(
    regions, observations, model=model, config=FitConfig(epochs=10, learning_rate=0.001, seed=4)
)
print("Selected shared encoder at epoch", result.selected_epoch)
print("Per-region training objectives:", result.history[-1].training_by_region)
for region in regions:
    print(region.name, result.predictor.predict(region).values, result.predictor.target.units)
