# ILG Toolkit

An independent Python toolkit for inverse landscape genetics. Fit landscape
encoders from prepared arrays and labelled pairwise genetic observations, then
predict at query sampling-unit locations without their genetic observations.

## Install

Python 3.12 or newer is required. From this checkout:

```bash
pip install .
# Optional geospatial preparation libraries: pip install '.[geo]'
# Optional CUDA JAX wheels: pip install '.[gpu]'
# Tests and formatting: pip install '.[dev]'
```

The minimal workflow uses NumPy, JAX, Equinox, and Optax. It has no paper-data,
benchmark, geospatial acquisition, or research-checkout dependency.

## Synthetic quickstart

Run `python examples/quickstart.py` after installing. The complete in-memory
workflow is:

```python
import numpy as np
from ilg_toolkit import FitConfig, PairwiseObservations, PreparedRegion, TargetSpec, fit

region = PreparedRegion(
    name="my-catchment",
    features=np.arange(32, dtype=np.float32).reshape(4, 4, 2) / 32,
    sampling_unit_ids=("a", "b", "c", "d"),
    grid_positions=np.array([[0, 0], [0, 3], [3, 0], [3, 3]]),
)
observations = PairwiseObservations.from_matrix(
    region.sampling_unit_ids,
    [[0, .2, .4, .6], [.2, 0, .2, .4], [.4, .2, 0, .2], [.6, .4, .2, 0]],
    target=TargetSpec("genetic dissimilarity", units="index"),
)
result = fit(region, observations, config=FitConfig(epochs=30, learning_rate=.01))
prediction = result.predictor.predict(region)
print(prediction.values, prediction.target)
scores = result.predictor.landscape_scores(region)
```

Features have shape `(rows, columns, channels)` and must already be prepared;
locations are explicit integer `(row, column)` positions. Labels declare the
matrix row/column identity and are aligned to the region before training.
Fitting performs no coordinate conversion, normalization, FST conversion, or
observation clipping. Query regions contain features and locations only.

The initial model is the cleaned architecture's U-Net embedding-distance family.
Its landscape scores are squared Euclidean distances between learned patch
embeddings. Direct regression trains these distances with nonnegative
`log1p`-MSE and returns them on the declared original target scale: `log1p` is a
loss transform, not a target transform. Signed relatedness is unsupported by
this objective. Direct regression requires no MLPE calibration head.

Pass a configured `ilg_toolkit.models.UNetEmbeddingDistance` as `model=` to
choose patch size, embedding dimensions, U-Net width, and dropout explicitly.
For example, `patch_size=1, base_channels=2, embedding_dim=2, dropout=0` is useful
for tiny synthetic problems. Raster dimensions must be divisible by patch size.

Without validation, the final fixed-budget encoder is returned. With
`validation=(prepared_validation_region, validation_observations)`, the lowest
validation log1p-MSE selects initialization or an updated encoder. Validation
targets never enter optimization, and test data is not required. The result
records every epoch's inference-mode training/validation objective and the
selected epoch. The optimizer uses fixed-rate Adam; no validation-dependent
schedule or early stopping is implicit.

## Development

```bash
pytest
ruff check .
```

Tests exercise public fit/predict behavior on synthetic arrays and build a wheel
to run outside the source tree. Extracted components and their license are listed
in `NOTICE`; study-specific code remains in its original repositories.
