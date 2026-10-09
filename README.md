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

## Labelled pairs, transformations, and partitions

Incomplete observations need no invented genetic targets:

```python
from ilg_toolkit import ObservationPartition

observations = PairwiseObservations.from_pairs(
    [("a", "b"), ("a", "d")], [.2, .6],
    target=TargetSpec("genetic dissimilarity", units="index", transform="log1p"),
)
training = ObservationPartition(region.name, (("a", "b"),), role="training")
validation = ObservationPartition(region.name, (("a", "d"),), role="validation")
result = fit(
    region, observations, partition=training,
    validation=(region, observations), validation_partition=validation,
    config=FitConfig(epochs=10),
)
```

Explicit pair inputs require finite values and distinct, known endpoints; reversed
pairs represent the same observation and cannot be duplicated. A matrix may use
symmetric off-diagonal `NaN` entries for absent pairs. `observed_pairs` and
`observed_values` expose measured observations only. Their original measurement
precision is retained. Partition selections must be nonempty, refer to observed
pairs, and match their region and requested role. Training and validation pairs
in the same declared region must be disjoint, with or without partition arguments.

The default target transform is `identity`. Explicit `log1p` and `sqrt` transforms
require nonnegative observations. They map measurements into the encoder's
regression scale; `predict()` applies their inverse and reports `scale="original"`
with the original `TargetSpec.units`. `landscape_scores()` exposes encoder scores
before this inverse operation. A square-root inverse rejects negative transformed
predictions rather than silently squaring them. No FST linearization or target
clipping is automatic. Target kinds distinguish dissimilarity, similarity, and
relatedness; direct regression accepts nonnegative dissimilarity only.

Declare `PreparedRegion(feature_names=("elevation", "canopy"), ...)` to record
feature meanings and channel order. Prediction and validation require that same
contract; equal channel counts alone do not validate declared feature meanings.
An unnamed feature contract remains available for the simplest single-region
prepared-array workflow. `sampling_unit_kinds` optionally supplies one
`"population"` or `"individual"` entry per label (default: population). This
identity metadata reserves individual-based use without implementing or asserting
statistical validity for individual-relatedness models.
