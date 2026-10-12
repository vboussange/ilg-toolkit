# ILG Toolkit

Inverse landscape genetics learns how environmental measurements relate to
measured genetic relationships among sampled populations or individuals. ILG
Toolkit fits that relationship and predicts at new locations from their
landscape features, without requiring genetic measurements there.

The conductance workflow connects three quantities:

- **Landscape features** are prepared environmental measurements such as
  elevation or habitat. A **sampling unit** is an identified population or
  individual at a location. **Pairwise genetic observations** measure the genetic
  relationship between two sampling units, with an explicitly declared meaning
  and units.
- **Conductance** describes ease of movement through the landscape. A model
  learns conductance from features, then calculates **effective resistance**:
  separation between locations through the resulting movement graph. Resistance
  is a **landscape score**, before relating it to genetic measurements.
- **Genetic calibration** relates landscape scores to the declared genetic
  target scale. A **calibrated model** keeps the landscape model and this
  relationship together for prediction.

[JAX](https://docs.jax.dev/) supplies numerical arrays and differentiation.
[Coordax](https://coordax.readthedocs.io/) wraps arrays with named dimensions:
`row` and `column` locate raster cells; `feature` identifies the environmental
measurements at each cell. Raster preparation, geographic coordinate conversion
and the biological interpretation of the observations belong to the caller.

## Install and run

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and use
Python 3.12 or newer. From this checkout:

```bash
uv sync
uv run python examples/quickstart.py
```

Optional extras add geospatial preparation (`uv sync --extra geo`), algebraic
multigrid resistance preconditioning (`--extra amg`), or CUDA JAX (`--extra gpu`).
The ordinary CPU workflow needs none of these extras or private research data.

## Synthetic conductance fit

This complete example also lives in [examples/quickstart.py](examples/quickstart.py).
It creates two environmental features on an 8-by-8 raster, three labelled
populations, and a symmetric matrix of invented genetic dissimilarities.
Resistance calculations require float64, so precision is set before creating inputs.

```python
import coordax as cx
import jax
import jax.numpy as jnp

from ilg_toolkit import PairwiseObservations, RegionBatch, TargetSpec, TrainingConfig, fit
from ilg_toolkit.models import ResNet9Conductance

# Set precision before creating inputs; resistance calculations require float64.
jax.config.update("jax_enable_x64", True)

region = RegionBatch(
    name="synthetic-valley",
    features=cx.field(
        jax.random.normal(jax.random.key(4), (8, 8, 2), dtype=jnp.float32),
        "row",
        "column",
        cx.LabeledAxis("feature", ["elevation", "habitat"]),
    ),
    sampling_unit_ids=("south", "north", "east"),
    grid_positions=[[0, 0], [0, 7], [7, 7]],
)
observations = PairwiseObservations.from_matrix(
    region.sampling_unit_ids,
    [[0, 0.375, 0.5], [0.375, 0, 0.375], [0.5, 0.375, 0]],
    target=TargetSpec("synthetic genetic dissimilarity", units="index", transform="identity"),
)
encoder = ResNet9Conductance(
    in_channels=2, patch_size=4, min_conductance=1e-6, key=jax.random.key(6)
)
result = fit(
    region,
    observations,
    model=encoder,
    config=TrainingConfig(objective="direct_log1p", epochs=4, learning_rate=0.001),
)
print("Conductance surface:\n", result.model.conductance_surface(region))
print("Resistance landscape scores:\n", result.model.landscape_scores(region))
print("Genetic predictions:\n", result.model.predict(region).values)
```

`patch_size=4` turns this raster into a 2-by-2 conductance graph. Locations are
integer `(row, column)` cell positions, and matrix rows/columns follow the given
sampling-unit labels. Query regions need only features and locations with the
same feature meanings and order. `predict()` returns original target units;
`predict_pairs(region, [("south", "east")])` requests selected pairs.

Both the encoder and objective are required choices. Here `direct_log1p` minimizes
mean squared differences after applying `log1p` to nonnegative scores and targets.
This loss does not change the declared `identity` target transform. Conductance,
resistance and genetic prediction remain separate outputs; a good genetic fit
alone does not identify a unique biological movement surface.

Inputs follow ordinary JAX conversion: compatible lists and NumPy arrays,
including NumPy-backed Coordax fields, are accepted. Supplied JAX floating
observation dtypes are preserved; other inputs follow the configured JAX
precision. Enable x64 **before** creating or converting measurements that need
float64, as above or with `JAX_ENABLE_X64=true`. Enabling it later cannot recover
rounded information. Resistance and population-effect calculations require
float64; the toolkit does not change global precision.

## Objectives and larger studies

**Maximum-likelihood population-effects (MLPE)** models account for dependence
between genetic pairs that share a sampled population. Selecting
`TrainingConfig(objective="mlpe")` fits the landscape model jointly with a
regional genetic calibration and population/residual variances. Calibration can
also be fitted separately with a frozen landscape model. MLPE currently supports
population dissimilarities; relatedness needs its own statistical model.

The [statistical guide](docs/statistics.md) explains both objectives, target
transforms and regional weighting. Missing observations can be supplied as
labelled pairs or symmetric off-diagonal `NaN` matrix entries. Explicit training,
validation, calibration, support and query partitions record which measurements
were accessed. Validation selects the best model while optimization follows the
fixed requested budget.

Use these focused examples and guides for other workflows:

- [Conductance settings](examples/conductance.py): graph solver tolerances and
  positive conductance. Larger rasters can use `patch_batch_size` to bound encoder
  activation memory.
- [Shared regions](examples/shared_regions.py): one explicitly constructed
  U-Net embedding model, which measures squared separation between learned
  environmental representations, with separate regional calibrations.
- [Regional transfer](examples/region_transfer.py) and
  [population-effect conditioning](docs/mlpe-conditioning.md): frozen-model
  recalibration, known effects and declared support measurements.
- [Fold ensembles](examples/ensemble.py): independently initialized models
  receive their declared seeds directly and average calibrated predictions on
  the original target scale. Member failures remain visible. Individual
  checkpoints are managed by the caller's script.
- [Out-of-fold evaluation](docs/evaluation.md): average only eligible members,
  score each unique pair once, and report coverage. Query targets are read only
  for scoring, after prediction.
- [Isolation by distance](examples/ibd.py): a geographic-separation reference
  independent of landscape resistance.

Run an example with `JAX_ENABLE_X64=true uv run python examples/<name>.py`.

## Save and continue

The fitted `CalibratedModel` and its `TrainingState` are Equinox modules, meaning
that learned numerical fields participate in native tree operations and leaf
serialization. Save a model and load it with an encoder constructed using the
**same full configuration**:

```python
from ilg_toolkit import load_model, save_model

save_model("model.ilg", result.model)
encoder_template = ResNet9Conductance(
    in_channels=2, patch_size=4, min_conductance=1e-6, key=jax.random.key(0)
)
restored = load_model("model.ilg", encoder=encoder_template)
print(restored.predict(region).values)
```

Initial template weights may differ; saved leaves replace them. Keeping static
settings such as patch size, conductance floor and patch batching identical is
the caller's responsibility. Custom compatible encoders use this same path.
Models retain target/feature contracts, regional calibration and observation-use
provenance without embedding the original measurements.

Individual checkpoints retain the latest optimizer and random state separately
from the selected best model. Continue with the same data and partitions, and
optionally increase the total epoch budget. Same-shape measurement changes are
not detected automatically. See [saving and continuation](docs/persistence.md)
for caller-template loading, atomic replacement and complete examples.

## Development and license

```bash
uv sync --extra dev
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true
uv run --no-sync ruff format --check .
uv run --no-sync ruff check .
uv run --no-sync pyright
uv run --no-sync pytest
for example in examples/*.py; do uv run --no-sync python "$example"; done
uv run --no-sync python -m experiments.wishart
uv run --no-sync python -m build
```

The installed package is independent of study repositories. Source-only research
references are described in [the Wishart guide](docs/statistics/wishart.md).

Developed by Victor Boussange and contributors. Licensed under the
[Apache License 2.0](LICENSE).
