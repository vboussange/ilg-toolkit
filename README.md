# ILG Toolkit

**Inverse landscape genetics in JAX.**

[![Checks](https://github.com/vboussange/ilg-toolkit/actions/workflows/checks.yml/badge.svg)](https://github.com/vboussange/ilg-toolkit/actions/workflows/checks.yml)

<p align="center">
  <img src="docs/assets/ilg-overview.webp" width="420" alt="From bottom to top: a raw Earth-observation raster, a patch encoder, a conductance grid graph, the effective resistance R_ij between two sampled locations, and the genetic dissimilarity D_ij between them.">
</p>

**Overview, from bottom to top.** A patch encoder maps each patch of a raw
Earth-observation raster to a positive conductance, and these conductances form a
conductance grid graph. The effective resistance $R_{ij}$ between the vertices of two
sampling units $i$ and $j$ is their ecological distance. Genetic calibration relates
it to the genetic dissimilarity $D_{ij}$ observed between them, which reflects gene
flow.

## Introduction

Landscape genetics studies how landscapes shape gene flow: the exchange of genes
between populations through dispersal and mating. Roads, rivers, settlements and
unsuitable habitat hinder movement, and populations that exchange few migrants
drift apart genetically. The genetic dissimilarity between two sampled populations
or individuals therefore records how hard it has been, over generations, to move
between them through the landscape.

Inverse landscape genetics (ILG) infers this effect of the landscape from genetic
data. ILG Toolkit casts it as **metric learning**: it learns a distance between
landscape locations, parameterised through a conductance grid graph whose
conductances a neural network predicts from landscape features. Supervision is
sparse and pairwise: genetic observations exist only between a limited set of
sampled locations. Gradients of the genetic loss flow through the effective
resistance computation back to the network, and the fitted model predicts genetic
dissimilarity at unsampled locations from landscape features alone. A good genetic
fit does not, however, identify a unique movement surface: different conductance
surfaces can induce nearly the same distances between the sampled locations.

## Concepts

**Landscape features.** A raster $X \in \mathbb{R}^{H \times W \times C}$ with one
channel per environmental covariate, such as elevation or land cover. Features are a
[Coordax](https://coordax.readthedocs.io/) field with the named dimensions `row`,
`column` and `feature`. Labelling the `feature` axis records covariate names and
order, which every query region must then match. Preparing rasters (reprojection,
alignment, normalisation) is the caller's responsibility.

**Conductance grid graph.** `patch_size` $p$ partitions the raster into
non-overlapping $p \times p$ patches. A **patch encoder**, such as
`ResNet9Conductance`, maps each patch to a conductance strictly greater than
`min_conductance`. The conductance grid graph has one vertex per patch, forming an
$(H/p) \times (W/p)$ lattice, and edges between 4-neighbouring vertices, weighted
from the conductances at their two ends. High conductance means easy movement. Each
sampling unit sits at the vertex of the patch containing its cell.

**Effective resistance.** Read the graph as an electrical network in which edge
$(u, v)$ is a conductor of conductance $w_{uv}$. Inject a unit current at vertex $i$
and withdraw it at vertex $j$. The vertex potentials $\phi$ then satisfy Kirchhoff's
current law,

$$\sum_{u \sim v} w_{uv}\,(\phi_v - \phi_u) = \mathbf{1}[v = i] - \mathbf{1}[v = j] \quad \text{for every vertex } v,$$

and the effective resistance $R_{ij} = \phi_i - \phi_j$ is the resulting potential
drop. Unlike a least-cost path, $R_{ij}$ aggregates all paths: a parallel corridor
lowers it, as a parallel resistor does. It is a metric on the vertices, and it is
proportional to the commute time of a random walk on the graph between $i$ and $j$.
Computing it is a sparse linear solve; [JAXScape](https://github.com/vboussange/jaxscape)
enables differentiating through that solve, so that gradients reach the encoder.

**Ecological distance.** A distance model's pairwise separation between two
sampling units, derived from the landscape before any genetic calibration. It is
not a genetic prediction and is not in genetic units. For a conductance model it is
$R_{ij}$; `model.distance_model.predict(region)` returns these values for query pairs.

**Distance model.** Any model that maps landscape features and sampling-unit
locations to ecological distances without access to genetic data. The conductance
model above is the main instance. An alternative, `UNet`, embeds
each location and uses the squared Euclidean distance between embeddings (see the
[shared-regions example](examples/shared_regions.py)). Custom models subclass
`AbstractDistanceModel`.

The model hierarchy is:

```text
AbstractDistanceModel: backbone(features), ecological_distances(latent, pairs, species_index)
├── ConductanceModel: conductance(latent, species_index)
│   ├── LogLinearConductance: log-linear conductance from patch-averaged covariates
│   └── ResNet9Conductance: patch-based conductance surfaces
└── EmbeddingModel: embedding(latent, species_index)
    └── UNet: landscape embeddings
```

Concrete models default to a static `num_species` of one. `patch_microbatch_size` sets
the number of raster patches processed together without changing predictions.

`LogLinearConductance` learns one coefficient per feature and one intercept per
species index, pooled across regions. Its conductance is `min_conductance + exp(predictor)`,
capped at `max_conductance`; defaults are 0.01 and 1000. Coefficients follow the
raster feature order and start at zero, as do intercepts. The model uses
`patch_size=32` by default and accepts the same `fit` and `predict` workflow.

**Sampling units and pairwise genetic observations.** A sampling unit is an
identified population or individual located at a raster cell. A pairwise genetic
observation is a measurement between two sampling units, such as a genetic
dissimilarity between populations. Its meaning, kind (dissimilarity, similarity or
relatedness), units and transform are declared in a `TargetSpec` rather than
inferred from the data. Observations may be missing for some pairs.

**Genetic calibration.** A map from ecological distances to predictions
$\hat D_{ij}$ of the declared genetic target. Direct calibration applies the
inverse of the declared target transform. MLPE calibration fits a linear mixed model
with population effects for each region, so one distance model can serve several
regions. A **landscape genetics model** (`LandscapeGeneticsModel`) is a distance
model composed with a calibration: `model.distance_model.predict(region)` returns
ecological distances, and `model.predict(region)` returns genetic predictions in
the target's original units.

**Fitting.** `fit` trains the distance model end to end. It computes ecological
distances for the observed pairs, compares them with the genetic observations
through a training objective, and updates the distance model's parameters with
Adam for a fixed number of epochs. Optional validation pairs select the best epoch
without changing the optimisation. The distance model and the objective are both
explicit choices.

## Install

```bash
uv add "ilg-toolkit @ git+https://github.com/vboussange/ilg-toolkit@v0.0.3"
```

ILG Toolkit requires Python 3.12 or newer; `pip install` accepts the same
requirement string. Optional extras are installed as `ilg-toolkit[amg] @ git+...`:

- `geo`: GeoPandas and Rasterio, for preparing rasters and locations from
  geospatial files;
- `amg`: algebraic multigrid preconditioning, which speeds up resistance solves on
  large graphs;
- `gpu`: CUDA-enabled JAX on Linux x86-64.

The quickstart needs only the base install and runs on CPU.

## Quickstart

Fit a conductance model to three synthetic populations, inspect what it learned,
then save and reload it. This is [examples/quickstart.py](examples/quickstart.py).

```python
"""Fit a conductance model to synthetic genetic dissimilarities, then save and reload it."""

from typing import cast

import coordax as cx
import jax
import jax.numpy as jnp

from ilg_toolkit import (
    PairwiseObservations,
    RegionBatch,
    TargetSpec,
    TrainingConfig,
    fit,
    load_model,
    save_model,
)
from ilg_toolkit.models import ConductanceModel, ResNet9Conductance

# Effective resistance needs float64: enable x64 before creating any inputs.
jax.config.update("jax_enable_x64", True)

# An 8x8 raster with two covariates, and three sampling units at (row, column) cells.
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
# Pairwise genetic observations, with their meaning, units and transform declared.
observations = PairwiseObservations.from_matrix(
    region.sampling_unit_ids,
    [[0, 0.375, 0.5], [0.375, 0, 0.375], [0.5, 0.375, 0]],
    target=TargetSpec("synthetic genetic dissimilarity", units="index", transform="identity"),
)
# patch_size=4 coarsens the 8x8 raster into a 2x2 conductance grid graph.
distance_model = ResNet9Conductance(
    in_channels=2, patch_size=4, min_conductance=1e-6, key=jax.random.key(6)
)
result = fit(
    region,
    observations,
    distance_model=distance_model,
    config=TrainingConfig(objective="direct_log1p", epochs=4, learning_rate=0.001),
)
model = result.model  # The selected landscape genetics model.

print("Conductance surface:\n", cast(ConductanceModel, model.distance_model).conductance_surface(region))
print("Ecological distances (effective resistance):\n", model.distance_model.predict(region).as_matrix())
# predict returns one calibrated value per distinct pair, in the target's units;
# as_matrix lays them out symmetrically in sampling-unit order.
prediction = model.predict(region)
print("Genetic predictions:", prediction.pairs, prediction.values)
print("Genetic prediction matrix:\n", prediction.as_matrix())

# Loading needs a template distance model with the same configuration.
save_model("model.ilg", model)
template = ResNet9Conductance(
    in_channels=2, patch_size=4, min_conductance=1e-6, key=jax.random.key(0)
)
restored = load_model("model.ilg", distance_model=template)
print("Restored predictions:\n", restored.predict(region).as_matrix())
```

## Examples

- [quickstart.py](examples/quickstart.py): the quickstart above.
- [conductance.py](examples/conductance.py): a conductance model with an explicit
  resistance-solver tolerance; on large rasters, `patch_microbatch_size` bounds encoder
  memory.
- [shared_regions.py](examples/shared_regions.py): one U-Net embedding distance
  model shared across regions of different extents.
- [mlpe_calibration.py](examples/mlpe_calibration.py): MLPE calibration fitted to
  the ecological distances of a frozen distance model.
- [region_transfer.py](examples/region_transfer.py): transfer a fitted distance
  model to a new region by calibrating it there.
- [ensemble.py](examples/ensemble.py): population-fold ensembles, out-of-fold
  evaluation, support conditioning and checkpoint continuation.
- [ibd.py](examples/ibd.py): an isolation-by-distance baseline from geographic
  separation alone.

From a clone, run one with `JAX_ENABLE_X64=true uv run python examples/<name>.py`.

## Documentation

**Training objectives** are selected with `TrainingConfig(objective=...)`. The
[statistics guide](docs/statistics.md) derives them and also covers calibration
modules, recalibration and the layout of predictions.

- `direct_log1p`: mean squared error between `log1p` of the ecological distances and
  of the transformed targets; prediction applies the inverse target transform.
- `mlpe`: maximum-likelihood population effects. It fits the distance model jointly
  with a regional linear mixed model in which pairs sharing a population are
  correlated. It supports population dissimilarities.
- Wishart: experimental, not yet trainable: a fixed-parameter scoring reference in
  the source checkout; fitting tracked in
  [#17](https://github.com/vboussange/ilg-toolkit/issues/17). See the
  [Wishart guide](docs/statistics/wishart.md).

**Guides**:

- [Evaluation and out-of-fold ensembles](docs/evaluation.md): deployment versus
  out-of-fold prediction, holdout rules, coverage and metrics.
- [MLPE conditioning](docs/mlpe-conditioning.md) and
  [regional transfer](examples/region_transfer.py): known population effects,
  declared support observations and calibration in new regions.
- [Saving and continuation](docs/persistence.md): model artifacts, training
  checkpoints and numerical precision.
- [MLPE performance](docs/performance/mlpe.md): exact population-sized arithmetic
  and its numerical limits.

## Development

```bash
git clone https://github.com/vboussange/ilg-toolkit.git
cd ilg-toolkit
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

To release, update `project.version` in `pyproject.toml` and merge into `main`. The
Release workflow tests the built wheel and source archive, then tags `v<version>`
and publishes a GitHub release; nothing is published to PyPI. Retry a failed
release from Actions → Release → Run workflow on `main`.

## License

Developed by Victor Boussange and contributors. Licensed under the
[Apache License 2.0](LICENSE).
