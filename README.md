# ILG Toolkit

**Inverse landscape genetics in JAX.**

[![Checks](https://github.com/vboussange/ilg-toolkit/actions/workflows/checks.yml/badge.svg)](https://github.com/vboussange/ilg-toolkit/actions/workflows/checks.yml)
[![Documentation](https://github.com/vboussange/ilg-toolkit/actions/workflows/docs.yml/badge.svg)](https://github.com/vboussange/ilg-toolkit/actions/workflows/docs.yml)

ILG Toolkit learns how landscape features relate to pairwise genetic observations
among populations or individuals. Start with one species: supply a raster, located
sampling units, and genetic observations to `train`, then request original-scale
genetic predictions with `predict`. Multispecies support is experimental.

<p align="center">
  <img src="docs/assets/ilg-overview.webp" width="420" alt="Landscape features become conductance, effective resistance, and genetic predictions.">
</p>

A **distance model** derives ecological distances from the landscape. A
**landscape genetics model** combines that relationship with genetic calibration.
`train` jointly updates distance parameters and loss-specific statistical
parameters; `calibrate` fits a genetic relationship for a frozen distance model.
Different conductance surfaces can produce similar distances, so interpret fitted
surfaces alongside ecological knowledge and a stated evaluation design.

## Install

Python 3.12 or newer is required. This documentation and its notebooks describe
the development API. Install from the development checkout:

```bash
git clone https://github.com/vboussange/ilg-toolkit.git
cd ilg-toolkit
uv sync --extra dev --extra docs
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true
```

Optional extras include `amg` for resistance preconditioning, `geo` for geospatial
input preparation, and `gpu` for CUDA-enabled JAX on Linux x86-64. Prepare aligned,
normalized rasters before constructing a `RegionBatch`. Enable float64 before
creating resistance or maximum-likelihood population effects (MLPE) inputs.

## Learn through notebooks

The [documentation site](https://vboussange.github.io/ilg-toolkit/) renders the
same five notebooks that CI executes. Each is self-contained, downloadable, and
uses small deterministic CPU examples:

1. [Quickstart](examples/quickstart.ipynb): first analysis, save/load, and
   checkpoint continuation.
2. [Isolation by distance](examples/ibd.ipynb): geographic reference, units, and
   interpretation.
3. [MLPE workflow](examples/mlpe_workflow.ipynb): ResNet9 conductance, pooled regions,
   repeated population splits, fresh calibration, conditioning, and regional transfer.
4. [Ensemble](examples/ensemble.ipynb): member training, selection, checkpoints,
   ordered predictions, and conductance summaries.
5. [Multispecies](examples/multispecies.ipynb): experimental species labeling,
   pooled feature learning, species-specific parameters, and calibration.

Public docstrings are the authoritative API reference for model details and
numerical requirements. See the [statistics guide](docs/statistics.md),
[custom-model guide](docs/models.md), and [canonical glossary](GLOSSARY.md).
The built-in concrete distance models are `LogLinearConductance`,
`ResNet9Conductance`, and `UNetEmbeddingDistance`.

## Documentation

| Area | Pages |
| --- | --- |
| Getting started | [README](README.md), [site overview](docs/index.md), [notebook index](docs/examples/index.md) |
| Concepts and extensions | [Statistics](docs/statistics.md), [custom distance models](docs/models.md), [glossary](GLOSSARY.md) |
| Core API | [Inputs and outputs](docs/api/data.md), [distance models](docs/api/models.md), [training and calibration](docs/api/training.md), [prediction](docs/api/prediction.md) |
| Supporting API | [Ensembles](docs/api/ensemble.md), [persistence](docs/api/persistence.md), [advanced MLPE](docs/api/mlpe.md), [isolation by distance](docs/api/ibd.md) |
| Contributing | [Development](docs/development.md) |

Contributor instructions: [repository conventions](AGENTS.md),
[domain documentation](docs/agents/domain.md), [issue tracker](docs/agents/issue-tracker.md),
and [triage labels](docs/agents/triage-labels.md). These instructions are excluded
from the public site.

## Development

```bash
uv run --no-sync ruff format --check .
uv run --no-sync ruff check .
uv run --no-sync pyright
uv run --no-sync pytest
uv run --no-sync pytest experiments/wishart/test_diagnostic.py
uv run --no-sync python -m experiments.wishart
uv run --no-sync python -m build
uv run --no-sync python scripts/build_docs.py
uv run --no-sync zensical build --strict
```

Notebook sources are committed without outputs. Documentation preparation executes
clean CPU/float64 kernels, exports Markdown and figures to `build/docs/`, copies
the canonical glossary, and retains the unexecuted notebooks as downloads.
Generated files are disposable. See [development details](docs/development.md)
for local preview and GitHub Pages publication.

## License

Developed by Victor Boussange and contributors. Licensed under the
[Apache License 2.0](LICENSE). Releases are distributed through GitHub, not PyPI.
