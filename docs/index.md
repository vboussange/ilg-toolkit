# ILG Toolkit

**Learn how landscapes relate to genetic observations.** ILG Toolkit combines
JAX distance models with genetic calibration, starting from landscape rasters,
located sampling units, and pairwise measurements. Begin with one species;
[multispecies use](examples/multispecies.md) is experimental.

![Landscape features become conductance, resistance, and genetic predictions.](assets/ilg-overview.webp){ width="500" }

A distance model derives **ecological distances** from the landscape. A fitted
landscape genetics model combines those distances with **genetic calibration** to
return predictions in the original measurement units. Conductance models use
effective resistance; embedding models use squared Euclidean distances between
learned embeddings. Different landscape relationships can produce similar
pairwise distances, so interpret fitted surfaces with ecological knowledge and
a stated evaluation design.

## Install and start

Python 3.12 or newer is required. Install from a tagged GitHub release:

```bash
uv add "ilg-toolkit @ git+https://github.com/vboussange/ilg-toolkit@v0.0.3"
```

For the development API documented here, install from the repository checkout:

```bash
uv sync --extra docs
```

Enable float64 before creating resistance or MLPE inputs:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

The [quickstart](examples/quickstart.md) constructs synthetic inputs, trains a
conductance model, inspects predictions, saves and reloads the result, and resumes
training from a checkpoint. Every tutorial includes a downloadable notebook and
runs from a fresh CPU Python session.

## Choose the next question

| Scientific task | Where to start |
| --- | --- |
| Complete a first analysis | [Quickstart](examples/quickstart.md) |
| Understand a geographic reference | [Isolation by distance](examples/ibd.md) |
| Learn across regions, evaluate population splits, and transfer | [MLPE workflow](examples/mlpe_workflow.md) |
| Compare and combine fitted candidates | [Ensemble](examples/ensemble.md) |
| Pool landscape learning across labeled species | [Experimental multispecies](examples/multispecies.md) |

`train` jointly optimizes distance parameters and any loss-specific statistical
parameters. `calibrate` freezes a supplied distance model and estimates its genetic
relationship on supplied observations. Built-in training validation retains
training-fitted calibration; fresh-calibration explanatory evaluation belongs in
the study's notebook. The [statistics guide](statistics.md) explains this distinction.

Consult the [API reference](api/training.md) for signatures and numerical
requirements, the [custom model guide](models.md) for extensions, and the
[canonical glossary](glossary.md) for domain terms.
