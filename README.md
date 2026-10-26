# ILG Toolkit

**Utilities for inverse landscape genetics in JAX.**

[![Run tests](https://github.com/vboussange/ilg-toolkit/actions/workflows/run_tests.yml/badge.svg)](https://github.com/vboussange/ilg-toolkit/actions/workflows/run_tests.yml)
[![Documentation](https://github.com/vboussange/ilg-toolkit/actions/workflows/docs.yml/badge.svg)](https://github.com/vboussange/ilg-toolkit/actions/workflows/docs.yml)

ILG Toolkit provides routines for estimating relationships between landscape
features and pairwise genetic observations among populations or individuals.
A distance model maps landscape features to ecological distances; genetic
calibration relates those distances to the observed genetic measurements.

<p align="center">
  <img src="https://raw.githubusercontent.com/vboussange/ilg-toolkit/main/docs/assets/ilg-overview.webp" width="420" alt="Landscape features become conductance, effective resistance, and genetic predictions.">
</p>

- Differentiable effective resistance and embedding distance models.
- Joint training of distance-model parameters and statistical parameters, with
  maximum-likelihood population effects (MLPE) for population dependence.
- Genetic calibration, conditional prediction, and ensembles on the target's
  original scale.
- Parameters pooled across regions, with experimental multispecies support.

## Installation

Requires Python 3.12 or newer. Install the development version with:

```bash
pip install git+https://github.com/vboussange/ilg-toolkit.git
```

Optional extras include `amg` for resistance preconditioning, `geo` for geospatial
input preparation, and `gpu` for CUDA-enabled JAX on Linux x86-64.

## Documentation

See the [documentation](https://vboussange.github.io/ilg-toolkit/) for examples,
statistical guidance, and the API reference.

## Quick example

Given a prepared landscape `region` and pairwise genetic `observations`, train
a conductance model and request genetic predictions:

```python
import jax

from ilg_toolkit import Log1pLoss, TrainingConfig, train
from ilg_toolkit.models import ResNet9Conductance

jax.config.update("jax_enable_x64", True)  # Before constructing inputs.
distance_model = ResNet9Conductance(in_channels=2, patch_size=4, key=jax.random.key(6))
state = train(
    region,
    observations,
    distance_model=distance_model,
    config=TrainingConfig(loss=Log1pLoss(), steps=4, learning_rate=0.001),
)
prediction = state.selected_model.predict(region)
```

The [quickstart](https://vboussange.github.io/ilg-toolkit/examples/quickstart/)
provides the complete runnable script, including synthetic inputs. Interpret
estimated landscape relationships alongside ecological knowledge and a stated
evaluation design.

## Contributing

See [CONTRIBUTING.md](https://github.com/vboussange/ilg-toolkit/blob/main/CONTRIBUTING.md)
for development checks, documentation conventions, and releases.

## License

Developed by Victor Boussange and contributors. Licensed under the
[Apache License 2.0](https://github.com/vboussange/ilg-toolkit/blob/main/LICENSE).
