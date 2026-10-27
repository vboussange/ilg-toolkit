# ILG Toolkit

**Utilities for inverse landscape genetics in JAX.**

[![Run tests](https://github.com/vboussange/ilg-toolkit/actions/workflows/run_tests.yml/badge.svg)](https://github.com/vboussange/ilg-toolkit/actions/workflows/run_tests.yml)
[![Documentation](https://github.com/vboussange/ilg-toolkit/actions/workflows/docs.yml/badge.svg)](https://github.com/vboussange/ilg-toolkit/actions/workflows/docs.yml)

Inverse landscape genetics (ILG) infers how landscapes shape movement from
geolocated genetic data. ILG Toolkit provides utilities to train a patch encoder
end-to-end through a differentiable effective-resistance layer: the encoder maps
patches of landscape features to the vertex conductances of a conductance grid
graph, whose effective resistances between sampling units predict their pairwise
genetic dissimilarity.

<p align="center">
  <img src="https://raw.githubusercontent.com/vboussange/ilg-toolkit/main/docs/assets/ilg-overview.webp" width="420" alt="Landscape features become conductance, effective resistance, and genetic predictions.">
</p>

- Log-linear and ResNet9 patch encoders, and a U-Net embedding-distance model.
- GPU-ready effective resistance with gradients, optionally preconditioned by
  algebraic multigrid.
- Training under complete pooling across regions, with a log1p objective or
  maximum-likelihood population effects (MLPE) for population dependence.
- Genetic calibration, conditional prediction, and ensembles on the target's
  original scale, with experimental multispecies support.

## Installation

Requires Python 3.12 or newer. Add the development version and the quickstart's
dependencies to a [uv](https://docs.astral.sh/uv/) project with:

```bash
uv add "ilg-toolkit[examples] @ git+https://github.com/vboussange/ilg-toolkit"
```

The `examples` extra installs rioxarray and Matplotlib for the examples. Other
extras are `amg` for algebraic-multigrid preconditioning and `gpu` for
CUDA-enabled JAX on Linux x86-64.

## Quickstart

The [quickstart](https://vboussange.github.io/ilg-toolkit/examples/quickstart/)
trains a log-linear patch encoder on pairwise Fst between 12 Mountain
Pygmy-possum sampling units in the Victorian Alps, kindly provided by
[Cesar Australia](https://cesaraustralia.com), using ESA WorldCover 2021 land
cover (CC BY 4.0). It finishes in about a minute on a CPU.

<p align="center">
  <img src="https://vboussange.github.io/ilg-toolkit/examples/quickstart.png" width="720" alt="Land cover and sampling units next to the trained log10 resistance surface.">
</p>

After turning land-cover fractions into landscape `features` and sampling-unit
coordinates into `grid_positions`, the script trains the patch encoder from the
best uniform conductance and reads its resistance surface:

```python
region = RegionBatch(
    name="victorian-alps",
    features=cx.from_xarray(features),
    sampling_unit_ids=tuple(sites.index),
    grid_positions=grid_positions,
)
observations = PairwiseObservations.from_matrix(
    tuple(fst.index),
    fst.to_numpy(),
    target=TargetSpec("pairwise Fst", units="proportion", transform="odds"),
)
loss = Log1pLoss()
distance_model = initialize_uniform_conductance(
    LogLinearConductance(in_channels=3, patch_size=PATCH_SIZE), region, observations, loss
)
state = train(
    region,
    observations,
    distance_model=distance_model,
    config=TrainingConfig(loss=loss, steps=100, learning_rate=0.1),
)
conductance = state.selected_model.distance_model.conductance_surface(region)
resistance = 1 / conductance  # Local resistance Q = 1/C of each vertex.
```

Swap in `ResNet9Conductance` as a deep patch encoder that maps raw patches of
landscape features to vertex conductances.

## Documentation

See the [documentation](https://vboussange.github.io/ilg-toolkit/) for the
examples, the glossary, and the API reference.

## Contributing

See [CONTRIBUTING.md](https://github.com/vboussange/ilg-toolkit/blob/main/CONTRIBUTING.md)
for development checks, documentation conventions, and releases.

## License

Developed by Victor Boussange and contributors. Licensed under the
[Apache License 2.0](https://github.com/vboussange/ilg-toolkit/blob/main/LICENSE).
