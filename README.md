# ILG Toolkit

**Differentiable, GPU-accelerated utilities for inverse landscape genetics in JAX.**

[![Run tests](https://github.com/vboussange/ilg-toolkit/actions/workflows/run_tests.yml/badge.svg)](https://github.com/vboussange/ilg-toolkit/actions/workflows/run_tests.yml)
[![Documentation](https://github.com/vboussange/ilg-toolkit/actions/workflows/docs.yml/badge.svg)](https://github.com/vboussange/ilg-toolkit/actions/workflows/docs.yml)

Inverse landscape genetics (ILG) infers spatial ecological networks from
geolocated genetic data by learning a conductance grid graph whose edge weights
encode local landscape permeability to movement, so that the effective
resistances it induces between sampled populations predict their genetic
dissimilarity.

ILG Toolkit provides differentiable, GPU-accelerated utilities for ILG in JAX.
A patch encoder maps patches of landscape features to the vertex conductances of
the conductance grid graph, and is trained end-to-end through a differentiable
effective-resistance layer.

<p align="center">
  <img src="https://raw.githubusercontent.com/vboussange/ilg-toolkit/main/docs/assets/ilg-overview.webp" width="420" alt="Landscape features become conductance, effective resistance, and genetic predictions.">
</p>

- Log-linear and ResNet9 patch encoders, and a U-Net embedding-distance model.
- Effective resistance with gradients on the GPU, optionally preconditioned by
  algebraic multigrid.
- Training under complete pooling across regions, with a log1p loss or
  maximum-likelihood population effects (MLPE) for population dependence.
- Genetic calibration, conditional prediction, and ensembles on the target's
  original scale, with experimental multispecies support.

## Installation

Requires Python 3.12 or newer. Add the development version and the quickstart's
dependencies to a [uv](https://docs.astral.sh/uv/) project with:

```bash
uv add "ilg-toolkit[examples] @ git+https://github.com/vboussange/ilg-toolkit"
```

The `examples` extra installs rioxarray, pandas and Matplotlib for the examples,
and the `amg` extra installs algebraic-multigrid preconditioning.

ILG Toolkit runs on the device JAX selects. To run it on an NVIDIA GPU, install
JAX with CUDA support:

```bash
uv add "jax[cuda12]"
```

The [JAX installation guide](https://docs.jax.dev/en/latest/installation.html)
lists the CUDA versions and platforms JAX supports.

## Quickstart

The [quickstart](https://vboussange.github.io/ilg-toolkit/examples/quickstart/)
trains a log-linear patch encoder on pairwise Fst between 12 Mountain
Pygmy-possum sampling units in the Victorian Alps, kindly provided by
[Cesar Australia](https://cesaraustralia.com), and ESA WorldCover 2021 land
cover (CC BY 4.0). Run it with uv and the `examples` extra:

```bash
curl -O https://raw.githubusercontent.com/vboussange/ilg-toolkit/main/examples/quickstart.py
uv run --with "ilg-toolkit[examples] @ git+https://github.com/vboussange/ilg-toolkit" python quickstart.py
```

The script reads the land-cover GeoTIFF with rioxarray, turns it into land-cover
fractions, places the sampling units by their coordinates, and trains the patch
encoder from the best uniform conductance:

```python
region = RegionBatch.from_points(
    "victorian-alps",
    cx.from_xarray(features),
    sampling_unit_ids=tuple(sites.index),
    coordinates=sites[["x", "y"]].to_numpy(),
)
observations = PairwiseObservations(
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
    config=TrainingConfig(loss=loss, steps=2000, eval_every=100),
)
conductance = state.selected_model.distance_model.conductance_surface(region)
resistance = 1 / conductance  # Local resistance Q = 1/C of each vertex.
```

<p align="center">
  <img src="https://raw.githubusercontent.com/vboussange/ilg-toolkit/main/docs/examples/quickstart_files/quickstart.png" width="720" alt="Land cover and sampling units next to the learned log10 resistance surface.">
</p>

The quickstart runs in about a minute on an NVIDIA RTX 3090.

Swap in `ResNet9Conductance` as a deep patch encoder that maps raw patches of
landscape features to vertex conductances.

## Documentation

See the [documentation](https://vboussange.github.io/ilg-toolkit/) for the
examples, the glossary, and the API reference.

## Contributing

See [CONTRIBUTING.md](https://github.com/vboussange/ilg-toolkit/blob/main/CONTRIBUTING.md)
for development checks, documentation conventions, and releases.

## Interpreting results

Interpret learned resistance surfaces, ecological distances and genetic
predictions with care, in light of each study's design and data. The authors and
contributors of ILG Toolkit accept no responsibility for results obtained with it
or for decisions based on them.

## License

Developed by Victor Boussange and contributors. Licensed under the
[Apache License 2.0](https://github.com/vboussange/ilg-toolkit/blob/main/LICENSE).
