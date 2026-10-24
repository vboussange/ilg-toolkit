# Distance models

A distance model derives ecological distances from a region batch's features and
sampling-unit locations. The [README](../README.md) introduces the model hierarchy
and fitting workflow; this page covers model settings and extension points.

## Model settings

All built-in models accept `in_channels`, `patch_size`, and
`patch_microbatch_size`. Raster height and width must be divisible by `patch_size`;
sampling units use the patch containing their grid position. Feature meanings and
order must match those used for training. The [performance guide](performance.md)
explains patch processing and resistance-solver settings.

`LogLinearConductance` averages covariates within each patch. Its coefficients
follow feature order and, together with its intercept, start at zero. For linear
predictor $\eta$, conductance is
$\min(\texttt{max_conductance},\texttt{min_conductance}+\exp(\eta))$.
The default limits are 0.01 and 1000, and the default patch size is 32.

Prepare covariates before constructing a region batch: choose their scaling and
categorical reference levels, and use a full-rank coding with the intercept.
For example, represent a category with $k$ levels using $k-1$ indicator columns.
Coefficients are fitted separately for each species and pooled across that
species' regions. Cross-species feature learning comes from the neural models'
pooled backbone.

The intercept scales the exponential term before the floor and cap are applied;
those limits can change relative conductances as the intercept varies. With
standardized maximum-likelihood population effects (MLPE) calibration, interpret
conductance patterns on a relative scale.

Note: standardized MLPE does not identify an absolute connectivity scale, and
redundant categorical indicators alongside an intercept produce collinearity.

`ResNet9Conductance` uses a residual network on each patch, with default patch size
4. `UNet` uses patch embeddings and a U-Net across the region, with default patch
size 32. Both accept a JAX random `key` for initialization. `UNet.embedding_dim`
controls the dimension of each sampling unit's embedding.

## Custom distance models

Subclass an abstract model and implement its abstract methods. Keep fields and
initialization together in the concrete class; use composition to reuse another
concrete model.

| Abstract class | Methods to implement | Output |
| --- | --- | --- |
| `ConductanceModel` | `backbone(features)`, `conductance(latent, species_index)` | Positive conductance grid |
| `EmbeddingModel` | `backbone(features)`, `embedding(latent, species_index)` | Grid with an embedding dimension last |
| `AbstractDistanceModel` | `backbone(features)`, `ecological_distances(latent, pairs, species_index)` | Ecological distances aligned with pairs |

Declare static `num_species` and `patch_size` fields. Conductance models also
provide a static `solver_config` containing `ResistanceSolverConfig`. The
`features` array has row, column, and feature dimensions. A species index may be
a scalar integer array during fitting; use array indexing to select its parameters.
For a new distance family, `pairs` is `(nodes, (left, right))`: `nodes` locates sampling units on the
flattened patch grid, and `left` and `right` index those sampling units.

This single-species embedding model learns a linear feature transformation:

```python
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp

from ilg_toolkit.models import EmbeddingModel


@final
class LinearEmbedding(EmbeddingModel):
    """A distance model using linearly transformed landscape features as embeddings."""

    weights: jax.Array
    num_species: int = eqx.field(static=True, default=1)
    patch_size: int = eqx.field(static=True, default=1)

    def backbone(self, features: jax.Array) -> jax.Array:
        return features @ self.weights

    def embedding(self, latent: jax.Array, species_index: int | jax.Array) -> jax.Array:
        return latent


distance_model = LinearEmbedding(jnp.eye(2))
```

The inherited `predict` returns a `Prediction`, and `fit` learns the array-valued
parameters using the selected training objective.

## Experimental multispecies use

Label sampling units with `RegionBatch.species` and each observations object with
`PairwiseObservations.species`. For each region, supply a sequence of observations
objects, one per species. Sampling-unit identifiers may repeat across species;
species may be absent from some regions.

Construct the distance model with `num_species` and pass its parameter order to
`fit(..., species=(...))`. The landscape genetics model retains this species order
as `model.species`. Fitted queries take `species="name"` for one `Prediction`, or a
sequence of names for a mapping. Multispecies queries require a species selection.

Standalone distance-model `predict` and `conductance_surface` take the same name
selection plus `species_order=model.species`. A sequence reuses the pooled
backbone for the region. Distance models retain `num_species`; the supplied order
associates names with their parameter rows at query time. See the
[multispecies example](../examples/multispecies.py) for a runnable workflow.
