# Custom distance models

Built-in model parameters, shapes, formulas, and numerical requirements live in
the [distance-model API](api/models.md). The [maximum-likelihood population effects (MLPE) tutorial](examples/mlpe_workflow.md)
shows practical resistance and patch-processing choices.

## Implement an abstract boundary

Subclass an abstract model and implement its abstract methods. Keep fields and
initialization together in the concrete class; use composition to reuse another
concrete model.

| Abstract class | Methods to implement | Output |
| --- | --- | --- |
| `ConductanceModel` | `backbone(features)`, `conductance(latent, species_index)` | Positive conductance grid |
| `EmbeddingDistanceModel` | `backbone(features)`, `embedding(latent, species_index)` | Grid with an embedding dimension last |
| `AbstractDistanceModel` | `backbone(features)`, `ecological_distances(latent, pairs, species_index)` | Ecological distances aligned with pairs |

Declare static `num_species` and `patch_size` fields. Conductance models also
provide a static `solver_config` containing `ResistanceSolverConfig`. The
`features` array has row, column, and feature dimensions. A species index may be
a scalar integer array during training; use array indexing to select its parameters.
For a new distance family, `pairs` is `(nodes, (left, right))`: `nodes` locates sampling units on the
flattened patch grid, and `left` and `right` index those sampling units.

This single-species embedding model applies a trainable linear feature transformation:

```python
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp

from ilg_toolkit.models import EmbeddingDistanceModel


@final
class LinearEmbedding(EmbeddingDistanceModel):
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

The inherited `predict` returns a `Prediction`, and `train` estimates the array-valued
parameters using the selected loss.

Use the [experimental multispecies tutorial](examples/multispecies.md) for
species labels and parameter order. Keep new models as concrete final classes
that implement an abstract boundary; use composition to reuse a concrete model.
