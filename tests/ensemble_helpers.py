"""Small independent labelled ensemble fixtures with repeated cross-species labels."""

from typing import final

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ilg_toolkit import (
    DirectCalibrationModule,
    Ensemble,
    EnsembleMember,
    LandscapeGeneticsModel,
    MemberIdentity,
    ObservationPartition,
    PopulationFold,
    RegionBatch,
    SamplingUnits,
    TargetSpec,
)
from ilg_toolkit.models import EmbeddingDistanceModel


@final
class SpeciesEmbedding(EmbeddingDistanceModel):
    species: tuple[str, ...] = eqx.field(static=True)
    weights: jax.Array
    patch_size: int = eqx.field(static=True, default=1)

    def _embedding_grid(
        self, features, species_index, *, inference=True, key=None, patch_batch_size=None
    ):
        return features * self.weights[species_index]


def labelled_ensemble():
    """Distances for c,d are squared weights: fox(1,4), lynx(9,16)."""
    species = ("fox", "lynx")
    ids = ("a", "b", "c", "d")
    region = RegionBatch(
        "alpine",
        cx.field(np.arange(4.0).reshape(1, 4, 1), "row", "column", "feature"),
        {name: SamplingUnits(ids, np.array([[0, i] for i in range(4)])) for name in species},
    )
    target = TargetSpec("divergence", units="index")
    members = []
    for i, weights in enumerate(((1.0, 3.0), (2.0, 4.0))):
        training = {
            region.name: {
                name: ObservationPartition(region.name, name, (("a", "b"),), "training")
                for name in species
            }
        }
        query = {
            region.name: {
                name: ObservationPartition(region.name, name, (("c", "d"),), "query")
                for name in species
            }
        }
        fold = PopulationFold(
            f"f{i}", {region.name: {name: ("c", "d") for name in species}}, training, query
        )
        model = LandscapeGeneticsModel(
            SpeciesEmbedding(species=species, weights=jnp.asarray(weights)),
            DirectCalibrationModule(target),
            1,
            training_objective="direct_log1p",
            training_pairs={region.name: {name: (("a", "b"),) for name in species}},
        )
        members.append(EnsembleMember(MemberIdentity(fold.fold_id, i), fold, "completed", model))
    return region, Ensemble(tuple(members)), target
