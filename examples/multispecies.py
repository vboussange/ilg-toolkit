"""Experimental multispecies fitting with a shared landscape and repeated sampling-unit ids."""

from typing import cast

import coordax as cx
import jax
import numpy as np

from ilg_toolkit import PairwiseObservations, RegionBatch, TargetSpec, TrainingConfig, fit
from ilg_toolkit.models import LogLinearConductance


def main():
    with jax.enable_x64():
        species = ("newt", "frog")
        region = RegionBatch(
            "valley",
            cx.field(
                np.array([[[0.0], [0.3], [1.0]], [[0.2], [0.7], [0.5]]]), "row", "column", "feature"
            ),
            ("west", "north", "east") * 2,
            np.tile([[0, 0], [0, 2], [1, 2]], (2, 1)),
            feature_names=("canopy",),
            species=("newt",) * 3 + ("frog",) * 3,
        )
        target = TargetSpec("synthetic genetic dissimilarity", units="index")
        observations = tuple(
            PairwiseObservations.from_pairs(
                [("west", "north"), ("west", "east"), ("north", "east")],
                values,
                target=target,
                species=name,
            )
            for name, values in zip(species, ([1.2, 1.6, 0.8], [0.6, 0.9, 0.4]), strict=True)
        )
        state = fit(
            region,
            observations,
            distance_model=LogLinearConductance(1, num_species=2, patch_size=1),
            species=species,
            config=TrainingConfig(objective="direct_log1p", steps=3, species_microbatch_size=1),
        )
        predictions = state.model.predict(region, species=species)
        surfaces = cast(LogLinearConductance, state.model.distance_model).conductance_surface(
            region,
            species=species,
            species_order=state.model.species,
        )
        for name in species:
            assert np.isfinite(predictions[name].values).all()
            assert np.all(surfaces[name] > 0)
            print(f"{name}: {np.asarray(predictions[name].values)}")


if __name__ == "__main__":
    main()
