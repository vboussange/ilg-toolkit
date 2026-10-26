"""Experimental neural fitting across regions with irregular species coverage."""

from itertools import combinations
from typing import cast

import coordax as cx
import jax
import numpy as np

from ilg_toolkit import (
    MLPECalibrationModule,
    MLPELoss,
    PairwiseObservations,
    RegionBatch,
    TargetSpec,
    TrainingConfig,
    train,
)
from ilg_toolkit.models import UNetEmbeddingDistance


def main():
    with jax.enable_x64():
        species = ("newt", "frog")
        coverage = (species, ("newt",))
        regions, observations = [], []
        target = TargetSpec("synthetic genetic dissimilarity", units="index")
        for index, (name, present) in enumerate(zip(("valley", "upland"), coverage, strict=True)):
            ids = tuple(f"{name}-{corner}" for corner in ("sw", "se", "nw", "ne"))
            features = np.random.default_rng(14 + index).uniform(0.8, 1.6, (3, 3, 2))
            regions.append(
                RegionBatch(
                    name,
                    cx.field(features.astype(np.float32), "row", "column", "feature"),
                    ids * len(present),
                    np.tile([[0, 0], [0, 2], [2, 0], [2, 2]], (len(present), 1)),
                    feature_names=("canopy", "moisture"),
                    species=tuple(label for label in present for _ in ids),
                )
            )
            observations.append(
                tuple(
                    PairwiseObservations.from_pairs(
                        tuple(combinations(ids, 2)),
                        np.array([0.4, 0.7, 1.4, 0.6, 0.9, 0.8]) * (1 + 0.2 * species.index(label)),
                        target=target,
                        species=label,
                    )
                    for label in present
                )
            )
        state = train(
            regions,
            observations,
            distance_model=UNetEmbeddingDistance(
                2,
                num_species=2,
                patch_size=1,
                patch_microbatch_size=4,
                base_channels=2,
                embedding_dim=2,
                dropout=0,
                key=jax.random.key(8),
            ),
            species=species,
            config=TrainingConfig(
                loss=MLPELoss(),
                steps=3,
                eval_every=3,
                region_batch_size=1,
                species_batch_size=1,
                species_microbatch_size=1,
            ),
        )
        print("Experimental multispecies training:", state.history[-1].training_by_species)
        calibration = cast(MLPECalibrationModule, state.selected_model.calibration)
        for region, present in zip(regions, coverage, strict=True):
            predictions = state.selected_model.predict(region, species=present)
            distances = state.selected_model.distance_model.predict(
                region, species=present, species_order=state.selected_model.species
            )
            for name in present:
                fitted = calibration.for_region(region.name, species.index(name))
                assert np.isfinite(predictions[name].values).all()
                assert np.isfinite(distances[name].values).all()
                print(f"{region.name}, {name}: ecological distances {distances[name].values}")
                print(f"  genetic predictions {predictions[name].values}")
                print(f"  calibration intercept={fitted.intercept:.3f}, slope={fitted.slope:.3f}")


if __name__ == "__main__":
    main()
