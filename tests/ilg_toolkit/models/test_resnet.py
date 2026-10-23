"""Scientific behavior of models.resnet."""

from dataclasses import replace
from typing import cast

import coordax as cx
import jax
import numpy as np
import pytest

from ilg_toolkit import (
    PairwiseObservations,
    RegionBatch,
    ResistanceSolverConfig,
    TargetSpec,
    TrainingConfig,
    fit,
)
from ilg_toolkit.models import ConductanceModel, ResNet9Conductance


@pytest.mark.parametrize("num_species", [1, 2])
def test_resnet_fit_exposes_surface_and_label_free_target_predictions(
    dense_resistance, num_species
):
    with jax.enable_x64():
        features = np.random.default_rng(4).normal(size=(8, 8, 2)).astype(np.float32)
        region = RegionBatch(
            "small-valley",
            cx.field(features, "row", "column", "feature"),
            ("south", "north", "east"),
            np.array([[0, 0], [0, 7], [7, 7]]),
        )
        observations = PairwiseObservations.from_matrix(
            region.sampling_unit_ids,
            dense_resistance(np.full((2, 2), 2.0), np.array([0, 1, 3])),
            target=TargetSpec("synthetic dissimilarity", units="index"),
        )
        model = ResNet9Conductance(
            in_channels=2,
            num_species=num_species,
            patch_size=4,
            patch_microbatch_size=3,
            solver_config=ResistanceSolverConfig(rtol=1e-8),
            key=jax.random.key(6),
        )
        species = None if num_species == 1 else ("frog", "newt")
        if species is not None:
            region = replace(
                region,
                sampling_unit_ids=region.sampling_unit_ids * 2,
                grid_positions=np.tile(region.grid_positions, (2, 1)),
                sampling_unit_kinds=None,
                species=("frog",) * 3 + ("newt",) * 3,
            )
            observations = tuple(
                replace(observations, species=name, values=observations.values * (i + 1))
                for i, name in enumerate(species)
            )
        result = fit(
            region,
            observations,
            distance_model=model,
            species=species,
            config=TrainingConfig(
                objective="direct_log1p",
                steps=4,
                learning_rate=0.001,
            ),
        )
        assert result.history[-1].training_loss < result.history[0].training_loss
        for name in (None,) if species is None else species:
            surface = cast(ConductanceModel, result.model.distance_model).conductance_surface(
                region, species=name, species_order=species
            )
            prediction = result.model.predict(region, species=name)
            assert surface.shape == (2, 2)
            assert np.all(np.isfinite(surface) & (surface > 0))
            expected = dense_resistance(surface, np.array([0, 1, 3]))
            np.fill_diagonal(expected, np.nan)
            np.testing.assert_allclose(prediction.as_matrix(), expected, atol=2e-6, rtol=2e-6)
            np.testing.assert_allclose(
                result.model.distance_model.predict(
                    region, species=name, species_order=species
                ).as_matrix(),
                prediction.as_matrix(),
            )
