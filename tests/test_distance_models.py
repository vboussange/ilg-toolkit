"""Custom distance models use the public fitting and native persistence boundary."""

from typing import final

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ilg_toolkit import (
    PairwiseObservations,
    RegionBatch,
    SamplingUnits,
    TargetSpec,
    TrainingConfig,
    fit,
    load_model,
    save_model,
)
from ilg_toolkit.models import AbstractDistanceModel


@final
class CustomDistance(AbstractDistanceModel):
    species: tuple[str, ...] = eqx.field(static=True, default=("species",), kw_only=True)
    scale: jax.Array

    def prepare(self, features):
        return None

    def _pair_distances(
        self,
        features,
        selection,
        species_index,
        *,
        context=None,
        inference=True,
        key=None,
        patch_batch_size=None,
    ):
        values = jnp.asarray(features).reshape(-1)[selection.nodes]
        return self.scale**2 * (values[selection.left] - values[selection.right]) ** 2


@pytest.mark.parametrize("objective", ["direct_log1p", "mlpe"])
def test_custom_common_distance_contract_fits_and_reloads_without_genetic_labels(
    tmp_path, objective
):
    region = RegionBatch(
        "custom",
        cx.field(np.array([[[0.0], [1.0], [3.0]]]), "row", "column", "feature"),
        {"species": SamplingUnits(("west", "middle", "east"), np.array([[0, 0], [0, 1], [0, 2]]))},
    )
    observations = PairwiseObservations.from_matrix(
        region.sampling_units["species"].ids,
        [[0, 4, 36], [4, 0, 16], [36, 16, 0]],
        target=TargetSpec("divergence"),
    )
    state = fit(
        [region],
        {region.name: {"species": observations}},
        distance_model=CustomDistance(jnp.array(1.0)),
        config=TrainingConfig(objective=objective, steps=3, learning_rate=0.1),
    )
    assert state.history[-1].training_loss < state.history[0].training_loss
    path = tmp_path / "custom.ilg"
    save_model(path, state.model)
    restored = load_model(path, distance_model=CustomDistance(jnp.array(0.0)))
    np.testing.assert_array_equal(
        restored.ecological_distances(region)["species"],
        state.model.ecological_distances(region)["species"],
    )
    np.testing.assert_array_equal(
        restored.predict(region)["species"].values, state.model.predict(region)["species"].values
    )
