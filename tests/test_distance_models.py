"""Custom distance models use the public fitting and native persistence boundary."""

from typing import final

import coordax as cx
import jax
import jax.numpy as jnp
import numpy as np

from ilg_toolkit import (
    PairwiseObservations,
    RegionBatch,
    TargetSpec,
    TrainingConfig,
    fit,
    load_model,
    save_model,
)
from ilg_toolkit.models import AbstractDistanceModel


@final
class CustomDistance(AbstractDistanceModel):
    scale: jax.Array

    def predict_distances(
        self,
        features,
        pixel_nodes,
        *,
        context=None,
        inference=True,
        key=None,
        patch_batch_size=None,
    ):
        values = jnp.asarray(features).reshape(-1)[jnp.asarray(pixel_nodes)]
        return self.scale**2 * (values[:, None] - values[None, :]) ** 2


def test_custom_common_distance_contract_fits_and_reloads_without_genetic_labels(tmp_path):
    region = RegionBatch(
        "custom",
        cx.field(np.array([[[0.0], [1.0], [3.0]]]), "row", "column", "feature"),
        ("west", "middle", "east"),
        np.array([[0, 0], [0, 1], [0, 2]]),
    )
    observations = PairwiseObservations.from_matrix(
        region.sampling_unit_ids,
        [[0, 4, 36], [4, 0, 16], [36, 16, 0]],
        target=TargetSpec("divergence"),
    )
    state = fit(
        region,
        observations,
        model=CustomDistance(jnp.array(1.0)),
        config=TrainingConfig(objective="direct_log1p", epochs=3, learning_rate=0.1),
    )
    assert state.history[-1].training_loss < state.history[0].training_loss
    path = tmp_path / "custom.ilg"
    save_model(path, state.model)
    restored = load_model(path, encoder=CustomDistance(jnp.array(0.0)))
    np.testing.assert_array_equal(
        restored.landscape_scores(region), state.model.landscape_scores(region)
    )
    np.testing.assert_array_equal(
        restored.predict(region).values, state.model.predict(region).values
    )
