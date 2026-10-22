"""Scientific behavior of models.base."""

import coordax as cx
import jax.numpy as jnp
import numpy as np
import pytest

from ilg_toolkit import (
    PairwiseObservations,
    RegionBatch,
    TargetSpec,
    TrainingConfig,
    fit,
    load_model,
    save_model,
)


@pytest.mark.parametrize("objective", ["direct_log1p", "mlpe"])
def test_custom_common_distance_contract_fits_and_reloads_without_genetic_labels(
    CustomDistance, tmp_path, objective
):
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
        distance_model=CustomDistance(jnp.array(1.0)),
        config=TrainingConfig(objective=objective, epochs=3, learning_rate=0.1),
    )
    assert state.history[-1].training_loss < state.history[0].training_loss
    path = tmp_path / "custom.ilg"
    save_model(path, state.model)
    restored = load_model(path, distance_model=CustomDistance(jnp.array(0.0)))
    np.testing.assert_array_equal(
        restored.ecological_distances(region), state.model.ecological_distances(region)
    )
    np.testing.assert_array_equal(
        restored.predict(region).values, state.model.predict(region).values
    )
