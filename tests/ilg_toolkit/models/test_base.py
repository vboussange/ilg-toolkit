"""Scientific behavior of models.base."""

from typing import cast

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
from ilg_toolkit.models import ConductanceModel


def test_custom_conductance_hooks_fit_and_predict_requested_pairs(ConstantConductance):
    region = RegionBatch(
        "river",
        cx.field(np.ones((1, 3, 1)), "row", "column", "feature"),
        ("a", "b", "c"),
        np.array([[0, 0], [0, 1], [0, 2]]),
    )
    distance_model = ConstantConductance(jnp.array(2.0))
    prediction = distance_model.predict(region, pairs=[("c", "a"), ("b", "a")])
    np.testing.assert_allclose(prediction.values, [1.0, 0.5], atol=1e-6)
    observations = PairwiseObservations.from_pairs(
        [("a", "b"), ("a", "c"), ("b", "c")],
        [1.0, 2.0, 1.0],
        target=TargetSpec("dissimilarity"),
    )
    state = fit(
        region,
        observations,
        distance_model=distance_model,
        config=TrainingConfig(objective="direct_log1p", epochs=3, learning_rate=0.1),
    )
    assert state.history[-1].training_loss < state.history[0].training_loss
    np.testing.assert_allclose(
        state.model.predict(region).values,
        state.model.distance_model.predict(region).values,
    )
    assert np.all(
        cast(ConductanceModel, state.model.distance_model).conductance_surface(region) > 0
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
        restored.distance_model.predict(region).as_matrix(),
        state.model.distance_model.predict(region).as_matrix(),
    )
    np.testing.assert_array_equal(
        restored.predict(region).values, state.model.predict(region).values
    )
