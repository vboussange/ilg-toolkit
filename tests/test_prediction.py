"""Public pair prediction and explicit representation conversion contracts."""

from dataclasses import replace

import numpy as np
import pytest
from test_observations import problem

from ilg_toolkit import DirectCalibrationModule, LandscapeGeneticsModel, Prediction, TargetSpec


def test_selected_embedding_pairs_do_not_evaluate_unrequested_separations():
    import coordax as cx
    import jax.numpy as jnp
    from test_observations import LinearEmbedding

    from ilg_toolkit import RegionBatch, SamplingUnits

    region = RegionBatch(
        "extreme-unrequested-unit",
        cx.field(np.array([[[0.0], [2.0], [1e20]]], dtype=np.float32), "row", "column", "feature"),
        {"species": SamplingUnits(("a", "b", "unrequested"), np.array([[0, 0], [0, 1], [0, 2]]))},
    )
    model = LandscapeGeneticsModel(
        LinearEmbedding(jnp.asarray(1.0, dtype=jnp.float32)),
        DirectCalibrationModule(TargetSpec("divergence")),
        1,
    )
    prediction = model.predict(region, pairs={"species": [("b", "a")]})["species"]
    np.testing.assert_array_equal(prediction.values, [4.0])
    assert prediction.pairs == (("b", "a"),)
    with pytest.raises(FloatingPointError, match="nonfinite"):
        model.ecological_distances(region)["species"]


@pytest.mark.parametrize("jit", [False, True])
def test_sparse_embedding_fit_uses_only_observed_pair_distances_and_gradients(jit):
    import coordax as cx
    import jax.numpy as jnp
    import optax
    from test_observations import LinearEmbedding

    from ilg_toolkit import PairwiseObservations, RegionBatch, SamplingUnits, TrainingConfig, fit

    region = RegionBatch(
        "sparse",
        cx.field(np.array([[[0.0], [2.0], [3e38]]], dtype=np.float32), "row", "column", "feature"),
        {"species": SamplingUnits(("a", "b", "unrequested"), np.array([[0, 0], [0, 1], [0, 2]]))},
    )
    observations = PairwiseObservations.from_pairs(
        [("a", "b")], [9.0], target=TargetSpec("divergence")
    )
    state = fit(
        [region],
        {region.name: {"species": observations}},
        distance_model=LinearEmbedding(jnp.asarray(1.0, dtype=jnp.float32)),
        config=TrainingConfig(objective="direct_log1p", steps=1, learning_rate=0.1, jit=jit),
    )
    assert isinstance(state.optimizer_state, tuple)
    adam = state.optimizer_state[0]
    assert isinstance(adam, optax.ScaleByAdamState)
    assert isinstance(adam.mu, LinearEmbedding)
    expected_gradient = 2 * (np.log(5) - np.log(10)) * 8 / 5
    assert float(adam.mu.weight) == pytest.approx(0.1 * expected_gradient, abs=1e-07)
    assert isinstance(state.distance_model, LinearEmbedding)
    assert float(state.distance_model.weight) == pytest.approx(1.1, abs=2e-06)
    assert state.history[-1].training_loss < state.history[0].training_loss


def test_selected_and_complete_predictions_share_pair_contract_and_matrix_order():
    region, target, distance_model = problem()
    model = LandscapeGeneticsModel(distance_model, DirectCalibrationModule(target), 2)
    selected = model.predict(region, pairs={"species": [("unit/99", "unit/7")]})["species"]
    assert isinstance(selected, Prediction)
    assert selected.region_name == region.name
    assert selected.sampling_unit_ids == region.sampling_units["species"].ids
    assert selected.pairs == (("unit/99", "unit/7"),)
    np.testing.assert_allclose(selected.values, [1.96])
    np.testing.assert_allclose(
        selected.as_matrix(), [[0, np.nan, 1.96], [np.nan, 0, np.nan], [1.96, np.nan, 0]]
    )
    np.testing.assert_allclose(
        selected.as_matrix(("extra", "unit/99", "unit/7")),
        [[0, np.nan, np.nan], [np.nan, 0, 1.96], [np.nan, 1.96, 0]],
    )
    complete = model.predict(region)["species"]
    assert complete.pairs == (("unit/7", "unit/20"), ("unit/7", "unit/99"), ("unit/20", "unit/99"))
    np.testing.assert_allclose(complete.values, [0.49, 1.96, 0.49])
    np.testing.assert_allclose(complete.as_matrix(), model.ecological_distances(region)["species"])
    for order in [("unit/7",), ("unit/99", "unit/7", "unit/7")]:
        with pytest.raises(ValueError, match="unique labels.*every predicted endpoint"):
            selected.as_matrix(order)
    relatedness = replace(selected, target=TargetSpec("relatedness", kind="relatedness"))
    with pytest.raises(ValueError, match="dissimilarity diagonal convention"):
        relatedness.as_matrix()
