"""Scientific behavior of prediction."""

import jax.numpy as jnp
import numpy as np
import pytest
from ilg_toolkit import DirectCalibrationModule, LandscapeGeneticsModel, Prediction, TargetSpec


def test_pair_values_and_extras_have_the_same_sparse_matrix_representation():
    prediction = Prediction(
        values=jnp.array([1.5, -0.2]),
        pairs=(("c", "a"), ("b", "c")),
        sampling_unit_ids=("a", "b", "c", "d"),
        region_name="valley",
        extras={"effect_variance": jnp.array([0.4, 0.7])},
    )
    np.testing.assert_allclose(
        prediction.as_matrix(),
        [
            [np.nan, np.nan, 1.5, np.nan],
            [np.nan, np.nan, -0.2, np.nan],
            [1.5, -0.2, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan],
        ],
    )
    np.testing.assert_allclose(
        prediction.as_matrix("effect_variance"),
        [
            [np.nan, np.nan, 0.4, np.nan],
            [np.nan, np.nan, 0.7, np.nan],
            [0.4, 0.7, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan],
        ],
    )
    with pytest.raises(ValueError):
        Prediction(
            jnp.array([1.5]),
            (("c", "a"),),
            "valley",
            ("a", "b", "c"),
            extras={"effect_variance": jnp.array([0.4, 0.7])},
        )


def test_selected_and_complete_predictions_share_pair_contract_and_matrix_order(
    observation_problem,
):
    region, target, distance_model = observation_problem()
    model = LandscapeGeneticsModel(distance_model, DirectCalibrationModule(target), 2)
    selected = model.predict(region, [("unit/99", "unit/7")])
    assert selected.region_name == region.name
    assert selected.sampling_unit_ids == region.sampling_unit_ids
    assert selected.pairs == (("unit/99", "unit/7"),)
    np.testing.assert_allclose(selected.values, [1.96])
    np.testing.assert_allclose(
        selected.as_matrix(),
        [[np.nan, np.nan, 1.96], [np.nan, np.nan, np.nan], [1.96, np.nan, np.nan]],
    )
    complete = model.predict(region)
    assert complete.pairs == (("unit/7", "unit/20"), ("unit/7", "unit/99"), ("unit/20", "unit/99"))
    np.testing.assert_allclose(complete.values, [0.49, 1.96, 0.49])
    relatedness = LandscapeGeneticsModel(
        distance_model, DirectCalibrationModule(TargetSpec("relatedness", kind="relatedness")), 2
    ).predict(region, [("unit/99", "unit/7")])
    np.testing.assert_allclose(relatedness.as_matrix(), selected.as_matrix())
