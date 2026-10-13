"""Public pair prediction and explicit representation conversion contracts."""

from dataclasses import replace

import numpy as np
import pytest
from test_observations import problem

from ilg_toolkit import DirectCalibrationModule, LandscapeGeneticsModel, Prediction, TargetSpec


def test_selected_and_complete_predictions_share_pair_contract_and_matrix_order():
    region, target, distance_model = problem()
    model = LandscapeGeneticsModel(distance_model, DirectCalibrationModule(target), 2)
    selected = model.predict(region, [("unit/99", "unit/7")])
    assert isinstance(selected, Prediction)
    assert selected.region_name == region.name
    assert selected.sampling_unit_ids == region.sampling_unit_ids
    assert selected.pairs == (("unit/99", "unit/7"),)
    np.testing.assert_allclose(selected.values, [1.96])
    np.testing.assert_allclose(
        selected.as_matrix(), [[0, np.nan, 1.96], [np.nan, 0, np.nan], [1.96, np.nan, 0]]
    )
    np.testing.assert_allclose(
        selected.as_matrix(("extra", "unit/99", "unit/7")),
        [[0, np.nan, np.nan], [np.nan, 0, 1.96], [np.nan, 1.96, 0]],
    )
    complete = model.predict(region)
    assert complete.pairs == (("unit/7", "unit/20"), ("unit/7", "unit/99"), ("unit/20", "unit/99"))
    np.testing.assert_allclose(complete.values, [0.49, 1.96, 0.49])
    np.testing.assert_allclose(complete.as_matrix(), model.ecological_distances(region))
    for order in [("unit/7",), ("unit/99", "unit/7", "unit/7")]:
        with pytest.raises(ValueError, match="unique labels.*every predicted endpoint"):
            selected.as_matrix(order)
    relatedness = replace(selected, target=TargetSpec("relatedness", kind="relatedness"))
    with pytest.raises(ValueError, match="dissimilarity diagonal convention"):
        relatedness.as_matrix()
