"""Scientific behavior of models.unet."""

import numpy as np

from ilg_toolkit import TrainingConfig, fit


def test_fixed_budget_fit_improves_and_predicts_without_query_targets(synthetic_problem):
    region, observations, model = synthetic_problem()
    result = fit(
        region,
        observations,
        distance_model=model,
        config=TrainingConfig(objective="direct_log1p", epochs=20, learning_rate=0.01, seed=3),
    )
    prediction = result.model.predict(region)
    assert result.history[-1].training_loss < result.history[0].training_loss * 0.75
    assert result.selected_epoch == 20
    assert result.selection == "final"
    assert prediction.target.units == "index"
    assert prediction.target.transform == "identity"
    assert prediction.as_matrix().shape == (4, 4)
    assert np.all(np.isfinite(prediction.as_matrix()))
    np.testing.assert_allclose(prediction.as_matrix(), prediction.as_matrix().T, atol=1e-6)
    np.testing.assert_allclose(np.diag(prediction.as_matrix()), 0, atol=1e-6)
    np.testing.assert_allclose(result.model.ecological_distances(region), prediction.as_matrix())
