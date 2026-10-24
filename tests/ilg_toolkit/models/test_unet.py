"""Scientific behavior of models.unet."""

import numpy as np
import pytest

from ilg_toolkit import TrainingConfig, fit


@pytest.mark.parametrize("num_species", [1, 2])
def test_fixed_budget_fit_improves_and_predicts_without_query_targets(
    synthetic_problem, num_species
):
    region, observations, model = synthetic_problem(num_species=num_species)
    species = None if num_species == 1 else ("frog", "newt")
    result = fit(
        region,
        observations,
        distance_model=model,
        species=species,
        config=TrainingConfig(objective="direct_log1p", steps=2, learning_rate=0.01, seed=3),
    )
    prediction = result.model.predict(region, species=None if species is None else "frog")
    assert result.history[-1].training_loss < result.history[0].training_loss * 0.75
    assert result.selected_step == 2
    assert result.selection == "final"
    assert prediction.as_matrix().shape == (4, 4)
    assert np.all(np.isfinite(prediction.values))
    np.testing.assert_allclose(prediction.as_matrix(), prediction.as_matrix().T, atol=1e-6)
    assert np.isnan(np.diag(prediction.as_matrix())).all()
    np.testing.assert_allclose(
        result.model.distance_model.predict(
            region, species=None if species is None else "frog", species_order=species
        ).as_matrix(),
        prediction.as_matrix(),
    )
