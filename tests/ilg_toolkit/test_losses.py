"""Loss settings and whole-group direct training behavior."""

import jax.numpy as jnp
import numpy as np
from ilg_toolkit import Log1pLoss, train, TrainingConfig


def test_direct_loss_trains_without_statistical_parameters(
    ScalarEmbedding, regions_and_observations
):
    regions, observations = regions_and_observations()
    model = ScalarEmbedding(jnp.asarray(1.0))
    state = train(
        regions[0],
        observations[0],
        distance_model=model,
        config=TrainingConfig(loss=Log1pLoss(), steps=2),
    )
    assert state.loss_state is None
    assert state.history[-1].training_loss < state.history[0].training_loss
    prediction = state.selected_model.predict(regions[0], pairs=observations[0].observed_pairs)
    assert np.isfinite(prediction.values).all()


def test_mlpe_losses_reject_individual_based_regions_and_log1p_accepts_them(
    ScalarEmbedding, four_population_calibration
):
    from dataclasses import replace

    import pytest
    from ilg_toolkit import calibrate, MLPELoss

    region, _, observations = four_population_calibration
    individuals = replace(region, individual_based=True)
    model = ScalarEmbedding(jnp.asarray(1.0))
    with pytest.raises(ValueError, match="individual-based"):
        train(
            individuals,
            observations,
            distance_model=model,
            config=TrainingConfig(loss=MLPELoss(), steps=0),
        )
    with pytest.raises(ValueError, match="individual-based"):
        calibrate(individuals, observations, model, loss=MLPELoss())
    state = train(
        individuals,
        observations,
        distance_model=model,
        config=TrainingConfig(loss=Log1pLoss(), steps=1),
    )
    assert np.isfinite(state.selected_model.predict(individuals).values).all()
    assert np.isfinite(
        calibrate(individuals, observations, model, loss=Log1pLoss()).predict(individuals).values
    ).all()
