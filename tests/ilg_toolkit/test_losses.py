"""Loss settings and whole-group direct training behavior."""

import jax.numpy as jnp
import numpy as np

from ilg_toolkit import Log1pLoss, TrainingConfig, train


def test_direct_loss_trains_without_statistical_parameters(
    ScalarEmbedding, regions_and_observations
):
    regions, observations = regions_and_observations()
    model = ScalarEmbedding(jnp.asarray(1.0))
    state = train(
        regions[0],
        observations[0],
        distance_model=model,
        config=TrainingConfig(loss=Log1pLoss(), steps=2, jit=False),
    )
    assert state.loss_state is None
    assert state.history[-1].training_loss < state.history[0].training_loss
    prediction = state.selected_model.predict(regions[0], pairs=observations[0].observed_pairs)
    assert np.isfinite(prediction.values).all()
