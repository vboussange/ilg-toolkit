"""Session-scoped fitted states shared by the public training and calibration tests.

Each fixture fits once per test process. Fitted models are immutable Equinox
modules, so tests read them without copying.
"""

import itertools
from types import SimpleNamespace

import jax.numpy as jnp
import pytest
from ilg_toolkit import calibrate, MLPELoss, train, TrainingConfig


@pytest.fixture(scope="session")
def four_population_calibration(calibration_problem):
    """Six-population region whose MLPE calibration pairs join its first four populations.

    The last two populations stay unseen, so predictions for them use the
    population-effect prior. All MLPE calibrations of this problem share one
    pair count and population count, so they reuse compiled programs.
    """
    region, observations, _, _ = calibration_problem()
    pairs = tuple(itertools.combinations(region.sampling_unit_ids[:4], 2))
    return region, observations, observations.select_pairs(pairs)


@pytest.fixture(scope="session")
def mlpe_calibrated_model(four_population_calibration, ScalarEmbedding):
    """MLPE calibration of a frozen unit-weight embedding on the four-population pairs."""
    region, _, training = four_population_calibration
    return calibrate(region, training, ScalarEmbedding(jnp.asarray(1.0)), loss=MLPELoss())


@pytest.fixture(scope="session")
def mlpe_embedding_fit(WeightedEmbedding, embedding_problem):
    """Three-step joint MLPE fit of a weighted embedding, with every callback state."""
    region, observations = embedding_problem(columns=2)
    callbacks = []
    state = train(
        region,
        observations,
        distance_model=WeightedEmbedding(jnp.zeros(2)),
        config=TrainingConfig(loss=MLPELoss(), steps=3, eval_every=3, learning_rate=0.03),
        on_step=callbacks.append,
    )
    return SimpleNamespace(
        region=region, observations=observations, state=state, callbacks=tuple(callbacks)
    )
