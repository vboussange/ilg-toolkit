"""Fixtures of session-scoped training states and calibration heads shared across tests.

Each fixture is built once per test process. Tests read its immutable Equinox
modules without copying.
"""

import itertools
from types import SimpleNamespace

import jax.numpy as jnp
import optax
import pytest
from ilg_toolkit import (
    fit_calibration_head,
    LandscapeGeneticsModel,
    MLPELoss,
    train,
    TrainingConfig,
)


@pytest.fixture(scope="session")
def four_population_calibration(calibration_problem):
    """Returns a six-population region, its observations, and MLPE calibration pairs.

    The calibration pairs join the region's first four populations.
    Predictions for the last two populations use the population-effect prior.
    All MLPE calibrations of this problem share one pair count and population
    count and reuse compiled programs.
    """
    region, observations, _, _ = calibration_problem()
    pairs = tuple(itertools.combinations(region.sampling_unit_ids[:4], 2))
    return region, observations, observations.select_pairs(pairs)


@pytest.fixture(scope="session")
def mlpe_calibrated_head(four_population_calibration, ScalarEmbedding):
    """Returns the MLPE calibration head of a frozen unit-weight embedding on four populations."""
    region, _, training = four_population_calibration
    return fit_calibration_head(
        region, training, ScalarEmbedding(jnp.asarray(1.0)), loss=MLPELoss()
    )


@pytest.fixture(scope="session")
def mlpe_calibrated_model(four_population_calibration, mlpe_calibrated_head, ScalarEmbedding):
    """Returns the frozen unit-weight embedding composed with its MLPE calibration head."""
    region, _, _ = four_population_calibration
    return LandscapeGeneticsModel(
        ScalarEmbedding(jnp.asarray(1.0)), mlpe_calibrated_head, region.feature_names
    )


@pytest.fixture(scope="session")
def mlpe_embedding_training(WeightedEmbedding, embedding_problem):
    """Returns a three-step MLPE training state of a weighted embedding and every callback state."""
    region, observations = embedding_problem(columns=2)
    callbacks = []
    state = train(
        region,
        observations,
        distance_model=WeightedEmbedding(jnp.zeros(2)),
        config=TrainingConfig(loss=MLPELoss(), steps=3, eval_every=3, optimizer=optax.adam(0.03)),
        on_step=callbacks.append,
    )
    return SimpleNamespace(
        region=region, observations=observations, state=state, callbacks=tuple(callbacks)
    )
