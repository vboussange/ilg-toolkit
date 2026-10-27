"""Scientific behavior of config."""

import numpy as np
import pytest

from ilg_toolkit import Log1pLoss, PairwiseObservations, TrainingConfig, train


@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, "2"])
@pytest.mark.parametrize(
    "name", ["region_batch_size", "species_batch_size", "species_microbatch_size"]
)
def test_region_batch_size_rejects_invalid_counts(value, name):
    with pytest.raises(ValueError):
        TrainingConfig(loss=Log1pLoss(), **{name: value})


def test_fresh_fit_requires_declared_encoder_and_objective_while_resume_reuses_them(
    observation_problem,
):
    region, target, encoder = observation_problem()
    observations = PairwiseObservations.from_pairs([("unit/7", "unit/20")], [0.5], target=target)
    omitted = {}
    with pytest.raises(TypeError):
        TrainingConfig(**omitted)
    config = TrainingConfig(loss=Log1pLoss(), steps=0)
    with pytest.raises(ValueError):
        train(region, observations, config=config)
    with pytest.raises(ValueError):
        train(region, observations, distance_model=encoder)
    initial = train(region, observations, distance_model=encoder, config=config)

    continued = train(region, observations, state=initial)
    assert continued.history == initial.history

    assert continued.config == config
    np.testing.assert_array_equal(
        continued.selected_model.predict(region).values,
        initial.selected_model.predict(region).values,
    )
    with pytest.raises(ValueError):
        train(region, observations, state=initial, distance_model=encoder)


def test_training_configuration_holds_only_scientific_settings():
    # Eager debugging uses jax.disable_jit() rather than a configuration flag.
    with pytest.raises(TypeError):
        TrainingConfig(loss=Log1pLoss(), jit=False)  # type: ignore[call-arg]
