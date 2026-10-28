"""Tests of training configurations and the inputs training requires."""

import numpy as np
import optax
import pytest
from ilg_toolkit import Log1pLoss, PairwiseObservations, train, TrainingConfig


@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, "2"])
@pytest.mark.parametrize(
    "name", ["region_batch_size", "species_batch_size", "species_microbatch_size"]
)
def test_batch_sizes_reject_invalid_counts(value, name):
    with pytest.raises(ValueError):
        TrainingConfig(loss=Log1pLoss(), **{name: value})


def test_fresh_training_requires_declared_distance_model_and_loss_while_continuation_reuses_them(
    observation_problem,
):
    region, target, distance_model = observation_problem()
    observations = PairwiseObservations.from_pairs([("unit/7", "unit/20")], [0.5], target=target)
    omitted = {}
    with pytest.raises(TypeError):
        TrainingConfig(**omitted)
    config = TrainingConfig(loss=Log1pLoss(), steps=0)
    with pytest.raises(ValueError):
        train(region, observations, config=config)
    with pytest.raises(ValueError):
        train(region, observations, distance_model=distance_model)
    initial = train(region, observations, distance_model=distance_model, config=config)

    continued = train(region, observations, state=initial)
    assert continued.history == initial.history

    assert continued.config == config
    np.testing.assert_array_equal(
        continued.selected_model.predict(region).values,
        initial.selected_model.predict(region).values,
    )
    with pytest.raises(ValueError):
        train(region, observations, state=initial, distance_model=distance_model)


def test_training_configuration_has_no_jit_flag():
    # Eager debugging uses jax.disable_jit() rather than a configuration flag.
    with pytest.raises(TypeError):
        TrainingConfig(loss=Log1pLoss(), jit=False)  # type: ignore[call-arg]


def test_any_optax_optimizer_configures_training_and_equality_ignores_it():
    default = TrainingConfig(loss=Log1pLoss())
    assert isinstance(default.optimizer, optax.GradientTransformation)
    assert TrainingConfig(loss=Log1pLoss(), optimizer=optax.lbfgs()) == default
    assert TrainingConfig(loss=Log1pLoss(), optimizer=optax.sgd(0.1)) == default
    with pytest.raises(ValueError, match="optimizer"):
        TrainingConfig(loss=Log1pLoss(), optimizer=0.001)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        TrainingConfig(loss=Log1pLoss(), learning_rate=0.001)  # type: ignore[call-arg]


def test_gradient_tolerance_is_optional_finite_and_nonnegative():
    assert TrainingConfig(loss=Log1pLoss()).gradient_tolerance is None
    tolerance = TrainingConfig(loss=Log1pLoss(), gradient_tolerance=np.float64(1e-3))
    assert type(tolerance.gradient_tolerance) is float
    assert TrainingConfig(loss=Log1pLoss(), gradient_tolerance=0).gradient_tolerance == 0.0
    assert tolerance != TrainingConfig(loss=Log1pLoss(), gradient_tolerance=1e-4)
    for invalid in (-1e-3, float("nan"), float("inf"), True, "1e-3"):
        with pytest.raises(ValueError, match="gradient_tolerance"):
            TrainingConfig(loss=Log1pLoss(), gradient_tolerance=invalid)  # type: ignore[arg-type]
