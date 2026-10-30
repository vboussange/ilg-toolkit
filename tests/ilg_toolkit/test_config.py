"""Tests of training configurations and the inputs training requires."""

from dataclasses import replace
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from ilg_toolkit import Log1pLoss, PairwiseObservations, RidgePenalty, train, TrainingConfig


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


def test_integer_settings_accept_numpy_integers():
    options: dict[str, Any] = dict(steps=np.int64(5), seed=np.uint32(7), eval_every=np.int32(2))
    config = TrainingConfig(loss=Log1pLoss(), **options)
    assert (config.steps, config.seed, config.eval_every) == (5, 7, 2)
    assert all(type(value) is int for value in (config.steps, config.seed, config.eval_every))
    invalid: tuple[tuple[str, Any], ...] = (
        ("steps", -1),
        ("steps", True),
        ("seed", 2**32),
        ("seed", 1.0),
    )
    for name, value in invalid:
        options = {name: value}
        with pytest.raises(ValueError, match=name):
            TrainingConfig(loss=Log1pLoss(), **options)


@pytest.mark.parametrize("penalty", [0.1, "ridge", (1, 2)])
def test_a_penalty_that_is_not_callable_is_rejected(penalty: Any):
    with pytest.raises(ValueError, match="penalty"):
        TrainingConfig(loss=Log1pLoss(), penalty=penalty)


def test_configuration_equality_covers_the_penalty():
    def where(model):
        return model.coefficients

    ridge = RidgePenalty(prior_sd=1.5, where=where)
    config = TrainingConfig(loss=Log1pLoss(), penalty=ridge)
    assert config.penalty is ridge
    assert TrainingConfig(loss=Log1pLoss()).penalty is None
    assert config != TrainingConfig(loss=Log1pLoss())
    assert config == TrainingConfig(loss=Log1pLoss(), penalty=RidgePenalty(1.5, where))
    assert config != TrainingConfig(loss=Log1pLoss(), penalty=RidgePenalty(2.0, where))
    assert config != TrainingConfig(
        loss=Log1pLoss(), penalty=RidgePenalty(1.5, lambda model: model.coefficients)
    )
    assert replace(config, steps=7) == replace(
        TrainingConfig(loss=Log1pLoss(), penalty=ridge), steps=7
    )
    # Two separately written functions compare unequal.
    assert TrainingConfig(loss=Log1pLoss(), penalty=lambda model: 0.0) != TrainingConfig(
        loss=Log1pLoss(), penalty=lambda model: 0.0
    )


def test_configuration_stays_hashable_with_a_penalty_holding_arrays():
    class _Centered(eqx.Module):
        center: jax.Array

        def __call__(self, distance_model):
            return jnp.sum((distance_model.coefficients - self.center) ** 2)

    config = TrainingConfig(loss=Log1pLoss(), penalty=_Centered(jnp.ones(2)))
    assert hash(config) == hash(TrainingConfig(loss=Log1pLoss()))
