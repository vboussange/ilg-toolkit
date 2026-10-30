"""Tests of training configurations and the inputs training requires."""

from typing import Any

import numpy as np
import optax
import pytest
from ilg_toolkit import (
    Log1pLoss,
    PairwiseObservations,
    ResistanceSolverConfig,
    train,
    TrainingConfig,
)


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


@pytest.mark.parametrize("solver", ["cg", "amg"])
@pytest.mark.parametrize(
    "options",
    [
        dict(rtol=True),
        dict(atol=float("nan")),
        dict(rtol=0, atol=0),
        dict(max_steps=1.5),
        dict(max_steps=False),
    ],
)
def test_iterative_solver_settings_reject_invalid_options(solver, options):
    with pytest.raises(ValueError):
        ResistanceSolverConfig(solver=solver, **options)


@pytest.mark.parametrize("solver", ["cg", "amg"])
def test_iterative_solvers_default_to_their_tolerances_and_iteration_limit(solver):
    config = ResistanceSolverConfig(solver=solver)
    assert (config.rtol, config.atol, config.max_steps) == (1e-6, 1e-6, 1000)
    assert config == ResistanceSolverConfig(solver=solver, rtol=1e-6, atol=1e-6, max_steps=1000)


def test_conjugate_gradient_is_the_default_solver():
    assert ResistanceSolverConfig().solver == "cg"


@pytest.mark.parametrize("solver", ["CG", "direct", "", None, 1])
def test_solver_settings_reject_an_unknown_solver(solver):
    with pytest.raises(ValueError, match="solver must be one of 'cg', 'amg' or 'cudss'"):
        ResistanceSolverConfig(solver=solver)


@pytest.mark.parametrize(
    ("name", "value"), [("rtol", 1e-6), ("atol", 1e-6), ("max_steps", 1000), ("rtol", 0)]
)
def test_direct_solver_rejects_iterative_settings_by_name(name, value):
    with pytest.raises(ValueError, match=f"{name} does not apply to solver 'cudss'"):
        ResistanceSolverConfig(solver="cudss", **{name: value})


def test_direct_solver_settings_construct_without_a_gpu_backend():
    config = ResistanceSolverConfig(solver="cudss")
    assert (config.rtol, config.atol, config.max_steps) == (None, None, None)


def test_solver_settings_have_no_use_amg_option():
    with pytest.raises(TypeError):
        ResistanceSolverConfig(use_amg=True)  # type: ignore[call-arg]
