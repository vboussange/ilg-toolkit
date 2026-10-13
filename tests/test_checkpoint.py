"""Disk continuation checked through the public fit/checkpoint workflow."""

from dataclasses import replace
from typing import TypedDict

import equinox as eqx
import jax
import numpy as np
import pytest
from artifact_helpers import checkpoint_problem, distance_model

from ilg_toolkit import TrainingConfig, fit, load_checkpoint, save_checkpoint
from ilg_toolkit.training import ObservationCollection, PartitionCollection, RegionCollection


class _ValidationOptions(TypedDict):
    partition: PartitionCollection
    validation: tuple[RegionCollection, ObservationCollection]
    validation_partition: PartitionCollection


def assert_same_state(actual, expected):
    # Includes latest/selected parameters, nuisance rows, moments, RNG and provenance.
    assert eqx.tree_equal(actual, expected)


def test_native_checkpoint_restores_latest_model_and_history(tmp_path):
    region, observations = checkpoint_problem()
    result = fit(
        [region],
        {region.name: observations},
        distance_model=distance_model(),
        config=TrainingConfig(objective="direct_log1p", steps=0),
    )
    path = tmp_path / "state.ilg"
    save_checkpoint(path, result)
    restored = load_checkpoint(path, distance_model=distance_model(seed=7))
    assert_same_state(restored, result)
    assert restored.step == restored.selected_step == 0
    for species in region.sampling_units:
        np.testing.assert_array_equal(
            restored.model.predict(region, species=species)[species].values,
            result.model.predict(region, species=species)[species].values,
        )


@pytest.mark.parametrize("family", ["custom", "unet", "resnet", "loglinear"])
@pytest.mark.parametrize("objective", ["direct_log1p", "mlpe"])
def test_between_evaluations_disk_resume_matches_uninterrupted(tmp_path, family, objective):
    from ilg_toolkit import load_model, save_model

    first, first_observed = checkpoint_problem()
    second, second_observed = checkpoint_problem("valley", seed=71)
    regions = [first, second]
    observations = {first.name: first_observed, second.name: second_observed}
    model = distance_model(family)
    config = TrainingConfig(
        objective=objective,
        steps=2,
        evaluation_interval=2,
        region_batch_size=1,
        species_batch_size=1,
        species_chunk_size=1,
        seed=21,
        learning_rate=0.001,
    )
    uninterrupted = fit(regions, observations, distance_model=model, config=config)
    path = tmp_path / "sampled.ilg"
    callbacks = []
    captured = []

    def interrupt(state):
        callbacks.append(state.step)
        assert [record.step for record in state.history] == [0]
        assert state.selected_step == 0
        save_checkpoint(path, state)
        captured.append(state)
        raise InterruptedError("after checkpoint between evaluations")

    with pytest.raises(InterruptedError, match="between evaluations"):
        fit(regions, observations, distance_model=model, config=config, on_step=interrupt)
    assert callbacks == [1]
    restored = load_checkpoint(path, distance_model=distance_model(family, seed=99))
    assert_same_state(restored, captured[0])
    assert not eqx.tree_equal(restored.latest_distance_model, restored.best_model.distance_model)
    if objective == "mlpe":
        assert restored.raw_variances is not None and restored.selected_raw_variances is not None
        assert restored.raw_variances.shape == restored.selected_raw_variances.shape == (4, 2)
        assert not np.array_equal(restored.raw_variances, restored.selected_raw_variances)
    resumed = fit(regions, observations, state=restored)
    assert_same_state(resumed, uninterrupted)
    assert resumed.step == resumed.selected_step == 2
    assert [record.step for record in resumed.history] == [0, 2]
    for region in regions:
        for species in region.sampling_units:
            assert eqx.tree_equal(
                resumed.model.predict(region, species=species),
                uninterrupted.model.predict(region, species=species),
            )
    inference_path = tmp_path / "inference.ilg"
    save_model(inference_path, resumed.model)
    loaded = load_model(inference_path, distance_model=model)
    assert eqx.tree_equal(loaded, resumed.model)


@pytest.fixture
def initial_state():
    region, observed = checkpoint_problem()
    state = fit(
        [region],
        {region.name: observed},
        distance_model=distance_model(),
        config=TrainingConfig(objective="direct_log1p", steps=0),
    )
    return state, region, observed


def test_checkpoint_rejects_incoherent_progress(initial_state, tmp_path):
    from ilg_toolkit import ArtifactError

    state, _, _ = initial_state
    invalid = eqx.tree_at(lambda value: value.step, state, 1)
    with pytest.raises(ArtifactError, match="step|progress"):
        save_checkpoint(tmp_path / "invalid.ilg", invalid)


def test_sparse_history_preserves_old_selected_snapshot_and_missing_coverage(tmp_path):
    from ilg_toolkit import ObservationPartition, PairwiseObservations

    region, observed = checkpoint_problem()
    model = distance_model()
    training = {
        name: ObservationPartition(region.name, name, value.observed_pairs[:-2], "training")
        for name, value in observed.items()
    }
    raw = model.predict_raw_distances(region, species="fox")["fox"]
    validation = PairwiseObservations.from_pairs(
        raw.pairs,
        raw.values,
        target=observed["fox"].target,
        sampling_unit_ids=region.sampling_units["fox"].ids,
    )
    options = _ValidationOptions(
        partition={region.name: training},
        validation=([region], {region.name: {"fox": validation}}),
        validation_partition={
            region.name: {
                "fox": ObservationPartition(
                    region.name, "fox", validation.observed_pairs[-2:], "validation"
                )
            }
        },
    )
    config = TrainingConfig(
        objective="direct_log1p",
        steps=4,
        evaluation_interval=3,
        seed=21,
        region_batch_size=1,
        species_batch_size=1,
    )
    expected = fit(
        [region], {region.name: observed}, distance_model=model, config=config, **options
    )
    path = tmp_path / "selected.ilg"

    def interrupt(state):
        if state.step == 2:
            save_checkpoint(path, state)
            raise InterruptedError("between sparse evaluations")

    with pytest.raises(InterruptedError):
        fit(
            [region],
            {region.name: observed},
            distance_model=model,
            config=config,
            on_step=interrupt,
            **options,
        )
    restored = load_checkpoint(path, distance_model=model)
    assert restored.step == 2
    assert restored.selected_step == 0
    assert [record.step for record in restored.history] == [0]
    resumed = fit([region], {region.name: observed}, state=restored, **options)
    assert_same_state(resumed, expected)
    assert resumed.selected_step == 0
    assert [record.step for record in resumed.history] == [0, 3, 4]
    assert resumed.history[-1].validation_missing_species == ("lynx",)
    assert resumed.history[-1].validation_missing_groups == ((region.name, "lynx"),)
    callbacks = []
    unchanged = fit(
        [region], {region.name: observed}, state=resumed, on_step=callbacks.append, **options
    )
    assert callbacks == []
    assert_same_state(unchanged, resumed)


@pytest.mark.parametrize(
    "setting,value",
    [
        ("region_batch_size", 1),
        ("species_batch_size", 1),
        ("species_chunk_size", 2),
        ("evaluation_interval", 3),
        ("learning_rate", 0.01),
        ("seed", 77),
        ("jit", False),
    ],
)
def test_loaded_checkpoint_rejects_changed_policy(initial_state, tmp_path, setting, value):
    state, region, observed = initial_state
    path = tmp_path / "policy.ilg"
    save_checkpoint(path, state)
    loaded = load_checkpoint(path, distance_model=distance_model())
    callbacks = []
    with pytest.raises(ValueError, match="policy"):
        fit(
            [region],
            {region.name: observed},
            state=loaded,
            config=replace(state.config, steps=1, **{setting: value}),
            on_step=callbacks.append,
        )
    assert callbacks == []
    with pytest.raises(ValueError, match="omit distance_model"):
        fit([region], {region.name: observed}, state=loaded, distance_model=distance_model())


def test_complete_declared_units_and_measurements_survive_disk_contract(tmp_path):
    import coordax as cx

    from ilg_toolkit import SamplingUnits

    first, observed_first = checkpoint_problem()
    second, observed_second = checkpoint_problem("valley", seed=71)
    observed = {first.name: observed_first, second.name: {"fox": observed_second["fox"]}}
    state = fit(
        [first, second],
        observed,
        distance_model=distance_model(),
        config=TrainingConfig(objective="direct_log1p", steps=0),
    )
    path = tmp_path / "declarations.ilg"
    save_checkpoint(path, state)
    loaded = load_checkpoint(path, distance_model=distance_model())
    assert loaded.input_contract == state.input_contract
    units = second.sampling_units["lynx"]
    assert second.feature_names is not None
    changes = [
        replace(
            second,
            feature_names=tuple(reversed(second.feature_names)),
            features=cx.field(np.asarray(second.feature_array), "row", "column", "feature"),
        ),
        replace(second, features=cx.field(second.feature_array + 0.01, "row", "column", "feature")),
        replace(
            second,
            features=cx.field(second.feature_array.astype(np.float64), "row", "column", "feature"),
        ),
        replace(
            second,
            sampling_units={
                **second.sampling_units,
                "lynx": SamplingUnits(tuple(reversed(units.ids)), units.grid_positions),
            },
        ),
        replace(
            second,
            sampling_units={
                **second.sampling_units,
                "lynx": replace(units, kinds=("individual",) * len(units.ids)),
            },
        ),
        replace(
            second,
            sampling_units={
                **second.sampling_units,
                "lynx": replace(units, grid_positions=np.asarray(units.grid_positions)[:, ::-1]),
            },
        ),
    ]
    callbacks = []
    for changed in changes:
        with pytest.raises(ValueError, match="contract|identit|feature"):
            fit([first, changed], observed, state=loaded, on_step=callbacks.append)
    changed_observed = {
        **observed,
        first.name: {
            **observed_first,
            "fox": replace(observed_first["fox"], values=observed_first["fox"].values * 1.1),
        },
    }
    with pytest.raises(ValueError, match="contract"):
        fit([first, second], changed_observed, state=loaded, on_step=callbacks.append)
    assert callbacks == []


def test_checkpoint_rejects_optimizer_precision_that_differs_from_model(tmp_path, initial_state):
    import jax.numpy as jnp

    from ilg_toolkit import ArtifactError

    state, _, _ = initial_state
    optimizer = jax.tree.map(
        lambda value: value.astype(jnp.float32) if value.dtype == np.float64 else value,
        state.optimizer_state,
    )
    invalid = eqx.tree_at(lambda value: value.optimizer_state, state, optimizer)
    path = tmp_path / "precision.ilg"
    with pytest.raises(ArtifactError, match="dtype"):
        save_checkpoint(path, invalid)
        load_checkpoint(path, distance_model=state.distance_model)


def test_mlpe_checkpoint_requires_original_precision(tmp_path):
    from ilg_toolkit import ArtifactError

    region, observed = checkpoint_problem()
    state = fit(
        [region],
        {region.name: observed},
        distance_model=distance_model(),
        config=TrainingConfig(objective="mlpe", steps=0),
    )
    path = tmp_path / "mlpe.ilg"
    save_checkpoint(path, state)
    with jax.enable_x64(False):
        with pytest.raises(ArtifactError, match="JAX_ENABLE_X64"):
            load_checkpoint(path, distance_model=state.distance_model)
        assert not jax.config.read("jax_enable_x64")


def test_checkpoint_rejects_truncation_wrong_kind_and_registry(tmp_path, initial_state):
    from ilg_toolkit import ArtifactError, load_model

    state, _, _ = initial_state
    path = tmp_path / "state.ilg"
    save_checkpoint(path, state)
    with pytest.raises(ArtifactError, match="model artifact"):
        load_model(path, distance_model=state.distance_model)
    with pytest.raises(ArtifactError, match="species"):
        load_checkpoint(path, distance_model=distance_model(species=("lynx", "fox")))
    path.write_bytes(path.read_bytes()[:-11])
    with pytest.raises(ArtifactError, match="payload"):
        load_checkpoint(path, distance_model=state.distance_model)


def test_checkpoint_preserves_declared_order_when_consumed_pairs_are_aligned(tmp_path):
    from ilg_toolkit import ObservationPartition, PairwiseObservations

    region, original = checkpoint_problem()
    observations = {
        species: PairwiseObservations.from_pairs(
            tuple(reversed(value.observed_pairs)),
            value.observed_values[::-1],
            sampling_unit_ids=tuple(reversed(region.sampling_units[species].ids)),
            target=value.target,
        )
        for species, value in original.items()
    }
    partitions = {
        species: ObservationPartition(
            region.name, species, tuple(reversed(value.observed_pairs[:-2])), "training"
        )
        for species, value in observations.items()
    }
    config = TrainingConfig(objective="direct_log1p", steps=2, evaluation_interval=2)
    expected = fit(
        [region],
        {region.name: observations},
        distance_model=distance_model(),
        config=config,
        partition={region.name: partitions},
    )
    path = tmp_path / "aligned.ilg"
    captured = []

    def interrupt(state):
        captured.append(state)
        save_checkpoint(path, state)
        raise InterruptedError("aligned pair checkpoint")

    with pytest.raises(InterruptedError):
        fit(
            [region],
            {region.name: observations},
            distance_model=distance_model(),
            config=config,
            on_step=interrupt,
            partition={region.name: partitions},
        )
    loaded = load_checkpoint(path, distance_model=distance_model())
    assert loaded.input_contract == captured[0].input_contract
    assert_same_state(loaded, captured[0])
    resumed = fit(
        [region], {region.name: observations}, state=loaded, partition={region.name: partitions}
    )
    assert_same_state(resumed, expected)
