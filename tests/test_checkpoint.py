"""Disk continuation checked through the public fit/checkpoint workflow."""

from typing import TypedDict

import jax
import numpy as np
import pytest

from ilg_toolkit import (
    PairwiseObservations,
    RegionBatch,
    TargetSpec,
    TrainingConfig,
    fit,
    load_checkpoint,
    save_checkpoint,
)
from ilg_toolkit.models import UNetEmbeddingDistance
from ilg_toolkit.training import (
    ObservationCollection,
    PartitionCollection,
    RegionCollection,
    TrainingState,
)


class _ValidationOptions(TypedDict, total=False):
    validation: tuple[RegionCollection, ObservationCollection]
    partition: PartitionCollection
    validation_partition: PartitionCollection


class _FitOptions(_ValidationOptions, total=False):
    region: RegionCollection
    observations: ObservationCollection
    model: UNetEmbeddingDistance
    config: TrainingConfig
    state: TrainingState


def checkpoint_problem(name="alpine", seed=14):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(4, 4, 2)).astype(np.float32)
    positions = np.array([[0, 0], [0, 3], [1, 1], [2, 2], [3, 0], [3, 3]])
    ids = tuple(f"population-{i}" for i in range(len(positions)))
    region = RegionBatch(name, features, ids, positions, feature_names=("elevation", "canopy"))
    pairs = [(ids[i], ids[j]) for i, j in zip(*np.triu_indices(len(ids), 1), strict=True)]
    observations = PairwiseObservations.from_pairs(
        pairs,
        rng.uniform(0.2, 1.5, size=len(pairs)),
        target=TargetSpec("synthetic divergence", units="index"),
        sampling_unit_ids=ids,
    )
    return region, observations


def checkpoint_model():
    return UNetEmbeddingDistance(
        2,
        patch_size=1,
        base_channels=2,
        embedding_dim=2,
        dropout=0.2,
        key=jax.random.PRNGKey(8),
    )


def test_checkpoint_loader_retains_schema_one_model_state(tmp_path):
    """Canonical Python state names continue to load the established disk format."""
    import json
    import zipfile

    region, observations = checkpoint_problem()
    result = fit(
        region,
        observations,
        model=checkpoint_model(),
        config=TrainingConfig(epochs=0),
    )
    assert result.state is not None
    path = tmp_path / "schema-one-checkpoint.ilg"
    save_checkpoint(path, result.state)
    with zipfile.ZipFile(path) as archive:
        content = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(content["manifest.json"])
    assert manifest["schema"] == 1
    payload = manifest["payload"]
    # Only Python API names changed; the numeric checkpoint schema did not.
    payload["latest_predictor"] = payload.pop("latest_model", payload.get("latest_predictor"))
    payload["best_predictor"] = payload.pop("best_model", payload.get("best_predictor"))
    content["manifest.json"] = json.dumps(manifest).encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in content.items():
            archive.writestr(name, value)
    restored = load_checkpoint(path)
    assert restored.epoch == 0
    np.testing.assert_array_equal(
        restored.latest_model.predict(region).values, result.model.predict(region).values
    )


def assert_same_state(actual, expected):
    assert actual.history == expected.history
    assert actual.epoch == expected.epoch
    assert actual.step == expected.step
    assert actual.selected_epoch == expected.selected_epoch
    assert actual.best_loss == expected.best_loss
    assert actual.selection == expected.selection
    assert actual.data_identity == expected.data_identity
    assert actual.schedule_state == expected.schedule_state
    assert actual.stopping_state == expected.stopping_state
    for left, right in zip(
        jax.tree.leaves(
            (actual.encoder, actual.raw_variances, actual.optimizer_state, actual.rng_key)
        ),
        jax.tree.leaves(
            (expected.encoder, expected.raw_variances, expected.optimizer_state, expected.rng_key)
        ),
        strict=True,
    ):
        np.testing.assert_array_equal(left, right)


def test_direct_disk_resume_matches_uninterrupted_dropout_training(tmp_path):
    region, observations = checkpoint_problem()
    config = TrainingConfig(epochs=3, seed=21, learning_rate=0.002)
    uninterrupted = fit(region, observations, model=checkpoint_model(), config=config)
    assert uninterrupted.state is not None
    path = tmp_path / "direct.ilg"
    epochs = []

    def interrupt(state):
        save_checkpoint(path, state)
        epochs.append(state.epoch)
        if state.epoch == 1:
            raise InterruptedError("synthetic interruption after complete checkpoint")

    with pytest.raises(InterruptedError, match="synthetic interruption"):
        fit(region, observations, model=checkpoint_model(), config=config, on_epoch=interrupt)
    assert epochs == [0, 1]
    restored = load_checkpoint(path)
    assert restored.epoch == 1
    from dataclasses import replace

    with pytest.raises(ValueError, match="epoch budget"):
        fit(region, observations, state=restored, config=replace(config, epochs=2))
    resumed = fit(region, observations, state=restored, config=config)
    assert resumed.state is not None
    assert_same_state(resumed.state, uninterrupted.state)
    np.testing.assert_array_equal(
        resumed.model.predict(region).values, uninterrupted.model.predict(region).values
    )
    assert all(record.validation_loss is None for record in resumed.history)
    assert resumed.selection == "final"


@pytest.mark.parametrize("objective", ["direct_log1p", "mlpe"])
def test_chunked_resnet_disk_resume_and_calibration_retain_execution_setting(tmp_path, objective):
    from dataclasses import replace

    from test_conductance import dense_resistance

    from ilg_toolkit import load_model, save_model
    from ilg_toolkit.models import ResNet9Conductance

    features = np.random.default_rng(4).normal(size=(8, 8, 2)).astype(np.float32)
    region = RegionBatch(
        "resnet-resume",
        features,
        ("south", "north", "west", "east"),
        np.array([[0, 0], [0, 7], [7, 0], [7, 7]]),
    )
    observations = PairwiseObservations.from_matrix(
        region.sampling_unit_ids,
        dense_resistance(np.array([[1.0, 1.4], [0.8, 2.0]]), np.arange(4)),
        target=TargetSpec("synthetic dissimilarity", units="index"),
    )
    encoder = ResNet9Conductance(2, patch_size=4, patch_batch_size=3, key=jax.random.key(6))
    config = TrainingConfig(objective=objective, epochs=2, seed=7, learning_rate=1e-4)
    with jax.enable_x64():
        uninterrupted = fit(region, observations, model=encoder, config=config)
        partial = fit(region, observations, model=encoder, config=replace(config, epochs=1))
        assert uninterrupted.state is not None and partial.state is not None
        path = tmp_path / "resnet.ilg"
        save_checkpoint(path, partial.state)
        restored = load_checkpoint(path)
        for saved in (restored.encoder, restored.latest_model.encoder, restored.best_model.encoder):
            assert isinstance(saved, ResNet9Conductance)
            assert saved.patch_batch_size == 3
        resumed = fit(region, observations, state=restored, config=config)
        assert resumed.state is not None
        assert_same_state(resumed.state, uninterrupted.state)
        np.testing.assert_array_equal(
            resumed.model.predict(region).values, uninterrupted.model.predict(region).values
        )
        # A continuation always keeps its saved execution setting, not a replacement model.
        with pytest.raises(ValueError, match="omit model"):
            fit(
                region,
                observations,
                state=restored,
                config=config,
                model=ResNet9Conductance(
                    2, patch_size=4, patch_batch_size=1, key=jax.random.key(6)
                ),
            )
        if objective == "direct_log1p":
            calibrated = resumed.model.recalibrate(region, observations)
            assert isinstance(calibrated.encoder, ResNet9Conductance)
            assert calibrated.encoder.patch_batch_size == 3
            np.testing.assert_array_equal(
                calibrated.conductance_surface(region), resumed.model.conductance_surface(region)
            )
            save_model(path, calibrated)
            loaded = load_model(path)
            assert isinstance(loaded.encoder, ResNet9Conductance)
            assert loaded.encoder.patch_batch_size == 3
            np.testing.assert_array_equal(
                loaded.predict(region).values, calibrated.predict(region).values
            )


def test_shared_mlpe_disk_resume_preserves_nuisance_heads_and_selected_model(tmp_path):
    from dataclasses import replace

    from ilg_toolkit import ObservationPartition

    problems = (checkpoint_problem(), checkpoint_problem("valley", seed=71))
    regions = tuple(problem[0] for problem in problems)
    observations = tuple(problem[1] for problem in problems)
    training = tuple(
        ObservationPartition(region.name, obs.observed_pairs[:-2], role="training")
        for region, obs in zip(regions, observations, strict=True)
    )
    validation = tuple(
        ObservationPartition(region.name, obs.observed_pairs[-2:], role="validation")
        for region, obs in zip(regions, observations, strict=True)
    )
    config = TrainingConfig(
        epochs=2,
        objective="mlpe",
        learning_rate=0.002,
        seed=21,
        mlpe_initial_variances=(0.05, 0.1),
    )
    arguments = _ValidationOptions(
        partition={part.region_name: part for part in training},
        validation=(regions, observations),
        validation_partition={part.region_name: part for part in validation},
    )
    with jax.enable_x64():
        uninterrupted = fit(
            regions, observations, model=checkpoint_model(), config=config, **arguments
        )
        assert uninterrupted.state is not None
        partial = fit(
            regions,
            observations,
            model=checkpoint_model(),
            config=replace(config, epochs=1),
            **arguments,
        )
        assert partial.state is not None
        path = tmp_path / "mlpe.ilg"
        save_checkpoint(path, partial.state)
        restored = load_checkpoint(path)
        assert_same_state(restored, partial.state)
        assert restored.latest_model.calibrations == partial.latest_model.calibrations
        assert restored.best_model.calibrations == partial.best_model.calibrations
        assert restored.latest_model.training_pairs == partial.latest_model.training_pairs
        assert restored.latest_model.validation_pairs == partial.latest_model.validation_pairs
        resumed = fit(
            dict(zip(reversed([r.name for r in regions]), reversed(regions), strict=True)),
            dict(zip(reversed([r.name for r in regions]), reversed(observations), strict=True)),
            state=restored,
            config=config,
            **arguments,
        )
        assert resumed.state is not None
        assert_same_state(resumed.state, uninterrupted.state)
        assert resumed.model.calibrations == uninterrupted.model.calibrations
        assert resumed.latest_model.calibrations == uninterrupted.latest_model.calibrations
        for region in regions:
            np.testing.assert_array_equal(
                resumed.model.predict(region).values,
                uninterrupted.model.predict(region).values,
            )


def test_zero_update_and_worsening_mlpe_resume_retain_previous_best_heads(tmp_path):
    from dataclasses import replace

    from ilg_toolkit import ObservationPartition

    problems = (checkpoint_problem(), checkpoint_problem("valley", seed=71))
    regions = tuple(problem[0] for problem in problems)
    observations = tuple(problem[1] for problem in problems)
    arguments = _ValidationOptions(
        partition={
            r.name: ObservationPartition(r.name, o.observed_pairs[:-2], role="training")
            for r, o in zip(regions, observations, strict=True)
        },
        validation=(regions, observations),
        validation_partition={
            r.name: ObservationPartition(r.name, o.observed_pairs[-2:], role="validation")
            for r, o in zip(regions, observations, strict=True)
        },
    )
    config = TrainingConfig(
        epochs=1, objective="mlpe", seed=21, learning_rate=0.002, mlpe_initial_variances=(0.05, 0.1)
    )
    with jax.enable_x64():
        partial = fit(regions, observations, model=checkpoint_model(), config=config, **arguments)
        assert partial.state is not None
        assert partial.selected_epoch == 0
        path = tmp_path / "selected.ilg"
        save_checkpoint(path, partial.state)
        restored = load_checkpoint(path)
        callbacks = []
        unchanged = fit(
            regions, observations, state=restored, on_epoch=callbacks.append, **arguments
        )
        assert unchanged.state is not None
        assert callbacks == []
        assert_same_state(unchanged.state, partial.state)
        continued = fit(
            regions, observations, state=restored, config=replace(config, epochs=2), **arguments
        )
        assert continued.state is not None
        assert continued.history[-1].validation_loss is not None
        assert continued.history[-1].validation_loss > partial.state.best_loss
        assert continued.selected_epoch == 0
        assert continued.model.calibrations == partial.model.calibrations
        for r in regions:
            np.testing.assert_array_equal(
                continued.model.predict(r).values, partial.model.predict(r).values
            )
        assert continued.latest_model.calibrations != partial.model.calibrations
        assert continued.state.epoch == 2


def test_checkpoint_rejects_optimizer_precision_that_differs_from_model(tmp_path):
    from dataclasses import replace

    with jax.enable_x64():
        region, observations = checkpoint_problem()
        result = fit(
            region, observations, model=checkpoint_model(), config=TrainingConfig(epochs=0)
        )
        assert result.state is not None
        changed_optimizer = jax.tree.map(
            lambda value: value.astype(np.float64) if value.dtype == np.float32 else value,
            result.state.optimizer_state,
        )
        path = tmp_path / "mixed-precision.ilg"
        save_checkpoint(path, replace(result.state, optimizer_state=changed_optimizer))
        with pytest.raises(ValueError, match="[Oo]ptimizer.*dtype"):
            load_checkpoint(path)


def test_interrupted_replacement_retains_previous_complete_generation(tmp_path, monkeypatch):
    import os
    from pathlib import Path

    region, observations = checkpoint_problem()
    completed = []
    config = TrainingConfig(epochs=1, seed=21, learning_rate=0.002)
    result = fit(
        region, observations, model=checkpoint_model(), config=config, on_epoch=completed.append
    )
    assert result.state is not None
    path = tmp_path / "atomic.ilg"
    save_checkpoint(path, completed[0])
    original = path.read_bytes()

    def interrupt_replace(source, destination):
        assert Path(source).is_file()
        assert Path(source).stat().st_size > 0
        raise InterruptedError("simulated filesystem interruption")

    with monkeypatch.context() as fault:
        fault.setattr(os, "replace", interrupt_replace)
        with pytest.raises(InterruptedError, match="filesystem interruption"):
            save_checkpoint(path, result.state)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
    restored = load_checkpoint(path)
    assert_same_state(restored, completed[0])
    continued = fit(region, observations, state=restored, config=config)
    assert continued.state is not None
    assert_same_state(continued.state, result.state)


def test_disk_continuation_rejects_changed_inputs_config_model_and_partitions(tmp_path):
    from dataclasses import replace

    from ilg_toolkit import ObservationPartition, ResistanceSolverConfig

    region, observations = checkpoint_problem()
    training = ObservationPartition(region.name, observations.observed_pairs[:-2], role="training")
    config = TrainingConfig(epochs=1, seed=21, learning_rate=0.002)
    result = fit(region, observations, model=checkpoint_model(), config=config, partition=training)
    assert result.state is not None
    path = tmp_path / "compatible.ilg"
    save_checkpoint(path, result.state)
    state = load_checkpoint(path)
    base = _FitOptions(
        region=region,
        observations=observations,
        state=state,
        config=replace(config, epochs=2),
        partition=training,
    )
    changed_labels = PairwiseObservations.from_pairs(
        observations.observed_pairs,
        observations.observed_values + 0.1,
        target=observations.target,
        sampling_unit_ids=region.sampling_unit_ids,
    )
    changed_target = replace(observations, target=TargetSpec("different target", units="index"))
    changes: list[_FitOptions] = [
        _FitOptions(region=replace(region, features=region.feature_array + 0.1)),
        _FitOptions(
            region=replace(
                region, features=region.feature_array, feature_names=("canopy", "elevation")
            )
        ),
        _FitOptions(region=replace(region, grid_positions=region.grid_positions[::-1])),
        _FitOptions(region=replace(region, sampling_unit_kinds=("individual",) * 6)),
        _FitOptions(observations=changed_labels),
        _FitOptions(observations=changed_target),
        _FitOptions(
            partition=ObservationPartition(region.name, training.pairs[:-1], role="training")
        ),
        _FitOptions(
            partition=ObservationPartition(region.name, training.pairs, role="calibration")
        ),
        _FitOptions(config=replace(config, epochs=2, learning_rate=0.003)),
        _FitOptions(config=replace(config, epochs=2, solver=ResistanceSolverConfig(rtol=1e-8))),
        _FitOptions(config=replace(config, epochs=0)),
        _FitOptions(model=checkpoint_model()),
    ]
    callbacks = []
    for change in changes:
        with pytest.raises(ValueError):
            fit(**(base | change), on_epoch=callbacks.append)
    assert callbacks == []
    extended = fit(**base)
    assert extended.state is not None
    assert extended.state.epoch == 2


def test_checkpoint_load_refuses_incomplete_schema_runtime_and_progress(tmp_path):
    import json
    import zipfile

    region, observations = checkpoint_problem()
    callbacks = []
    result = fit(
        region,
        observations,
        model=checkpoint_model(),
        config=TrainingConfig(epochs=0),
        on_epoch=callbacks.append,
    )
    assert result.state is not None
    assert [state.epoch for state in callbacks] == [0]
    original = tmp_path / "complete.ilg"
    save_checkpoint(original, result.state)
    with zipfile.ZipFile(original) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}

    def corrupt(name, change, message):
        manifest = json.loads(entries["manifest.json"])
        change(manifest)
        path = tmp_path / name
        with zipfile.ZipFile(path, "w") as archive:
            for entry, content in entries.items():
                archive.writestr(
                    entry, json.dumps(manifest) if entry == "manifest.json" else content
                )
        with pytest.raises(ValueError, match=message):
            load_checkpoint(path)

    corrupt("version.ilg", lambda m: m["payload"].update(checkpoint_version=2), "schema")
    corrupt("missing.ilg", lambda m: m["payload"].pop("best_predictor"), "incomplete")
    corrupt("runtime.ilg", lambda m: m["payload"]["resume_runtime"].update(jax="0.0"), "runtime")
    corrupt("progress.ilg", lambda m: m["payload"].update(step=1), "progress")
    corrupt("history.ilg", lambda m: m["payload"].update(history=[]), "history")
    corrupt("kind.ilg", lambda m: m.update(kind="model"), "training_checkpoint")


def test_checkpoint_requires_calibration_for_every_training_region(tmp_path):
    from dataclasses import replace

    from ilg_toolkit import ArtifactError

    region, observations = checkpoint_problem()
    other = replace(region, name="second-region")
    with jax.enable_x64():
        result = fit(
            (region, other),
            (observations, observations),
            model=checkpoint_model(),
            config=TrainingConfig(epochs=0, objective="mlpe"),
        )
        assert result.state is not None
        partial = replace(
            result.model, calibrations={region.name: result.model.calibrations[region.name]}
        )
        state = replace(result.state, latest_model=partial, best_model=partial)
        with pytest.raises(ArtifactError, match="missing regional MLPE calibration"):
            save_checkpoint(tmp_path / "incomplete-calibration.ilg", state)
