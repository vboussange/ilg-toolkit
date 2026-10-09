"""Disk continuation checked through the public fit/checkpoint workflow."""

import jax
import numpy as np
import pytest

from ilg_toolkit import (
    FitConfig,
    PairwiseObservations,
    PreparedRegion,
    TargetSpec,
    fit,
    load_checkpoint,
    save_checkpoint,
)
from ilg_toolkit.models import UNetEmbeddingDistance


def checkpoint_problem(name="alpine", seed=14):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(4, 4, 2)).astype(np.float32)
    positions = np.array([[0, 0], [0, 3], [1, 1], [2, 2], [3, 0], [3, 3]])
    ids = tuple(f"population-{i}" for i in range(len(positions)))
    region = PreparedRegion(name, features, ids, positions, feature_names=("elevation", "canopy"))
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
    config = FitConfig(epochs=3, seed=21, learning_rate=0.002)
    uninterrupted = fit(region, observations, model=checkpoint_model(), config=config)
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
    resumed = fit(region, observations, state=restored, config=config)
    assert_same_state(resumed.state, uninterrupted.state)
    np.testing.assert_array_equal(
        resumed.predictor.predict(region).values, uninterrupted.predictor.predict(region).values
    )
    assert all(record.validation_loss is None for record in resumed.history)
    assert resumed.selection == "final"


def test_shared_mlpe_disk_resume_preserves_nuisance_heads_and_selected_predictor(tmp_path):
    from dataclasses import replace

    from ilg_toolkit import ObservationPartition

    regions, observations = zip(
        checkpoint_problem(), checkpoint_problem("valley", seed=71), strict=True
    )
    training = tuple(
        ObservationPartition(region.name, obs.observed_pairs[:-2], role="training")
        for region, obs in zip(regions, observations, strict=True)
    )
    validation = tuple(
        ObservationPartition(region.name, obs.observed_pairs[-2:], role="validation")
        for region, obs in zip(regions, observations, strict=True)
    )
    config = FitConfig(
        epochs=2,
        objective="mlpe",
        learning_rate=0.002,
        seed=21,
        mlpe_initial_variances=(0.05, 0.1),
    )
    arguments = dict(
        partition={part.region_name: part for part in training},
        validation=(regions, observations),
        validation_partition={part.region_name: part for part in validation},
    )
    with jax.enable_x64():
        uninterrupted = fit(
            regions, observations, model=checkpoint_model(), config=config, **arguments
        )
        partial = fit(
            regions,
            observations,
            model=checkpoint_model(),
            config=replace(config, epochs=1),
            **arguments,
        )
        path = tmp_path / "mlpe.ilg"
        save_checkpoint(path, partial.state)
        restored = load_checkpoint(path)
        assert_same_state(restored, partial.state)
        assert restored.latest_predictor.calibrations == partial.latest_predictor.calibrations
        assert restored.best_predictor.calibrations == partial.best_predictor.calibrations
        assert restored.latest_predictor.training_pairs == partial.latest_predictor.training_pairs
        assert (
            restored.latest_predictor.validation_pairs == partial.latest_predictor.validation_pairs
        )
        resumed = fit(
            dict(zip(reversed([r.name for r in regions]), reversed(regions), strict=True)),
            dict(zip(reversed([r.name for r in regions]), reversed(observations), strict=True)),
            state=restored,
            config=config,
            **arguments,
        )
        assert_same_state(resumed.state, uninterrupted.state)
        assert resumed.predictor.calibrations == uninterrupted.predictor.calibrations
        assert resumed.latest_predictor.calibrations == uninterrupted.latest_predictor.calibrations
        for region in regions:
            np.testing.assert_array_equal(
                resumed.predictor.predict(region).values,
                uninterrupted.predictor.predict(region).values,
            )


def test_zero_update_and_worsening_mlpe_resume_retain_previous_best_heads(tmp_path):
    from dataclasses import replace

    from ilg_toolkit import ObservationPartition

    regions, observations = zip(
        checkpoint_problem(), checkpoint_problem("valley", seed=71), strict=True
    )
    arguments = dict(
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
    config = FitConfig(
        epochs=1, objective="mlpe", seed=21, learning_rate=0.002, mlpe_initial_variances=(0.05, 0.1)
    )
    with jax.enable_x64():
        partial = fit(regions, observations, model=checkpoint_model(), config=config, **arguments)
        assert partial.selected_epoch == 0
        path = tmp_path / "selected.ilg"
        save_checkpoint(path, partial.state)
        restored = load_checkpoint(path)
        callbacks = []
        unchanged = fit(
            regions, observations, state=restored, on_epoch=callbacks.append, **arguments
        )
        assert callbacks == []
        assert_same_state(unchanged.state, partial.state)
        continued = fit(
            regions, observations, state=restored, config=replace(config, epochs=2), **arguments
        )
        assert continued.history[-1].validation_loss > partial.state.best_loss
        assert continued.selected_epoch == 0
        assert continued.predictor.calibrations == partial.predictor.calibrations
        for r in regions:
            np.testing.assert_array_equal(
                continued.predictor.predict(r).values, partial.predictor.predict(r).values
            )
        assert continued.latest_predictor.calibrations != partial.predictor.calibrations
        assert continued.state.epoch == 2


def test_checkpoint_rejects_optimizer_precision_that_differs_from_model(tmp_path):
    from dataclasses import replace

    with jax.enable_x64():
        region, observations = checkpoint_problem()
        result = fit(region, observations, model=checkpoint_model(), config=FitConfig(epochs=0))
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
    config = FitConfig(epochs=1, seed=21, learning_rate=0.002)
    result = fit(
        region, observations, model=checkpoint_model(), config=config, on_epoch=completed.append
    )
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
    assert_same_state(continued.state, result.state)


def test_disk_continuation_rejects_changed_inputs_config_model_and_partitions(tmp_path):
    from dataclasses import replace

    from ilg_toolkit import ObservationPartition, SolverConfig

    region, observations = checkpoint_problem()
    training = ObservationPartition(region.name, observations.observed_pairs[:-2], role="training")
    config = FitConfig(epochs=1, seed=21, learning_rate=0.002)
    result = fit(region, observations, model=checkpoint_model(), config=config, partition=training)
    path = tmp_path / "compatible.ilg"
    save_checkpoint(path, result.state)
    state = load_checkpoint(path)
    base = dict(
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
    changes = [
        dict(region=replace(region, features=region.features + 0.1)),
        dict(region=replace(region, feature_names=("canopy", "elevation"))),
        dict(region=replace(region, grid_positions=region.grid_positions[::-1])),
        dict(region=replace(region, sampling_unit_kinds=("individual",) * 6)),
        dict(observations=changed_labels),
        dict(observations=changed_target),
        dict(partition=ObservationPartition(region.name, training.pairs[:-1], role="training")),
        dict(partition=ObservationPartition(region.name, training.pairs, role="calibration")),
        dict(config=replace(config, epochs=2, learning_rate=0.003)),
        dict(config=replace(config, epochs=2, solver=SolverConfig(rtol=1e-8))),
        dict(config=replace(config, epochs=0)),
        dict(model=checkpoint_model()),
    ]
    callbacks = []
    for change in changes:
        with pytest.raises(ValueError):
            fit(**(base | change), on_epoch=callbacks.append)
    assert callbacks == []
    extended = fit(**base)
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
        config=FitConfig(epochs=0),
        on_epoch=callbacks.append,
    )
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
    corrupt("kind.ilg", lambda m: m.update(kind="predictor"), "training_checkpoint")
