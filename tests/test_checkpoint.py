"""Disk continuation checked through the public fit/checkpoint workflow."""

from dataclasses import replace
from typing import TypedDict, cast

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
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
    region = RegionBatch(
        name,
        cx.field(features, "row", "column", "feature"),
        ids,
        positions,
        feature_names=("elevation", "canopy"),
    )
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


def test_native_checkpoint_restores_latest_model_and_history(tmp_path):
    region, observations = checkpoint_problem()
    result = fit(
        region,
        observations,
        model=checkpoint_model(),
        config=TrainingConfig(objective="direct_log1p", epochs=0),
    )
    assert isinstance(result, TrainingState)
    path = tmp_path / "state.eqx"
    save_checkpoint(path, result)
    restored = load_checkpoint(path, encoder=checkpoint_model())
    assert restored.history == result.history
    assert restored.epoch == 0
    np.testing.assert_array_equal(
        restored.latest_model.predict(region).values, result.model.predict(region).values
    )


def assert_same_state(actual, expected):
    # Includes history, both calibrated models, posterior arrays, optimizer and RNG.
    assert eqx.tree_equal(actual, expected)


def test_direct_disk_resume_matches_uninterrupted_dropout_training(tmp_path):
    region, observations = checkpoint_problem()
    config = TrainingConfig(objective="direct_log1p", epochs=3, seed=21, learning_rate=0.002)
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
    restored = load_checkpoint(path, encoder=checkpoint_model())
    assert restored.epoch == 1
    with pytest.raises(ValueError, match="epoch budget"):
        fit(region, observations, state=restored, config=replace(config, epochs=2))
    resumed = fit(region, observations, state=restored, config=config)

    assert_same_state(resumed, uninterrupted)
    np.testing.assert_array_equal(
        resumed.model.predict(region).values, uninterrupted.model.predict(region).values
    )
    assert all(record.validation_loss is None for record in resumed.history)
    assert resumed.selection == "final"


@pytest.mark.parametrize("objective", ["direct_log1p", "mlpe"])
def test_chunked_resnet_disk_resume_and_calibration_retain_execution_setting(tmp_path, objective):
    from test_conductance import dense_resistance

    from ilg_toolkit import load_model, save_model
    from ilg_toolkit.models import ResNet9Conductance

    features = np.random.default_rng(4).normal(size=(8, 8, 2)).astype(np.float32)
    region = RegionBatch(
        "resnet-resume",
        cx.field(features, "row", "column", "feature"),
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

        path = tmp_path / "resnet.ilg"
        save_checkpoint(path, partial)
        restored = load_checkpoint(path, encoder=encoder)
        for saved in (restored.encoder, restored.latest_model.encoder, restored.best_model.encoder):
            assert isinstance(saved, ResNet9Conductance)
            assert saved.patch_batch_size == 3
        resumed = fit(region, observations, state=restored, config=config)

        assert_same_state(resumed, uninterrupted)
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
            loaded = load_model(path, encoder=encoder)
            assert isinstance(loaded.encoder, ResNet9Conductance)
            assert loaded.encoder.patch_batch_size == 3
            np.testing.assert_array_equal(
                loaded.predict(region).values, calibrated.predict(region).values
            )


def test_shared_mlpe_disk_resume_preserves_nuisance_heads_and_selected_model(tmp_path):
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

        partial = fit(
            regions,
            observations,
            model=checkpoint_model(),
            config=replace(config, epochs=1),
            **arguments,
        )

        path = tmp_path / "mlpe.ilg"
        save_checkpoint(path, partial)
        restored = load_checkpoint(path, encoder=checkpoint_model())
        assert_same_state(restored, partial)
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

        assert_same_state(resumed, uninterrupted)
        assert resumed.model.calibrations == uninterrupted.model.calibrations
        assert resumed.latest_model.calibrations == uninterrupted.latest_model.calibrations
        for region in regions:
            np.testing.assert_array_equal(
                resumed.model.predict(region).values,
                uninterrupted.model.predict(region).values,
            )


def test_zero_update_and_worsening_mlpe_resume_retain_previous_best_heads(tmp_path):
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

        assert partial.selected_epoch == 0
        path = tmp_path / "selected.ilg"
        save_checkpoint(path, partial)
        restored = load_checkpoint(path, encoder=checkpoint_model())
        callbacks = []
        unchanged = fit(
            regions, observations, state=restored, on_epoch=callbacks.append, **arguments
        )

        assert callbacks == []
        assert_same_state(unchanged, partial)
        continued = fit(
            regions, observations, state=restored, config=replace(config, epochs=2), **arguments
        )

        assert continued.history[-1].validation_loss is not None
        assert continued.history[-1].validation_loss > partial.best_loss
        assert continued.selected_epoch == 0
        assert continued.model.calibrations == partial.model.calibrations
        for r in regions:
            np.testing.assert_array_equal(
                continued.model.predict(r).values, partial.model.predict(r).values
            )
        assert continued.latest_model.calibrations != partial.model.calibrations
        assert continued.epoch == 2


def test_checkpoint_rejects_optimizer_precision_that_differs_from_model(tmp_path):
    with jax.enable_x64():
        region, observations = checkpoint_problem()
        result = fit(
            region,
            observations,
            model=checkpoint_model(),
            config=TrainingConfig(objective="direct_log1p", epochs=0),
        )

        changed_optimizer = jax.tree.map(
            lambda value: value.astype(np.float64) if value.dtype == np.float32 else value,
            result.optimizer_state,
        )
        path = tmp_path / "mixed-precision.ilg"
        save_checkpoint(path, replace(result, optimizer_state=changed_optimizer))
        with pytest.raises(ValueError, match="dtype"):
            load_checkpoint(path, encoder=checkpoint_model())


def test_interrupted_replacement_retains_previous_complete_generation(tmp_path, monkeypatch):
    import os
    from pathlib import Path

    region, observations = checkpoint_problem()
    completed = []
    config = TrainingConfig(objective="direct_log1p", epochs=1, seed=21, learning_rate=0.002)
    result = fit(
        region, observations, model=checkpoint_model(), config=config, on_epoch=completed.append
    )

    assert_same_state(completed[-1], result)
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
            save_checkpoint(path, result)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
    restored = load_checkpoint(path, encoder=checkpoint_model())
    assert_same_state(restored, completed[0])
    continued = fit(region, observations, state=restored, config=config)

    assert_same_state(continued, result)


def test_disk_continuation_rejects_changed_inputs_config_model_and_partitions(tmp_path):
    from ilg_toolkit import ObservationPartition

    region, observations = checkpoint_problem()
    training = ObservationPartition(region.name, observations.observed_pairs[:-2], role="training")
    config = TrainingConfig(objective="direct_log1p", epochs=1, seed=21, learning_rate=0.002)
    result = fit(region, observations, model=checkpoint_model(), config=config, partition=training)

    path = tmp_path / "compatible.ilg"
    save_checkpoint(path, result)
    state = load_checkpoint(path, encoder=checkpoint_model())
    base = _FitOptions(
        region=region,
        observations=observations,
        state=state,
        config=replace(config, epochs=2),
        partition=training,
    )
    changed_target = replace(observations, target=TargetSpec("different target", units="index"))
    changes: list[_FitOptions] = [
        _FitOptions(
            region=replace(
                region,
                features=cx.field(
                    np.concatenate([region.feature_array, region.feature_array], axis=0),
                    "row",
                    "column",
                    "feature",
                ),
            )
        ),
        _FitOptions(
            region=replace(
                region,
                features=cx.field(region.feature_array, "row", "column", "feature"),
                feature_names=("canopy", "elevation"),
            )
        ),
        _FitOptions(region=replace(region, sampling_unit_ids=region.sampling_unit_ids[::-1])),
        _FitOptions(region=replace(region, sampling_unit_kinds=("individual",) * 6)),
        _FitOptions(
            region=replace(
                region,
                features=cx.field(
                    jnp.asarray(region.feature_array, dtype=jnp.float64),
                    "row",
                    "column",
                    "feature",
                ),
            )
        ),
        _FitOptions(
            observations=replace(observations, values=observations.values.astype(jnp.float32))
        ),
        _FitOptions(observations=changed_target),
        _FitOptions(
            partition=ObservationPartition(region.name, training.pairs[:-1], role="training")
        ),
        _FitOptions(
            partition=ObservationPartition(region.name, training.pairs, role="calibration")
        ),
        _FitOptions(config=replace(config, epochs=2, learning_rate=0.003)),
        _FitOptions(config=replace(config, epochs=0)),
        _FitOptions(model=checkpoint_model()),
    ]
    callbacks = []
    for change in changes:
        with pytest.raises(ValueError):
            fit(**(base | change), on_epoch=callbacks.append)
    assert callbacks == []
    extended = fit(**base)

    assert extended.epoch == 2


def test_same_shape_measurement_edits_are_the_callers_responsibility(tmp_path):
    region, observations = checkpoint_problem()
    result = fit(
        region,
        observations,
        model=checkpoint_model(),
        config=TrainingConfig(objective="direct_log1p", epochs=0),
    )

    path = tmp_path / "unchanged-structure.eqx"
    save_checkpoint(path, result)
    state = load_checkpoint(path, encoder=checkpoint_model())
    changed_region = replace(
        region,
        features=cx.field(region.feature_array + 0.1, "row", "column", "feature"),
        grid_positions=cast(jax.Array, region.grid_positions)[::-1],
    )
    changed_observations = PairwiseObservations.from_pairs(
        observations.observed_pairs,
        observations.observed_values + 0.1,
        target=observations.target,
        sampling_unit_ids=region.sampling_unit_ids,
    )
    callbacks = []
    resumed = fit(changed_region, changed_observations, state=state, on_epoch=callbacks.append)

    assert callbacks == []
    assert_same_state(resumed, state)


def test_checkpoint_rejects_truncated_leaves_and_wrong_encoder_structure(tmp_path):
    from ilg_toolkit import ArtifactError

    region, observations = checkpoint_problem()
    result = fit(
        region,
        observations,
        model=checkpoint_model(),
        config=TrainingConfig(objective="direct_log1p", epochs=0),
    )

    path = tmp_path / "incompatible.eqx"
    save_checkpoint(path, result)
    with pytest.raises(ArtifactError, match="encoder template"):
        load_checkpoint(
            path,
            encoder=UNetEmbeddingDistance(
                2,
                patch_size=1,
                base_channels=4,
                embedding_dim=2,
                dropout=0.2,
                key=jax.random.key(8),
            ),
        )
    path.write_bytes(path.read_bytes()[:-20])
    with pytest.raises(ArtifactError, match="encoder template"):
        load_checkpoint(path, encoder=checkpoint_model())


def test_mlpe_checkpoint_requires_original_precision_without_changing_global_settings(tmp_path):
    from ilg_toolkit import ArtifactError

    with jax.enable_x64():
        region, observations = checkpoint_problem()
        result = fit(
            region,
            observations,
            model=checkpoint_model(),
            config=TrainingConfig(objective="mlpe", epochs=0),
        )

        path = tmp_path / "precision.eqx"
        save_checkpoint(path, result)
    with jax.enable_x64(False):
        with pytest.raises(ArtifactError, match="JAX_ENABLE_X64"):
            load_checkpoint(path, encoder=checkpoint_model())
        assert not jax.config.read("jax_enable_x64")


def test_individual_ensemble_member_disk_resume_uses_a_custom_encoder(tmp_path):
    from test_mlpe_training import WeightedEmbedding

    from ilg_toolkit import fit_ensemble_member, generate_population_folds

    region, observations = checkpoint_problem()
    fold = generate_population_folds(region, observations, n_folds=1, holdout_size=2, seed=47)[0]
    config = TrainingConfig(objective="direct_log1p", epochs=2, learning_rate=0.002)

    def factory(key):
        return WeightedEmbedding(jax.random.normal(key, (2,)) * 0.1)

    uninterrupted = fit_ensemble_member(
        region,
        observations,
        config=config,
        fold=fold,
        initialization_seed=13,
        model_factory=factory,
    )
    path = tmp_path / "fold-0-seed-13.eqx"
    partial = fit_ensemble_member(
        region,
        observations,
        config=replace(config, epochs=1),
        on_epoch=lambda identity, state: save_checkpoint(path, state),
        fold=fold,
        initialization_seed=13,
        model_factory=factory,
    )
    assert partial.status == uninterrupted.status == "completed"
    state = load_checkpoint(path, encoder=factory(jax.random.PRNGKey(13)))
    resumed = fit_ensemble_member(
        region,
        observations,
        config=config,
        state=state,
        fold=fold,
        initialization_seed=13,
        model_factory=factory,
    )
    assert resumed.status == "completed"
    assert resumed.identity == uninterrupted.identity
    assert resumed.training_state is not None and uninterrupted.training_state is not None
    assert_same_state(resumed.training_state, uninterrupted.training_state)
    assert resumed.model is not None and uninterrupted.model is not None
    np.testing.assert_array_equal(
        resumed.model.predict(region).values, uninterrupted.model.predict(region).values
    )
