"""Scientific behavior of checkpoint."""

from dataclasses import replace
from typing import TypedDict, cast

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ilg_toolkit import (
    MLPECalibrationModule,
    PairwiseObservations,
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


@pytest.mark.parametrize("family", ["embedding", "conductance", "unet", "resnet"])
@pytest.mark.parametrize("objective", ["direct_log1p", "mlpe"])
def test_resume_equals_uninterrupted(tmp_path, training_problem, family, objective):
    region, observations, encoder = training_problem(family)
    other = replace(region, name="valley")
    other_observations = replace(observations, values=observations.values * 1.2)
    regions, targets = (region, other), (observations, other_observations)
    config = TrainingConfig(objective=objective, epochs=2, seed=21, learning_rate=1e-4)
    states = []
    uninterrupted = fit(
        regions,
        targets,
        distance_model=encoder,
        config=config,
        on_epoch=states.append,
    )
    partial = states[1]
    path = tmp_path / "checkpoint.eqx"
    save_checkpoint(path, partial)
    restored = load_checkpoint(path, distance_model=encoder)
    assert eqx.tree_equal(restored, partial)
    # Reversing region order also checks restoration of the random trajectory.
    resumed = fit(regions[::-1], targets[::-1], state=restored)
    assert eqx.tree_equal(resumed, uninterrupted)
    for prepared in regions:
        np.testing.assert_array_equal(
            resumed.model.predict(prepared).values,
            uninterrupted.model.predict(prepared).values,
        )


class _ValidationOptions(TypedDict, total=False):
    validation: tuple[RegionCollection, ObservationCollection]
    partition: PartitionCollection
    validation_partition: PartitionCollection


class _FitOptions(_ValidationOptions, total=False):
    region: RegionCollection
    observations: ObservationCollection
    distance_model: UNetEmbeddingDistance
    config: TrainingConfig
    state: TrainingState


def test_zero_update_and_worsening_mlpe_resume_retain_previous_best_heads(
    assert_same_state, checkpoint_model, checkpoint_problem, tmp_path
):
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
        partial = fit(
            regions, observations, distance_model=checkpoint_model(), config=config, **arguments
        )

        assert partial.selected_epoch == 0
        path = tmp_path / "selected.ilg"
        save_checkpoint(path, partial)
        restored = load_checkpoint(path, distance_model=checkpoint_model())
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
        assert (
            cast(MLPECalibrationModule, continued.model.calibration).regions
            == cast(MLPECalibrationModule, partial.model.calibration).regions
        )
        for r in regions:
            np.testing.assert_array_equal(
                continued.model.predict(r).values, partial.model.predict(r).values
            )
        assert (
            cast(MLPECalibrationModule, continued.latest_model.calibration).regions
            != cast(MLPECalibrationModule, partial.model.calibration).regions
        )
        assert continued.epoch == 2


def test_checkpoint_rejects_optimizer_precision_that_differs_from_model(
    checkpoint_model, checkpoint_problem, tmp_path
):
    with jax.enable_x64():
        region, observations = checkpoint_problem()
        result = fit(
            region,
            observations,
            distance_model=checkpoint_model(),
            config=TrainingConfig(objective="direct_log1p", epochs=0),
        )

        changed_optimizer = jax.tree.map(
            lambda value: value.astype(np.float64) if value.dtype == np.float32 else value,
            result.optimizer_state,
        )
        path = tmp_path / "mixed-precision.ilg"
        save_checkpoint(path, replace(result, optimizer_state=changed_optimizer))
        with pytest.raises(ValueError):
            load_checkpoint(path, distance_model=checkpoint_model())


def test_interrupted_replacement_retains_previous_complete_generation(
    assert_same_state, checkpoint_model, checkpoint_problem, tmp_path, monkeypatch
):
    import os
    from pathlib import Path

    region, observations = checkpoint_problem()
    completed = []
    config = TrainingConfig(objective="direct_log1p", epochs=1, seed=21, learning_rate=0.002)
    result = fit(
        region,
        observations,
        distance_model=checkpoint_model(),
        config=config,
        on_epoch=completed.append,
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
        with pytest.raises(InterruptedError):
            save_checkpoint(path, result)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
    restored = load_checkpoint(path, distance_model=checkpoint_model())
    assert_same_state(restored, completed[0])


def test_disk_continuation_rejects_changed_inputs_config_model_and_partitions(
    checkpoint_model, checkpoint_problem, tmp_path
):
    from ilg_toolkit import ObservationPartition

    region, observations = checkpoint_problem()
    training = ObservationPartition(region.name, observations.observed_pairs[:-2], role="training")
    config = TrainingConfig(objective="direct_log1p", epochs=1, seed=21, learning_rate=0.002)
    result = fit(
        region, observations, distance_model=checkpoint_model(), config=config, partition=training
    )

    path = tmp_path / "compatible.ilg"
    save_checkpoint(path, result)
    state = load_checkpoint(path, distance_model=checkpoint_model())
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
        _FitOptions(distance_model=checkpoint_model()),
    ]
    callbacks = []
    for change in changes:
        with pytest.raises(ValueError):
            fit(**(base | change), on_epoch=callbacks.append)
    assert callbacks == []
    extended = fit(**base)

    assert extended.epoch == 2


def test_same_shape_measurement_edits_are_the_callers_responsibility(
    assert_same_state, checkpoint_model, checkpoint_problem, tmp_path
):
    region, observations = checkpoint_problem()
    result = fit(
        region,
        observations,
        distance_model=checkpoint_model(),
        config=TrainingConfig(objective="direct_log1p", epochs=0),
    )

    path = tmp_path / "unchanged-structure.eqx"
    save_checkpoint(path, result)
    state = load_checkpoint(path, distance_model=checkpoint_model())
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


def test_checkpoint_rejects_truncated_leaves_and_wrong_encoder_structure(
    checkpoint_model, checkpoint_problem, tmp_path
):
    from ilg_toolkit import ArtifactError

    region, observations = checkpoint_problem()
    result = fit(
        region,
        observations,
        distance_model=checkpoint_model(),
        config=TrainingConfig(objective="direct_log1p", epochs=0),
    )

    path = tmp_path / "incompatible.eqx"
    save_checkpoint(path, result)
    with pytest.raises(ArtifactError):
        load_checkpoint(
            path,
            distance_model=UNetEmbeddingDistance(
                2,
                patch_size=1,
                base_channels=4,
                embedding_dim=2,
                dropout=0.2,
                key=jax.random.key(8),
            ),
        )
    path.write_bytes(path.read_bytes()[:-20])
    with pytest.raises(ArtifactError):
        load_checkpoint(path, distance_model=checkpoint_model())
