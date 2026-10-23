"""Scientific behavior of checkpoint."""

import json
from dataclasses import replace
from typing import TypedDict, cast

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ilg_toolkit import (
    MLPEConfig,
    ObservationPartition,
    PairwiseObservations,
    TrainingConfig,
    fit,
    load_checkpoint,
    save_checkpoint,
)
from ilg_toolkit.training import (
    ObservationCollection,
    PartitionCollection,
    RegionCollection,
)


@pytest.mark.parametrize(
    ("family", "objective", "selection", "num_species"),
    [
        (family, objective, "final", 1)
        for family in ("embedding", "conductance", "unet", "resnet")
        for objective in ("direct_log1p", "mlpe")
    ]
    + [("unet", objective, "validation", 1) for objective in ("direct_log1p", "mlpe")]
    + [("embedding", objective, "final", 2) for objective in ("direct_log1p", "mlpe")]
    + [("embedding", "mlpe", "validation", 2)],
)
def test_resume_equals_uninterrupted(
    tmp_path,
    training_problem,
    checkpoint_problem,
    checkpoint_model,
    family,
    objective,
    selection,
    num_species,
):
    region, observations, encoder = training_problem(family, num_species=num_species)
    species = ("species-1", "species-0") if num_species > 1 else None
    other = replace(region, name="valley")
    other_observations = (
        replace(observations, values=observations.values * 1.2)
        if isinstance(observations, PairwiseObservations)
        else tuple(replace(observed, values=observed.values * 1.2) for observed in observations)
    )
    regions, targets = (region, other), (observations, other_observations)
    if family in ("unet", "resnet"):
        regions, targets = (region,), (observations,)
    config = TrainingConfig(
        objective=objective,
        steps=2,
        eval_every=2,
        seed=21,
        learning_rate=1e-4,
        region_batch_size=1 if family in ("embedding", "conductance") else None,
        species_batch_size=1 if num_species > 1 else None,
        species_microbatch_size=1 if num_species > 1 else None,
    )
    options = _ValidationOptions()
    if selection == "validation" and num_species > 1:
        group_targets = cast(tuple[tuple[PairwiseObservations, ...], ...], targets)
        options = _ValidationOptions(
            partition={
                r.name: tuple(
                    ObservationPartition(r.name, o.observed_pairs[:-1], role="training")
                    for o in groups
                )
                for r, groups in zip(regions, group_targets, strict=True)
            },
            validation=(regions, tuple(groups[0] for groups in group_targets)),
            validation_partition={
                r.name: ObservationPartition(
                    r.name, groups[0].observed_pairs[-1:], role="validation"
                )
                for r, groups in zip(regions, group_targets, strict=True)
            },
        )
    elif selection == "validation":
        problems = (checkpoint_problem(), checkpoint_problem("valley", seed=71))
        regions = tuple(problem[0] for problem in problems)
        targets = tuple(problem[1] for problem in problems)
        encoder = checkpoint_model()
        config = replace(
            config, learning_rate=0.002, mlpe=MLPEConfig(initial_variances=(0.05, 0.1))
        )
        if objective == "mlpe":
            options = _ValidationOptions(
                partition={
                    r.name: ObservationPartition(r.name, o.observed_pairs[:-2], role="training")
                    for r, o in zip(regions, targets, strict=True)
                },
                validation=(regions, targets),
                validation_partition={
                    r.name: ObservationPartition(r.name, o.observed_pairs[-2:], role="validation")
                    for r, o in zip(regions, targets, strict=True)
                },
            )
        else:
            validation_regions = tuple(replace(r, name=r.name + "-validation") for r in regions)
            predictions = tuple(encoder.predict(r) for r in regions)
            validation_targets = tuple(
                PairwiseObservations.from_pairs(
                    prediction.pairs, prediction.values, target=o.target
                )
                for prediction, o in zip(predictions, targets, strict=True)
            )
            options = _ValidationOptions(validation=(validation_regions, validation_targets))
    states = []
    uninterrupted = fit(
        regions,
        targets,
        distance_model=encoder,
        config=config,
        species=species,
        on_step=states.append,
        **options,
    )
    partial = states[1]
    if selection == "validation":
        assert partial.selected_step == 0
        assert not eqx.tree_equal(partial.latest_model, partial.selected_model)
    path = tmp_path / "checkpoint.eqx"
    save_checkpoint(path, partial)
    restored = load_checkpoint(path, distance_model=encoder)
    assert eqx.tree_equal(restored, partial)
    # Reversing region order also checks restoration of the random trajectory.
    resumed = fit(regions[::-1], targets[::-1], state=restored, **options)
    assert eqx.tree_equal(resumed, uninterrupted)
    if selection == "validation" and objective == "mlpe" and num_species == 1:
        save_checkpoint(path, resumed)
        completed = load_checkpoint(path, distance_model=encoder)
        callbacks = []
        unchanged = fit(regions, targets, state=completed, on_step=callbacks.append, **options)
        assert callbacks == []
        assert eqx.tree_equal(unchanged, resumed)
        assert resumed.selected_step == 0
        validation_loss = resumed.history[-1].validation_loss
        assert validation_loss is not None and validation_loss > resumed.selected_loss
    for prepared in regions:
        for name in species or (None,):
            np.testing.assert_array_equal(
                resumed.model.predict(prepared, species=name).values,
                uninterrupted.model.predict(prepared, species=name).values,
            )
    if num_species > 1 and selection == "validation":
        assert resumed.history[-1].validation_by_species is not None
        assert set(resumed.history[-1].validation_by_species) == {"species-0"}


class _ValidationOptions(TypedDict, total=False):
    validation: tuple[RegionCollection, ObservationCollection]
    partition: PartitionCollection
    validation_partition: PartitionCollection


def test_checkpoint_rejects_optimizer_precision_that_differs_from_model(training_problem, tmp_path):
    with jax.enable_x64():
        region, observations, model = training_problem("embedding")
        model = jax.tree.map(
            lambda value: value.astype(np.float32) if eqx.is_inexact_array(value) else value, model
        )
        result = fit(
            region,
            observations,
            distance_model=model,
            config=TrainingConfig(objective="direct_log1p", steps=0),
        )

        changed_optimizer = jax.tree.map(
            lambda value: value.astype(np.float64) if value.dtype == np.float32 else value,
            result.optimizer_state,
        )
        path = tmp_path / "mixed-precision.eqx"
        save_checkpoint(path, replace(result, optimizer_state=changed_optimizer))
        with pytest.raises(ValueError):
            load_checkpoint(path, distance_model=model)


def test_interrupted_replacement_retains_previous_complete_generation(
    training_problem, tmp_path, monkeypatch
):
    import os
    from pathlib import Path

    region, observations, model = training_problem("embedding")
    completed = []
    config = TrainingConfig(objective="direct_log1p", steps=1, seed=21, learning_rate=0.002)
    result = fit(
        region,
        observations,
        distance_model=model,
        config=config,
        on_step=completed.append,
    )

    assert eqx.tree_equal(completed[-1], result)
    path = tmp_path / "atomic.eqx"
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
    restored = load_checkpoint(path, distance_model=model)
    assert eqx.tree_equal(restored, completed[0])


def test_disk_continuation_uses_caller_supplied_inputs_and_partitions(training_problem, tmp_path):
    region, observations, distance_model = training_problem("embedding")
    config = TrainingConfig(objective="direct_log1p", steps=0)
    original = fit(region, observations, distance_model=distance_model, config=config)
    path = tmp_path / "continuation.eqx"
    save_checkpoint(path, original)
    restored = load_checkpoint(path, distance_model=distance_model)
    changed_region = replace(
        region,
        features=cx.field(
            np.concatenate([region.feature_array, region.feature_array + 0.1], axis=0),
            "row",
            "column",
            "feature",
        ),
        sampling_unit_ids=region.sampling_unit_ids[::-1],
    )
    changed_observations = replace(observations, values=observations.values * 1.2)
    partition = ObservationPartition(region.name, observations.observed_pairs[:-1], role="training")
    resumed = fit(
        changed_region,
        changed_observations,
        state=restored,
        config=replace(config, steps=1),
        partition=partition,
    )
    assert resumed.step == 1
    assert not eqx.tree_equal(resumed.distance_model, restored.distance_model)
    assert np.isfinite(resumed.model.predict(changed_region).values).all()


def test_checkpoint_rejects_truncated_leaves_and_wrong_encoder_structure(
    training_problem, WeightedEmbedding, tmp_path
):
    from ilg_toolkit import ArtifactError

    region, observations, model = training_problem("embedding")
    result = fit(
        region,
        observations,
        distance_model=model,
        config=TrainingConfig(objective="direct_log1p", steps=0),
    )

    path = tmp_path / "incompatible.eqx"
    save_checkpoint(path, result)
    with pytest.raises(ArtifactError):
        load_checkpoint(
            path,
            distance_model=WeightedEmbedding(jnp.zeros(3)),
        )
    path.write_bytes(path.read_bytes()[:-20])
    with pytest.raises(ArtifactError):
        load_checkpoint(path, distance_model=model)


@pytest.mark.parametrize("boundary", ["save", "load", "resume"])
def test_checkpoint_enforces_progress_bounds(training_problem, tmp_path, boundary):
    region, observations, distance_model = training_problem("embedding")
    state = fit(
        region,
        observations,
        distance_model=distance_model,
        config=TrainingConfig(objective="direct_log1p", steps=0),
    )
    path = tmp_path / "bounds.eqx"
    save_checkpoint(path, state)
    with path.open("rb") as stream:
        header = json.loads(stream.readline())
    changes = (
        {"step": -1},
        {"step": 1},
        {"selected_step": -1},
        {"selected_step": 1},
        {"history": ()},
        {"history": state.history * 2},
    )
    for change in changes:
        invalid = replace(state, **change)
        if boundary == "load":
            header["metadata"]["history_length"] = len(invalid.history)
            with path.open("wb") as stream:
                stream.write(json.dumps(header).encode() + b"\n")
                eqx.tree_serialise_leaves(stream, invalid)
        with pytest.raises(ValueError):
            if boundary == "save":
                save_checkpoint(path, invalid)
            elif boundary == "load":
                load_checkpoint(path, distance_model=distance_model)
            else:
                fit(region, observations, state=invalid)
