"""Portable ensembles and real interrupted member-by-member continuation."""

import json
import shutil
import zipfile
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np
import pytest
from test_checkpoint import assert_same_state, checkpoint_problem

from ilg_toolkit import ArtifactError, TrainingConfig, fit_ensemble, generate_population_folds
from ilg_toolkit.models import UNetEmbeddingDistance


def factory(key):
    return UNetEmbeddingDistance(
        2, patch_size=1, base_channels=2, embedding_dim=2, dropout=0.2, key=key
    )


def forbid_initialization(key):
    raise AssertionError("saved members must never be reinitialized")


def manifest(path):
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read("manifest.json"))


def alter_manifest(path, change):
    with zipfile.ZipFile(path) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    record = json.loads(entries["manifest.json"])
    change(record["payload"])
    entries["manifest.json"] = json.dumps(record).encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in entries.items():
            archive.writestr(name, value)


@pytest.fixture(scope="module")
def saved_run(tmp_path_factory):
    from ilg_toolkit import fit_ensemble_run

    directory = tmp_path_factory.mktemp("ensemble-run")
    region, observations = checkpoint_problem()
    folds = generate_population_folds(region, observations, n_folds=1, holdout_size=2, seed=47)
    with jax.enable_x64():
        ensemble = fit_ensemble_run(
            directory,
            region,
            observations,
            folds=folds,
            initialization_seeds=(13,),
            config=TrainingConfig(objective="mlpe", epochs=0),
            model_factory=factory,
        )
    record = manifest(directory / "run.ilg")
    assert record["schema"] == record["payload"]["run_version"] == 1
    member = record["payload"]["members"][0]
    assert "model" not in member
    assert member["predictor"]["kind"] == "predictor"
    return directory, region, observations, folds, ensemble


def test_portable_ensemble_preserves_calibrated_members_and_descriptive_spread(tmp_path):
    from ilg_toolkit import load_ensemble, save_ensemble

    region, observations = checkpoint_problem()
    with jax.enable_x64():
        ensemble = fit_ensemble(
            region,
            observations,
            n_folds=1,
            holdout_size=2,
            fold_seed=47,
            initialization_seeds=(13, 29),
            config=TrainingConfig(objective="mlpe", epochs=0),
            model_factory=factory,
        )
        path = tmp_path / "portable.ilg"
        save_ensemble(path, ensemble)
        loaded = load_ensemble(path)
        before, after = ensemble.predict(region), loaded.predict(region)
        np.testing.assert_array_equal(after.values, before.values)
        np.testing.assert_array_equal(after.member_values, before.member_values)
        np.testing.assert_array_equal(after.member_spread, before.member_spread)
        assert after.member_ids == before.member_ids
        for original, restored in zip(ensemble.members, loaded.members, strict=True):
            assert restored.identity == original.identity
            assert restored.fold == original.fold
            assert restored.model.calibrations == original.model.calibrations
            assert restored.model.training_pairs == original.model.training_pairs
            assert restored.fit_result is None


def test_ensemble_retains_schema_one_predictor_fields(saved_run, tmp_path):
    from ilg_toolkit import load_ensemble, save_ensemble

    _, region, _, _, ensemble = saved_run
    path = tmp_path / "schema-one-ensemble.ilg"
    with jax.enable_x64():
        save_ensemble(path, ensemble)
        assert "predictor" in manifest(path)["payload"]["members"][0]
        # Disk field names are independent of the canonical Python model API.
        alter_manifest(
            path,
            lambda payload: [
                member.update(predictor=member.pop("model", member.get("predictor")))
                for member in payload["members"]
            ],
        )
        loaded = load_ensemble(path)
        np.testing.assert_array_equal(
            loaded.predict(region).values, ensemble.predict(region).values
        )
    record = manifest(path)
    assert record["schema"] == record["payload"]["ensemble_version"] == 1
    assert "predictor" in record["payload"]["members"][0]


def test_real_interruption_skips_complete_and_resumes_unfinished_members(tmp_path):
    from ilg_toolkit import fit_ensemble_run

    region, observations = checkpoint_problem()
    folds = generate_population_folds(region, observations, n_folds=1, holdout_size=2, seed=47)
    config = TrainingConfig(objective="mlpe", epochs=2, learning_rate=0.002)
    arguments = dict(folds=folds, initialization_seeds=(13, 29), config=config)
    initialized, completed = [], []

    def tracked(key):
        initialized.append(np.asarray(key).copy())
        return factory(key)

    def interrupt(identity, state):
        if identity.initialization_seed == 29 and state.epoch == 1:
            raise InterruptedError("interrupt second member after saved epoch")

    with jax.enable_x64():
        baseline = fit_ensemble(region, observations, model_factory=factory, **arguments)
        with pytest.raises(InterruptedError, match="second member"):
            fit_ensemble_run(
                tmp_path / "run",
                region,
                observations,
                model_factory=tracked,
                on_epoch=interrupt,
                on_member=completed.append,
                **arguments,
            )
        assert len(initialized) == 2
        assert len(completed) == 1
        epochs = []

        def no_initialization(key):
            raise AssertionError("saved members must never be reinitialized")

        resumed = fit_ensemble_run(
            tmp_path / "run",
            region,
            observations,
            resume=True,
            model_factory=no_initialization,
            on_epoch=lambda identity, state: epochs.append(
                (identity.initialization_seed, state.epoch)
            ),
        )
        assert epochs == [(29, 2)]
        np.testing.assert_array_equal(
            resumed.predict(region).values, baseline.predict(region).values
        )
        for actual, expected in zip(resumed.members, baseline.members, strict=True):
            assert_same_state(actual.fit_result.state, expected.fit_result.state)
        epochs.clear()
        skipped = fit_ensemble_run(
            tmp_path / "run",
            region,
            observations,
            resume=True,
            model_factory=no_initialization,
            on_epoch=lambda identity, state: epochs.append(state.epoch),
        )
        assert epochs == []
        np.testing.assert_array_equal(
            skipped.predict(region).values, baseline.predict(region).values
        )


def test_larger_budget_continues_previously_completed_members(tmp_path):
    from ilg_toolkit import fit_ensemble_run

    region, observations = checkpoint_problem()
    folds = generate_population_folds(region, observations, n_folds=1, holdout_size=2, seed=47)
    config = TrainingConfig(objective="mlpe", epochs=1, learning_rate=0.002)
    arguments = dict(folds=folds, initialization_seeds=(13, 29))
    with jax.enable_x64():
        fit_ensemble_run(
            tmp_path, region, observations, config=config, model_factory=factory, **arguments
        )
        extended_config = replace(config, epochs=3)
        baseline = fit_ensemble(
            region, observations, config=extended_config, model_factory=factory, **arguments
        )
        epochs = []
        extended = fit_ensemble_run(
            tmp_path,
            region,
            observations,
            resume=True,
            config=extended_config,
            model_factory=forbid_initialization,
            on_epoch=lambda identity, state: epochs.append(
                (identity.initialization_seed, state.epoch)
            ),
        )
        assert epochs == [(13, 2), (13, 3), (29, 2), (29, 3)]
        for actual, expected in zip(extended.members, baseline.members, strict=True):
            assert_same_state(actual.fit_result.state, expected.fit_result.state)
        np.testing.assert_array_equal(
            extended.predict(region).values, baseline.predict(region).values
        )


def test_resume_checks_accessed_data_config_folds_and_composition_before_work(saved_run):
    from ilg_toolkit import fit_ensemble_run

    directory, region, observations, folds, ensemble = saved_run
    pair = folds[0].training[region.name].pairs[0]
    indices = [observations.sampling_unit_ids.index(label) for label in pair]
    values = observations.values.copy()
    values[indices[0], indices[1]] += 0.1
    values[indices[1], indices[0]] += 0.1
    changed_observations = replace(observations, values=values)
    changed_region = replace(region, features=region.feature_array + 0.1)
    alternatives = [
        (region, changed_observations, {}),
        (changed_region, observations, {}),
        (region, observations, {"initialization_seeds": (13, 29)}),
        (region, observations, {"folds": (replace(folds[0], fold_id="other"),)}),
        (
            region,
            observations,
            {"config": TrainingConfig(objective="mlpe", epochs=0, learning_rate=0.1)},
        ),
    ]
    before = (directory / "run.ilg").read_bytes()
    with jax.enable_x64():
        for prepared, observed, options in alternatives:
            with pytest.raises(ArtifactError, match="configuration|composition"):
                fit_ensemble_run(
                    directory,
                    prepared,
                    observed,
                    resume=True,
                    model_factory=forbid_initialization,
                    **options,
                )
            assert (directory / "run.ilg").read_bytes() == before
        # Unaccessed query measurements do not change a training continuation.
        pair = folds[0].query[region.name].pairs[0]
        indices = [observations.sampling_unit_ids.index(label) for label in pair]
        values = observations.values.copy()
        values[indices[0], indices[1]] += 100
        values[indices[1], indices[0]] += 100
        skipped = fit_ensemble_run(
            directory,
            region,
            replace(observations, values=values),
            resume=True,
            model_factory=forbid_initialization,
        )
        np.testing.assert_array_equal(
            skipped.predict(region).values, ensemble.predict(region).values
        )


@pytest.mark.parametrize("damage", ["missing", "corrupt", "traversal", "symlink"])
def test_resume_reports_damaged_member_without_shrinking_composition(saved_run, tmp_path, damage):
    from ilg_toolkit import fit_ensemble_run

    directory, region, observations, _, _ = saved_run
    run = tmp_path / "copy"
    shutil.copytree(directory, run)
    member = manifest(run / "run.ilg")["payload"]["members"][0]
    checkpoint = run / "members" / member["checkpoint"]["file"]
    if damage == "missing":
        checkpoint.unlink()
    elif damage == "corrupt":
        checkpoint.write_bytes(b"partial checkpoint")
    elif damage == "traversal":
        alter_manifest(
            run / "run.ilg",
            lambda payload: payload["members"][0]["checkpoint"].update(file="../../outside.ilg"),
        )
    else:
        external = tmp_path / "outside.ilg"
        checkpoint.rename(external)
        checkpoint.symlink_to(external)
    before = (run / "run.ilg").read_bytes()
    with jax.enable_x64(), pytest.raises(ArtifactError, match=member["identity"]["member_id"]):
        fit_ensemble_run(
            run, region, observations, resume=True, model_factory=forbid_initialization
        )
    assert (run / "run.ilg").read_bytes() == before


def test_failed_members_remain_explicit_in_run_and_portable_artifact(tmp_path):
    from ilg_toolkit import fit_ensemble_run, load_ensemble, save_ensemble

    region, observations = checkpoint_problem()

    def fail(key):
        raise RuntimeError("deliberate initialization failure")

    failed = fit_ensemble_run(
        tmp_path / "failed",
        region,
        observations,
        n_folds=1,
        holdout_size=2,
        initialization_seeds=(13, 29),
        config=TrainingConfig(epochs=0),
        model_factory=fail,
    )
    resumed = fit_ensemble_run(
        tmp_path / "failed",
        region,
        observations,
        resume=True,
        model_factory=forbid_initialization,
    )
    path = tmp_path / "failed.ilg"
    save_ensemble(path, failed)
    loaded = load_ensemble(path)
    assert len(failed.expected_member_ids) == 2
    assert loaded.expected_member_ids == resumed.expected_member_ids == failed.expected_member_ids
    assert loaded.failures == resumed.failures == failed.failures
    for ensemble in (loaded, resumed):
        with pytest.raises(RuntimeError, match="unavailable members"):
            ensemble.predict(region)


def test_atomic_manifest_failure_retains_previous_member_generation(
    saved_run, tmp_path, monkeypatch
):
    import os

    from ilg_toolkit import fit_ensemble_run

    directory, region, observations, folds, ensemble = saved_run
    run = tmp_path / "copy"
    shutil.copytree(directory, run)
    original = os.replace
    publications = []

    def interrupt(source, destination):
        if Path(destination) == run / "run.ilg":
            publications.append(destination)
            if len(publications) == 2:
                raise OSError("injected manifest publication failure")
        original(source, destination)

    with jax.enable_x64():
        config = replace(ensemble.members[0].fit_result.state.config, epochs=1, seed=0)
        baseline = fit_ensemble(
            region,
            observations,
            folds=folds,
            initialization_seeds=(13,),
            config=config,
            model_factory=factory,
        )
        with monkeypatch.context() as patch:
            patch.setattr(os, "replace", interrupt)
            with pytest.raises(OSError, match="publication failure"):
                fit_ensemble_run(
                    run,
                    region,
                    observations,
                    resume=True,
                    config=config,
                    model_factory=forbid_initialization,
                )
        retained = manifest(run / "run.ilg")["payload"]["members"][0]
        assert retained["epoch"] == 0
        resumed = fit_ensemble_run(
            run, region, observations, resume=True, model_factory=forbid_initialization
        )
        assert_same_state(resumed.members[0].fit_result.state, baseline.members[0].fit_result.state)


def test_pending_member_cannot_change_saved_encoder_architecture(tmp_path):
    from ilg_toolkit import fit_ensemble_run

    region, observations = checkpoint_problem()

    def stop_after_first(member):
        raise InterruptedError("first member durably completed")

    def different_dropout(key):
        return UNetEmbeddingDistance(
            2, patch_size=1, base_channels=2, embedding_dim=2, dropout=0.3, key=key
        )

    with jax.enable_x64():
        with pytest.raises(InterruptedError, match="first member"):
            fit_ensemble_run(
                tmp_path,
                region,
                observations,
                n_folds=1,
                holdout_size=2,
                initialization_seeds=(13, 29),
                config=TrainingConfig(objective="mlpe", epochs=0),
                model_factory=factory,
                on_member=stop_after_first,
            )
        before = manifest(tmp_path / "run.ilg")["payload"]["members"]
        assert [member["status"] for member in before] == ["completed", "pending"]
        with pytest.raises(ArtifactError, match="incompatible encoder architecture"):
            fit_ensemble_run(
                tmp_path, region, observations, resume=True, model_factory=different_dropout
            )
        after = manifest(tmp_path / "run.ilg")["payload"]["members"]
        assert after == before


def test_portable_artifact_rejects_missing_or_incompatible_member(saved_run, tmp_path):
    from ilg_toolkit import load_ensemble, save_ensemble

    _, _, _, _, ensemble = saved_run
    for damage in ("missing", "incompatible"):
        path = tmp_path / f"{damage}.ilg"
        save_ensemble(path, ensemble)
        if damage == "missing":
            alter_manifest(path, lambda payload: payload.update(members=[]))
        else:
            alter_manifest(path, lambda payload: payload["members"][0].update(predictor=None))
        with jax.enable_x64(), pytest.raises(ArtifactError, match="composition|model"):
            load_ensemble(path)
