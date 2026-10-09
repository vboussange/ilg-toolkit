"""Portable inference ensembles and durable sequential member training runs."""

import hashlib
import json
import re
import uuid
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from ._archive import ArrayWriter, ArtifactError, read_archive, require_fields, write_archive
from ._codecs import decode_predictor, encode_predictor
from .checkpoint import load_checkpoint, save_checkpoint
from .config import FitConfig, SolverConfig
from .data import ObservationPartition
from .ensemble import (
    Ensemble,
    EnsembleMember,
    MemberFailure,
    MemberIdentity,
    PopulationFold,
    _fold_inputs,
    _prepare_ensemble,
    fit_ensemble_member,
)
from .persistence import load_predictor, save_predictor
from .training import FitResult, _fit_data_identity


def _identity(record):
    require_fields(
        record,
        {"fold_id", "initialization_seed", "effective_seed", "member_id"},
        "Ensemble member identity",
    )
    if type(record["effective_seed"]) is not int:
        raise ArtifactError("Member effective seed must be an integer")
    return MemberIdentity(**record)


def _fold(record):
    require_fields(
        record,
        {"fold_id", "held_out_units", "training", "query", "validation", "query_regime"},
        "Population fold",
    )
    values = dict(record)
    for role in ("training", "query", "validation"):
        if not isinstance(values[role], dict):
            raise ArtifactError("Fold partitions must be mappings")
        partitions = {}
        for name, partition in values[role].items():
            require_fields(partition, {"region_name", "pairs", "role"}, "Fold partition")
            partitions[name] = ObservationPartition(**partition)
        values[role] = partitions
    if not isinstance(values["held_out_units"], dict):
        raise ArtifactError("Fold endpoint holdouts must be a mapping")
    return PopulationFold(**values)


def _failure(record):
    if record is None:
        return None
    require_fields(record, {"stage", "error_type", "message"}, "Member failure")
    if any(not isinstance(value, str) for value in record.values()):
        raise ArtifactError("Member failure diagnostics must be strings")
    return MemberFailure(**record)


def _member_metadata(member):
    return {
        "identity": asdict(member.identity),
        "fold": asdict(member.fold),
        "status": member.status,
        "failure": None if member.failure is None else asdict(member.failure),
    }


def save_ensemble(path, ensemble: Ensemble) -> None:
    """Atomically save all requested outcomes and completed deployment predictors."""
    try:
        if not isinstance(ensemble, Ensemble):
            raise ArtifactError("save_ensemble requires an Ensemble")
        arrays = ArrayWriter()
        records = []
        for member in ensemble.members:
            record = _member_metadata(member)
            record["predictor"] = (
                encode_predictor(member.predictor, arrays) if member.status == "completed" else None
            )
            records.append(record)
        write_archive(
            path,
            kind="ensemble",
            payload={
                "ensemble_version": 1,
                "expected_member_ids": ensemble.expected_member_ids,
                "members": records,
            },
            arrays=arrays,
        )
    except ArtifactError:
        raise
    except (TypeError, KeyError, ValueError, AttributeError) as error:
        raise ArtifactError(f"Invalid ensemble artifact: {error}") from error


def load_ensemble(path) -> Ensemble:
    """Load a portable ensemble without checkpoints or genetic query observations."""
    try:
        with read_archive(path, expected_kind="ensemble") as archive:
            record = archive.payload
            require_fields(
                record, {"ensemble_version", "expected_member_ids", "members"}, "Ensemble"
            )
            if type(record["ensemble_version"]) is not int or record["ensemble_version"] != 1:
                raise ArtifactError("Unsupported ensemble schema")
            if not isinstance(record["members"], list):
                raise ArtifactError("Ensemble members must be a sequence")
            members = []
            for saved in record["members"]:
                require_fields(
                    saved, {"identity", "fold", "status", "failure", "predictor"}, "Ensemble member"
                )
                predictor = (
                    None
                    if saved["predictor"] is None
                    else decode_predictor(saved["predictor"], archive)
                )
                members.append(
                    EnsembleMember(
                        _identity(saved["identity"]),
                        _fold(saved["fold"]),
                        saved["status"],
                        predictor=predictor,
                        failure=_failure(saved["failure"]),
                    )
                )
            result = Ensemble(tuple(members), tuple(record["expected_member_ids"]))
            archive.finish()
            return result
    except ArtifactError:
        raise
    except (TypeError, KeyError, ValueError, AttributeError) as error:
        raise ArtifactError(f"Invalid ensemble artifact: {error}") from error


# A fingerprint uses the existing model codec but records only leaf shape/dtype,
# not parameter values. Members share architecture, never fitted parameters.
class _StructureWriter:
    def add(self, value):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}


def _architecture(model):
    from ._codecs import encode_model

    return _json_hash(encode_model(model, _StructureWriter()))


def _json_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _member_path(directory, member_id, reference=None):
    prefix = "m-" + hashlib.sha256(member_id.encode()).hexdigest() + "-"
    if reference is None:
        name = prefix + uuid.uuid4().hex + ".ilg"
    else:
        require_fields(reference, {"file", "sha256", "kind"}, "Member file reference")
        name = reference["file"]
        if (
            not isinstance(name, str)
            or re.fullmatch(re.escape(prefix) + r"[0-9a-f]{32}\.ilg", name) is None
            or not isinstance(reference["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", reference["sha256"]) is None
        ):
            raise ArtifactError(f"Member {member_id}: unsafe path or invalid integrity reference")
    parent = directory / "members"
    path = parent / name
    if parent.is_symlink() or path.is_symlink() or path.resolve().parent != parent.resolve():
        raise ArtifactError(f"Member {member_id}: reference escapes the run member directory")
    if parent.resolve().parent != directory.resolve():
        raise ArtifactError("Member directory must belong to the ensemble run")
    return path


def _reference(directory, member_id, value, kind):
    path = _member_path(directory, member_id)
    (save_checkpoint if kind == "training_checkpoint" else save_predictor)(path, value)
    return {"file": path.name, "sha256": _file_hash(path), "kind": kind}


def _load_reference(directory, member_id, reference, kind):
    path = _member_path(directory, member_id, reference)
    if reference["kind"] != kind or not path.is_file():
        raise ArtifactError(f"Member {member_id}: missing or incompatible {kind} artifact")
    try:
        if _file_hash(path) != reference["sha256"]:
            raise ArtifactError("artifact integrity check failed")
        return (load_checkpoint if kind == "training_checkpoint" else load_predictor)(path)
    except (ArtifactError, OSError) as error:
        raise ArtifactError(f"Member {member_id}: {error}") from error


def _publish(directory, record):
    write_archive(directory / "run.ilg", kind="ensemble_run", payload=record, arrays=ArrayWriter())


def _read_run(directory):
    with read_archive(directory / "run.ilg", expected_kind="ensemble_run") as archive:
        record = archive.payload
        require_fields(
            record,
            {
                "run_version",
                "expected_member_ids",
                "config",
                "contracts",
                "architecture",
                "members",
            },
            "Ensemble run",
        )
        if type(record["run_version"]) is not int or record["run_version"] != 1:
            raise ArtifactError("Unsupported ensemble run schema")
        if not isinstance(record["members"], list) or not record["members"]:
            raise ArtifactError("Ensemble run is missing its member composition")
        identities = []
        for member in record["members"]:
            require_fields(
                member,
                {
                    "identity",
                    "fold",
                    "status",
                    "failure",
                    "data_identity",
                    "checkpoint",
                    "predictor",
                    "epoch",
                },
                "Run member",
            )
            identity, fold = _identity(member["identity"]), _fold(member["fold"])
            if identity.fold_id != fold.fold_id:
                raise ArtifactError("Run member identity and fold disagree")
            identities.append(identity.member_id)
            status, checkpoint, predictor = (
                member["status"],
                member["checkpoint"],
                member["predictor"],
            )
            if status not in {"pending", "running", "completed", "failed"}:
                raise ArtifactError("Run member status is invalid")
            if (checkpoint is None) != (member["epoch"] is None):
                raise ArtifactError("Run member epoch and checkpoint must align")
            if checkpoint is not None and (type(member["epoch"]) is not int or member["epoch"] < 0):
                raise ArtifactError("Run member checkpoint progress is invalid")
            if (
                status == "completed"
                and (predictor is None or checkpoint is None)
                or status != "completed"
                and predictor is not None
                or status == "pending"
                and checkpoint is not None
                or status == "running"
                and checkpoint is None
            ):
                raise ArtifactError("Run member status and required artifacts disagree")
            if (member["failure"] is not None) != (status == "failed"):
                raise ArtifactError("Run member failure diagnostics disagree with status")
            _failure(member["failure"])
            if (
                not isinstance(member["data_identity"], str)
                or re.fullmatch(r"[0-9a-f]{64}", member["data_identity"]) is None
            ):
                raise ArtifactError("Run member data identity is invalid")
        if record["expected_member_ids"] != identities or len(set(identities)) != len(identities):
            raise ArtifactError("Ensemble run composition is missing, duplicated or reordered")
        architecture = record["architecture"]
        if architecture is not None and (
            not isinstance(architecture, str) or re.fullmatch(r"[0-9a-f]{64}", architecture) is None
        ):
            raise ArtifactError("Ensemble run architecture contract is invalid")
        if architecture is None and any(
            member["checkpoint"] is not None for member in record["members"]
        ):
            raise ArtifactError("Ensemble run is missing its architecture contract")
        archive.finish()
        return record


def _config(record):
    require_fields(record, set(FitConfig.__dataclass_fields__), "Run fit config")
    values = dict(record)
    require_fields(values["solver"], set(SolverConfig.__dataclass_fields__), "Run solver config")
    values["solver"] = SolverConfig(**values["solver"])
    return FitConfig(**values)


def _compatible_config(old, new):
    before, after = asdict(old), asdict(new)
    before.pop("epochs")
    after.pop("epochs")
    if _json_hash(before) != _json_hash(after) or new.epochs < old.epochs:
        raise ArtifactError(
            "Ensemble continuation requires identical configuration and no reduced epoch budget"
        )


def _data_identity(region, observations, fold):
    inputs = _fold_inputs(region, observations, fold)
    validation = tuple(
        (prepared, observed, fold.validation[prepared.name])
        for prepared, observed, _ in inputs
        if prepared.name in fold.validation
    )
    return _fit_data_identity(inputs, validation)


class _ContentWriter:
    def add(self, value):
        array = np.asarray(value)
        return {
            "shape": list(array.shape),
            "dtype": array.dtype.str,
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }


def _saved_member(directory, record, run, config):
    identity, fold = _identity(record["identity"]), _fold(record["fold"])
    state = predictor = None
    if record["checkpoint"] is not None:
        state = _load_reference(
            directory, identity.member_id, record["checkpoint"], "training_checkpoint"
        )
        _compatible_config(state.config, replace(config, seed=identity.effective_seed))
        if (
            state.data_identity != record["data_identity"]
            or state.epoch != record["epoch"]
            or _architecture(state.encoder) != run["architecture"]
            or _architecture(state.best_predictor.encoder) != run["architecture"]
        ):
            raise ArtifactError(
                f"Member {identity.member_id}: checkpoint data, progress or model contract differs"
            )
    if record["status"] == "completed":
        predictor = _load_reference(directory, identity.member_id, record["predictor"], "predictor")
        if state.epoch != _config(run["config"]).epochs:
            raise ArtifactError(
                f"Member {identity.member_id}: completed progress does not match saved budget"
            )
        if _json_hash(encode_predictor(predictor, _ContentWriter())) != _json_hash(
            encode_predictor(state.best_predictor, _ContentWriter())
        ):
            raise ArtifactError(
                f"Member {identity.member_id}: selected predictor and checkpoint disagree"
            )
    return identity, fold, state, predictor


def _completed_member(identity, fold, predictor, state):
    result = FitResult(
        predictor, state.history, state.selected_epoch, state.selection, state.region_names, state
    )
    return EnsembleMember(identity, fold, "completed", predictor, result)


def fit_ensemble_run(
    directory,
    region,
    observations,
    *,
    folds=None,
    n_folds=None,
    holdout_size=None,
    fold_seed=0,
    query_regime="both_unseen",
    initialization_seeds=None,
    config=None,
    model_factory=None,
    resume=False,
    on_epoch=None,
    on_member=None,
) -> Ensemble:
    """Fit a durable sequential run, skipping compatible members already at budget.

    Resume validates all saved data/configuration/composition before any work.
    Completed old-budget members resume full state when total budget increases.
    The factory initializes pending members only; it never overrides saved models.
    Callbacks run after durable publication. Use one writer per run directory.
    """
    directory = Path(directory).resolve()
    if type(resume) is not bool:
        raise ArtifactError("resume must be a boolean")
    manifest = directory / "run.ilg"
    if manifest.exists() and not resume:
        raise ArtifactError("An ensemble run already exists; use resume=True")
    if resume and not manifest.is_file():
        raise ArtifactError("Cannot resume: ensemble run manifest is missing")
    try:
        saved = _read_run(directory) if resume else None
        if saved is not None:
            previous = _config(saved["config"])
            config = config or previous
            _compatible_config(previous, config)
            if folds is None and n_folds is None and holdout_size is None:
                distinct = {_fold(m["fold"]).fold_id: _fold(m["fold"]) for m in saved["members"]}
                folds = tuple(distinct.values())
            if initialization_seeds is None:
                initialization_seeds = sorted(
                    {m["identity"]["initialization_seed"] for m in saved["members"]}
                )
        config = config or FitConfig()
        folds, seeds, identities = _prepare_ensemble(
            region,
            observations,
            config=config,
            folds=folds,
            n_folds=n_folds,
            holdout_size=holdout_size,
            fold_seed=fold_seed,
            query_regime=query_regime,
            initialization_seeds=initialization_seeds,
        )
        jobs = [
            (fold, identity)
            for fold in folds
            for identity in identities
            if identity.fold_id == fold.fold_id
        ]
        inputs = _fold_inputs(region, observations, folds[0])
        contracts = [
            {
                "region": r.name,
                "target": asdict(o.target),
                "feature_count": r.features.shape[-1],
                "feature_names": r.feature_names,
            }
            for r, o, _ in inputs
        ]
        members = [
            {
                "identity": asdict(identity),
                "fold": asdict(fold),
                "status": "pending",
                "failure": None,
                "data_identity": _data_identity(region, observations, fold),
                "checkpoint": None,
                "predictor": None,
                "epoch": None,
            }
            for fold, identity in jobs
        ]
        expected = [identity.member_id for identity in identities]
        restored = {}
        if saved is not None:
            if (
                saved["expected_member_ids"] != expected
                or _json_hash(saved["contracts"]) != _json_hash(contracts)
                or any(
                    _json_hash({k: old[k] for k in ("identity", "fold", "data_identity")})
                    != _json_hash({k: new[k] for k in ("identity", "fold", "data_identity")})
                    for old, new in zip(saved["members"], members, strict=True)
                )
            ):
                raise ArtifactError("Ensemble run data, fold or requested composition changed")
            restored = {
                m["identity"]["member_id"]: _saved_member(directory, m, saved, config)
                for m in saved["members"]
            }
            record = saved
            record["config"] = asdict(config)
            for member in record["members"]:
                if member["status"] == "completed" and member["epoch"] < config.epochs:
                    member["status"], member["predictor"] = "running", None
        else:
            record = {
                "run_version": 1,
                "expected_member_ids": expected,
                "config": asdict(config),
                "contracts": contracts,
                "architecture": None,
                "members": members,
            }
        directory.mkdir(parents=True, exist_ok=True)
        member_directory = directory / "members"
        if member_directory.is_symlink():
            raise ArtifactError("Ensemble member directory cannot be a symlink")
        member_directory.mkdir(exist_ok=True)
        _publish(directory, record)
    except ArtifactError:
        raise
    except (TypeError, KeyError, ValueError, AttributeError) as error:
        raise ArtifactError(f"Invalid ensemble run: {error}") from error

    outcomes = []
    for (fold, identity), member_record in zip(jobs, record["members"], strict=True):
        _, _, state, predictor = restored.get(identity.member_id, (None, None, None, None))
        if member_record["status"] == "failed":
            member = EnsembleMember(
                identity, fold, "failed", failure=_failure(member_record["failure"])
            )
        elif member_record["status"] == "completed":
            member = _completed_member(identity, fold, predictor, state)
        else:

            def progress(current_identity, current_state, member_record=member_record):
                architecture = _architecture(current_state.encoder)
                if record["architecture"] is not None and record["architecture"] != architecture:
                    raise ArtifactError(
                        f"Member {current_identity.member_id}: incompatible encoder architecture"
                    )
                checkpoint = _reference(
                    directory, current_identity.member_id, current_state, "training_checkpoint"
                )
                record["architecture"] = architecture
                member_record.update(
                    status="running",
                    checkpoint=checkpoint,
                    predictor=None,
                    epoch=current_state.epoch,
                )
                _publish(directory, record)
                if on_epoch is not None:
                    on_epoch(current_identity, current_state)

            member = fit_ensemble_member(
                region,
                observations,
                fold=fold,
                initialization_seed=identity.initialization_seed,
                config=config,
                model_factory=model_factory,
                state=state,
                on_epoch=progress,
            )
            if member.status == "completed":
                # Zero-update finalization can occur after interruption of final publication.
                if member_record["checkpoint"] is None:
                    progress(identity, member.fit_result.state)
                predictor_reference = _reference(
                    directory, identity.member_id, member.predictor, "predictor"
                )
                member_record.update(
                    status="completed", predictor=predictor_reference, failure=None
                )
            else:
                member_record.update(
                    status="failed", failure=asdict(member.failure), predictor=None
                )
            _publish(directory, record)
        outcomes.append(member)
        if on_member is not None:
            on_member(member)
    return Ensemble(tuple(outcomes), tuple(expected))
