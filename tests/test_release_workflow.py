"""Release staging must preserve evidence and refuse incomplete publication."""

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / ".github/scripts/github-release.py"


@pytest.fixture
def release(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("github_release", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    (tmp_path / ".github").mkdir()
    (tmp_path / "pyproject.toml").write_text('[project]\nname="ilg-toolkit"\nversion="0.0.3"\n')
    payload = b"synthetic evidence archive"
    (tmp_path / ".github/release-assets.json").write_text(
        json.dumps(
            {
                "0.0.3": {
                    "production_source_tree": "measured-src",
                    "assets": [
                        {"name": "evidence.tar.gz", "sha256": hashlib.sha256(payload).hexdigest()}
                    ],
                }
            }
        )
    )
    state: dict = dict(draft=True, exists=True, payload=payload, source="measured-src", tag="head")
    calls = []

    def run(*args):
        calls.append(args)
        if args[:3] == ("gh", "release", "list"):
            return json.dumps(
                [{"tagName": "v0.0.3", "isDraft": state["draft"]}] if state["exists"] else []
            )
        if args[:3] == ("gh", "release", "download"):
            if state["payload"] is None:
                raise subprocess.CalledProcessError(1, args)
            (Path(args[args.index("--dir") + 1]) / "evidence.tar.gz").write_bytes(state["payload"])
        if args[:2] == ("git", "rev-parse"):
            return state["source"] if args[2] == "HEAD:src" else "head"
        return ""

    monkeypatch.setattr(module, "run", run)
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess(
            a, 0 if state["tag"] else 1, stdout=state["tag"] or "", stderr=""
        ),
    )
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "output"))
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "ilg_toolkit-0.0.3.whl").write_bytes(b"tested wheel")
    (dist / "ilg_toolkit-0.0.3.tar.gz").write_bytes(b"tested source")
    return module, state, calls, dist


@pytest.mark.parametrize(
    "exists,draft,needed", [(False, False, True), (True, True, True), (True, False, False)]
)
def test_only_published_releases_skip_builds(release, monkeypatch, exists, draft, needed):
    module, state, _, _ = release
    state.update(exists=exists, draft=draft)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "status"])
    module.main()
    assert f"needed={str(needed).lower()}" in (module.ROOT / "output").read_text()


@pytest.mark.parametrize(
    "problem", ["missing", "checksum", "source", "tag", "published", "distributions"]
)
def test_invalid_release_never_publishes(release, problem):
    module, state, calls, dist = release
    if problem == "missing":
        state["payload"] = None
    elif problem == "checksum":
        state["payload"] = b"changed evidence"
    elif problem == "source":
        state["source"] = "unmeasured-source"
    elif problem == "tag":
        state["tag"] = "untested-commit"
    elif problem == "published":
        state["draft"] = False
    else:
        (dist / "ilg_toolkit-0.0.3.whl").unlink()
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        module.publish("owner/repo", {"name": "ilg-toolkit", "version": "0.0.3"}, dist)
    assert not any(
        call[:3] in [("gh", "release", "edit"), ("gh", "release", "upload")] for call in calls
    )


def test_staged_draft_preserves_evidence_and_uploads_distributions_before_publication(release):
    module, _, calls, dist = release
    module.publish("owner/repo", {"name": "ilg-toolkit", "version": "0.0.3"}, dist)
    assert not any(call[:3] == ("gh", "release", "create") for call in calls)
    upload = next(call for call in calls if call[:3] == ("gh", "release", "upload"))
    assert str(dist / "ilg_toolkit-0.0.3.whl") in upload
    assert str(dist / "ilg_toolkit-0.0.3.tar.gz") in upload
    assert not any("evidence.tar.gz" in arg for arg in upload)
    assert calls[-1][:3] == ("gh", "release", "edit")
    assert "--draft=false" in calls[-1] and "--verify-tag" in calls[-1]


def test_release_without_evidence_creates_draft_then_publishes(release):
    module, state, calls, dist = release
    state.update(exists=False, tag=None)
    module.publish("owner/repo", {"name": "ilg-toolkit", "version": "0.0.4"}, dist)
    assert not any(call[:3] == ("gh", "release", "download") for call in calls)
    create = next(call for call in calls if call[:3] == ("gh", "release", "create"))
    assert "--draft" in create and "--verify-tag" in create
    assert any(call[:3] == ("git", "push", "origin") for call in calls)
    assert calls.index(create) < next(
        i for i, call in enumerate(calls) if call[:3] == ("gh", "release", "upload")
    )
    assert calls[-1][:3] == ("gh", "release", "edit")


def test_failed_upload_leaves_draft_unpublished_and_retry_completes(release, monkeypatch):
    module, _, calls, dist = release
    successful_run = module.run

    def failed_upload(*args):
        if args[:3] == ("gh", "release", "upload"):
            raise subprocess.CalledProcessError(1, args)
        return successful_run(*args)

    monkeypatch.setattr(module, "run", failed_upload)
    with pytest.raises(subprocess.CalledProcessError):
        module.publish("owner/repo", {"name": "ilg-toolkit", "version": "0.0.3"}, dist)
    assert not any(call[:3] == ("gh", "release", "edit") for call in calls)
    monkeypatch.setattr(module, "run", successful_run)
    module.publish("owner/repo", {"name": "ilg-toolkit", "version": "0.0.3"}, dist)
    assert calls[-1][:3] == ("gh", "release", "edit")
