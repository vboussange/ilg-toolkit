"""Publish tested distributions, preserving and verifying staged evidence assets."""

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def release_state(repo, tag):
    releases = json.loads(
        run("gh", "release", "list", "--repo", repo, "--limit", "1000", "--json", "tagName,isDraft")
    )
    return next((release for release in releases if release["tagName"] == tag), None)


def verify_evidence(repo, version):
    config = json.loads((ROOT / ".github/release-assets.json").read_text()).get(version)
    if config is None:
        return
    if run("git", "rev-parse", "HEAD:src") != config["production_source_tree"]:
        raise ValueError("Production source differs from the measured release evidence")
    with tempfile.TemporaryDirectory() as directory:
        for asset in config["assets"]:
            run(
                "gh",
                "release",
                "download",
                f"v{version}",
                "--repo",
                repo,
                "--pattern",
                asset["name"],
                "--dir",
                directory,
            )
            path = Path(directory) / asset["name"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != asset["sha256"]:
                raise ValueError(f"Release evidence checksum mismatch: {asset['name']}")


def publish(repo, project, distributions):
    version = project["version"]
    tag = f"v{version}"
    state = release_state(repo, tag)
    if state is not None and not state["isDraft"]:
        raise ValueError(f"{tag} is already published")
    # Verify again immediately before publication, including on a retry.
    verify_evidence(repo, version)
    wheels = sorted(distributions.glob("*.whl"))
    sources = sorted(distributions.glob("*.tar.gz"))
    if not wheels or not sources:
        raise ValueError("Both tested wheel and source distribution are required")
    head = run("git", "rev-parse", "HEAD")
    existing = subprocess.run(
        ["git", "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if existing.returncode == 0:
        if existing.stdout.strip() != head:
            raise ValueError("Release tag does not identify the tested commit")
    else:
        run("git", "config", "user.name", "github-actions[bot]")
        run("git", "config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com")
        run("git", "tag", "-a", tag, "-m", f"{project['name']} {tag}")
        run("git", "push", "origin", f"refs/tags/{tag}")
    if state is None:
        run(
            "gh",
            "release",
            "create",
            tag,
            "--repo",
            repo,
            "--draft",
            "--verify-tag",
            "--title",
            f"{project['name']} {tag}",
            "--generate-notes",
        )
    # Draft uploads are retryable; all assets precede publication, including when
    # GitHub's immutable releases are enabled. Never replace evidence here.
    run("gh", "release", "upload", tag, "--repo", repo, "--clobber", *map(str, sources + wheels))
    run("gh", "release", "edit", tag, "--repo", repo, "--draft=false", "--verify-tag")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["status", "verify-evidence", "publish"])
    parser.add_argument("--distributions", type=Path)
    args = parser.parse_args()
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    repo = os.environ["GITHUB_REPOSITORY"]
    tag = f"v{project['version']}"
    if args.command == "status":
        state = release_state(repo, tag)
        needed = state is None or state["isDraft"]
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            for key in ("name", "version"):
                print(f"{key}={project[key]}", file=output)
            print(f"tag={tag}", file=output)
            print(f"needed={str(needed).lower()}", file=output)
        print(f"{tag}: " + ("release required" if needed else "already released"))
    elif args.command == "verify-evidence":
        verify_evidence(repo, project["version"])
    else:
        if args.distributions is None:
            parser.error("publish requires --distributions")
        publish(repo, project, args.distributions.resolve())


if __name__ == "__main__":
    main()
