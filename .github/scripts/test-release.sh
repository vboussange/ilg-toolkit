#!/usr/bin/env bash
# Test each installed distribution outside the checkout, including AMG support.
set -euo pipefail

source_root=${GITHUB_WORKSPACE:-$PWD}
distributions=$(realpath "$1")

for artifact in "$distributions"/*.tar.gz "$distributions"/*.whl; do
  (
    workdir=$(mktemp -d)
    export UV_PROJECT_ENVIRONMENT="$workdir/.venv"
    unset PYTHONPATH
    uv venv --python 3.12 "$UV_PROJECT_ENVIRONMENT"
    uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" "$artifact"

    # The package-boundary test also builds a wheel. Keep its source fixture
    # separate from the environment so its isolated subprocess sees dependencies.
    # The README test resolves the README's links, including those into docs.
    mkdir -p "$workdir/source/src"
    cp -R "$source_root/tests" "$source_root/examples" "$source_root/experiments" \
      "$source_root/docs" "$source_root/pyproject.toml" "$source_root/README.md" \
      "$source_root/LICENSE" "$workdir/source/"
    cp -R "$source_root/src/ilg_toolkit" "$workdir/source/src/"
    cd "$workdir/source"
    uv sync --extra dev --extra amg --no-install-project --inexact
    uv run --no-sync python - <<'PY'
from pathlib import Path
import importlib.metadata
import tomllib
import ilg_toolkit

project = tomllib.loads(Path('pyproject.toml').read_text())['project']
assert importlib.metadata.version(project['name']) == project['version']
assert not Path(ilg_toolkit.__file__).resolve().is_relative_to(Path.cwd())
print(f"Testing installed {project['name']} {project['version']} from {ilg_toolkit.__file__}")
PY
    uv run --no-sync pytest
  )
done
