#!/usr/bin/env bash
# Test the distribution already installed in the release action's environment.
set -euo pipefail

source_root=${GITHUB_WORKSPACE:?The release action must provide GITHUB_WORKSPACE}
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
unset PYTHONPATH

if [ ! -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]; then
  echo 'The release action must install the distribution in .venv before testing.' >&2
  exit 1
fi

# Keep the copied test suite separate from the checkout and its Python sources.
mkdir source
cp -R "$source_root/tests" "$source_root/pyproject.toml" source/
cd source
uv sync --extra dev --extra amg --no-install-project --inexact
uv run --no-sync python - <<'PY'
from pathlib import Path
import importlib.metadata
import os
import tomllib
import ilg_toolkit

project = tomllib.loads(Path('pyproject.toml').read_text())['project']
installed = Path(ilg_toolkit.__file__).resolve()
environment = Path(os.environ['UV_PROJECT_ENVIRONMENT']).resolve()
checkout = Path(os.environ['GITHUB_WORKSPACE']).resolve()
assert importlib.metadata.version(project['name']) == project['version']
assert installed.is_relative_to(environment), 'The package must come from the artifact environment'
assert not installed.is_relative_to(checkout), 'The package must not come from the checkout'
print(f"Testing installed {project['name']} {project['version']} from {installed}")
PY
uv run --no-sync pytest -n auto
