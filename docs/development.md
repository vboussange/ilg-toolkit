# Development and documentation

The authoritative examples are the five unexecuted notebooks in `examples/`.
General guides live at `docs/`, API directives at `docs/api/`, and the root
`GLOSSARY.md` is the only maintained glossary. Agent instructions are excluded
from the public build.

## Reproduce the checks

From the repository root with Python 3.12 or newer:

```bash
uv sync --extra dev --extra docs
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true
uv run --no-sync ruff format --check .
uv run --no-sync ruff check .
uv run --no-sync pyright
uv run --no-sync pytest
uv run --no-sync pytest experiments/wishart/test_diagnostic.py
uv run --no-sync python -m experiments.wishart
uv run --no-sync python -m build
```

The documentation pipeline uses the installed toolkit and the interpreter that
invokes it. Each notebook runs in a fresh Python session and its own temporary working
directory, on CPU with float64 enabled. Execution errors fail the build, including
cells tagged as expected errors. Sources must have empty outputs and execution
counts before building.

```bash
uv run --no-sync python scripts/build_docs.py
uv run --no-sync zensical build --strict
uv run --no-sync zensical serve
```

`build/docs/` contains disposable executed Markdown, image assets, downloadable
unexecuted notebook sources, and the staged canonical glossary. `site/` contains
the final website. Never commit either generated directory. After changing a
notebook, rerun preparation before serving or building. The preparation command's
`--timeout` option controls the per-cell execution limit in seconds (default 1200).

The documentation workflow executes the same collection and runs a strict site
build on pull requests and relevant source/docstring changes. Successful main
builds publish through the GitHub Pages deployment environment. The repository's
Pages source must be **GitHub Actions**.

## Release

Update `project.version` in `pyproject.toml` and merge into main. The release
workflow tests built distributions, tags `v<version>`, and publishes a GitHub
release. Releases are distributed through GitHub rather than PyPI.
