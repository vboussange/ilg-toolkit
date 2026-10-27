# Contributing

## Development checks

From the repository root with Python 3.12 or newer:

```bash
uv sync --extra dev --extra docs --extra amg --extra examples
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true
uv run --no-sync ruff format --check .
uv run --no-sync ruff check .
uv run --no-sync pyright
uv run --no-sync pytest -n 4
uv run --no-sync pytest experiments/wishart/test_diagnostic.py
uv run --no-sync python -m experiments.wishart
uv run --no-sync python examples/quickstart.py
uv run --no-sync python -m build
```

Plain `pytest` runs serially, so a many-core workstation does not start one JAX
process per core; pass `-n` to run tests in parallel with pytest-xdist. Generic
behavior is tested with cheap analytic distance models, and ResNet9 and U-Net
are each covered once at tiny width, so test time follows the number of
behaviors rather than network size. Keep each parametrization on a distinct code
path, since every compiled program costs compilation time.

The **Run tests** workflow runs on pull requests and every push to main. It
checks formatting, lint, and types, runs the full CPU test suite in parallel
including algebraic multigrid (AMG), tests and runs the Wishart diagnostic, runs
the examples, and checks package installation. Dependency bounds are maintained
in `pyproject.toml`; locally generated `uv.lock` files are ignored.

## Documentation and examples

The documentation consists of the overview generated from README, the
executable examples, the glossary, and the API reference. The authoritative
examples are `examples/quickstart.py` and the four unexecuted notebooks in
`examples/`. API directives live in `docs/api/`, and the root `GLOSSARY.md` is
the maintained domain glossary. Statistical definitions and extension guidance
belong in docstrings, next to the code they describe. README is copied unchanged
to the documentation homepage during preparation; use absolute web links and
image URLs in README so both renderings work. Agent instructions and
contributor documentation remain outside the public site.

The quickstart trains on real data downloaded from the `example-data-v1`
pre-release, which is never marked latest; `uv run scripts/build_example_data.py`
rebuilds that asset from its inline (PEP 723) dependencies. Credit Cesar Australia (genetic data) and ESA WorldCover
2021 (CC BY 4.0) wherever the example data appears. The README's quickstart
code block must appear verbatim in `examples/quickstart.py`: documentation
preparation fails otherwise, and regenerates the quickstart figure that README
shows from the deployed site.

```bash
uv run --no-sync python scripts/build_docs.py
uv run --no-sync zensical build --strict
uv run --no-sync zensical serve
```

Documentation preparation uses the installed toolkit, with the docs and
examples extras, and the invoking Python interpreter. The quickstart and each notebook run in separate processes and
temporary working directories, on CPU with float64 enabled. Execution errors
fail the build, including notebook cells tagged as expected errors. Commit
notebook sources with empty outputs and execution counts. The `--timeout`
option sets the script and per-cell execution limit in seconds (default 1200).

`build/docs/` contains disposable executed Markdown, image assets including the
quickstart figure, downloadable example sources, the homepage, and the glossary. `site/` contains the final
website. Both directories are ignored. Rerun preparation after changing examples
or documentation before serving or building.

The documentation workflow executes the examples and builds the site strictly
on relevant pull requests and main pushes. Successful main builds deploy through
the GitHub Pages environment. Configure the repository's Pages source as
**GitHub Actions**.

### Docstrings and prose

Use [Google-style Python docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
throughout the codebase. Start with what the object is or does, using the
canonical terms from `GLOSSARY.md`. Add `Args:`, `Returns:`, `Raises:`,
`Attributes:`, `Notes:`, and `Examples:` sections when they clarify the contract.
Indent section contents by four spaces. Describe units, array shapes, ordering,
constraints, and scientific meaning; type annotations already describe Python
types. Preserve substantive examples for prediction and calibration.

Document initialization on the class; constructors and standard lifecycle
methods do not need duplicate docstrings. Document public methods and properties.
Ruff checks the Google convention, and mkdocstrings renders it in the API
reference. Review scientific meaning and parameter coverage alongside those
automated style checks.

Describe current behavior in clear US English, with one glossary term per
concept, following the paper's terminology mapped in `GLOSSARY.md`. ILG Toolkit
**provides utilities** and never learns: reserve **learned** for trained models'
parameters and surfaces. Training estimates distance-model and statistical
parameters; calibration fits a genetic relationship for a fixed distance model. Use **training** for the optimization workflow and retain
statistical terms such as **fitted calibration** where they describe the
operation. Write **region–species** with an en dash and expand abbreviations on
first use in each module or page. Use implementation terminology when it helps
readers make a numerical or programming decision. Explain limitations beside
the behavior they constrain, without historical comparisons or unsupported
claims of ecological inference.

## Releases

Release automation follows the
[Equinox/Lineax release action](https://github.com/patrick-kidger/action_update_python_project/tree/v8).
After **Run tests** succeeds for a push to main, the Release workflow checks out
that tested commit and invokes `action_update_python_project@v8` with Python
3.12. For a version newer than the latest PyPI release, the action builds the
source and wheel distributions, runs the full tests against each installed
distribution, then publishes to PyPI and creates a GitHub tag and release.
The workflow verifies imports come from the installed distribution rather than
the source checkout.

Publication is skipped until the repository has a `pypi_token` secret. The
initial PyPI publication is a separate maintainer operation: build and validate
the intended `0.0.3` distributions, upload them manually to PyPI, then configure
that secret with a token scoped to the project. This setup does not change the
current version or the existing draft release.

For later releases, update `project.version` in `pyproject.toml` and merge into
main. Successful tests for that commit enable the automated distribution tests
and publication. Keep credentials in GitHub secrets.
