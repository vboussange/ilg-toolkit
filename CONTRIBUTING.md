# Contributing

Contributions (pull requests) are very welcome! Here's how to get started.

---

### Getting started

[We assume that you have `uv` installed.](https://docs.astral.sh/uv/) Now fork the library on GitHub. Then clone and install the library:

```bash
git clone https://github.com/your-username-here/ilg-toolkit.git
cd ilg-toolkit
uv run prek install  # Creates a local venv + installs dependencies + installs pre-commit hooks.
```

---

### If you're making changes to the code

Now make your changes. Make sure to include additional tests if necessary. Next verify the tests all pass:

```bash
uv run pytest -n 8
```

Then push your changes back to your fork of the repository:

```bash
git push
```

Finally, open a pull request on GitHub!

---

### If you're making changes to the documentation

Make your changes. You can then build the documentation by doing

```bash
uv run zensical serve
```

You can then see your local copy of the documentation by navigating to `localhost:8000` in a web browser.
The site renders the committed example results; see [Running the examples](#running-the-examples).

---

### Docstring format

Use [Google-style Python docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
throughout the codebase. The summary line of a function or method opens with a
descriptive third-person verb whose subject is the function ("Returns the
conductance surface ..."); a class's opens with what the object is ("A region
batch ..."); a property's reads like an attribute ("The target contract ...").

The sections are `Args:`, `Returns:`, `Raises:`, `Attributes:`, `Notes:`, and
`Examples:`. Indent section contents by four spaces. Describe units, array shapes,
ordering, constraints, and scientific meaning; type annotations describe Python
types. Keep substantive examples for prediction and calibration.

Document initialization on the class; constructors and standard lifecycle
methods carry no duplicate docstrings. Document public methods and properties.
Ruff checks the Google convention, and mkdocstrings renders it in the API
reference. Review scientific meaning and parameter coverage alongside those
automated style checks. The Prose section of `AGENTS.md` sets the wording rules
for docstrings, including the required `Args:` sections.

---

### Running the examples

CI never runs the examples: the documentation and GitHub show the executed notebooks
and example pages committed in the repository. After changing an example, or any API
an example calls, run the examples again from the repository root:

```bash
uv run --extra examples python scripts/run_examples.py             # Every example.
uv run --extra examples python scripts/run_examples.py quickstart  # Only the named examples.
```

The script runs each example in this environment with float64 enabled. It writes each
executed notebook back into `examples/`, and each example's Markdown page and figures
into `docs/examples/`. It fails when the Python code block in the README's Quickstart
section is not verbatim in `examples/quickstart.py`.

Run the examples on an NVIDIA GPU: install JAX with CUDA support into the project
environment with `uv pip install "jax[cuda12]==<locked jax version>"`, and select the GPU
with `CUDA_VISIBLE_DEVICES`. On an NVIDIA RTX 3090, all examples run in about five minutes.

Commit the regenerated notebooks, pages and figures together with the change that
caused them.

---

### Running the Wishart experiment

The experimental Wishart diagnostic in `experiments/wishart/` is not part of the
test suite. After changing it, or any API it calls, run its tests and its
reference run from the repository root:

```bash
JAX_ENABLE_X64=true uv run pytest experiments/wishart/test_diagnostic.py
JAX_ENABLE_X64=true uv run python -m experiments.wishart
```
