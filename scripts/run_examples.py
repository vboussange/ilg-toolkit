"""Run the examples and write their results into the repository.

Run from any directory, with the examples extra:

    uv run --extra examples python scripts/run_examples.py [NAME ...]

An example is a script or a notebook in ``examples/``, named by its file stem.
Without names, every example runs. Examples run in this Python environment with
float64 enabled, so on the GPU when JAX has CUDA, where they select the NVIDIA CUDA
Direct Sparse Solver through the ``cudss`` extra. Each executed notebook is
written back in place with its outputs. Each example's Markdown page goes to
``docs/examples/NAME.md`` and its figures to ``docs/examples/NAME_files/``,
replacing that example's previous page and figures. Before running anything, the
script fails unless the Python code block in the README's Quickstart section
appears verbatim in ``examples/quickstart.py``.
"""

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path

import nbformat
from jupyter_client.kernelspec import KernelSpecManager
from jupyter_client.manager import KernelManager
from nbconvert import MarkdownExporter
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.writers.files import FilesWriter


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
PAGES = ROOT / "docs" / "examples"
README = ROOT / "README.md"
QUICKSTART = EXAMPLES / "quickstart.py"
SUFFIXES = (".py", ".ipynb")
FIGURE_SUFFIXES = (".png", ".svg", ".jpg", ".jpeg", ".gif", ".webp")
FLOAT64 = {"JAX_ENABLE_X64": "true"}
KERNEL = "python3"


def check_readme_excerpt(readme: Path, script: Path) -> None:
    """Raises unless every README Quickstart code block appears verbatim in the script.

    Each block matches whole lines of the script.

    Args:
        readme: The README whose `## Quickstart` section holds fenced Python blocks.
        script: The quickstart script the blocks are excerpted from.

    Raises:
        ValueError: If the section or its Python blocks are missing or empty, or a block
            does not appear verbatim, as whole lines, in the script.
    """
    section = re.search(
        r"^## Quickstart\n(.*?)(?=^## |\Z)",
        readme.read_text(encoding="utf-8"),
        flags=re.DOTALL | re.MULTILINE,
    )
    if section is None:
        raise ValueError(f"{readme.name} has no '## Quickstart' section.")
    blocks = re.findall(r"^```python\n(.*?)^```$", section.group(1), flags=re.DOTALL | re.MULTILINE)
    if not blocks or not all(block.strip() for block in blocks):
        raise ValueError(
            f"The Quickstart section of {readme.name} has an empty or no Python block."
        )
    lines = "\n" + script.read_text(encoding="utf-8")
    for block in blocks:
        if "\n" + block not in lines:
            raise ValueError(
                f"A Python code block in the Quickstart section of {readme.name} does not "
                f"appear verbatim, as whole lines, in {script.relative_to(ROOT)}."
            )


def find_examples(names: list[str]) -> list[Path]:
    """Returns the example sources with the given stems, or all of them without names.

    Args:
        names: Example names, the file stems of scripts and notebooks in `examples/`.

    Returns:
        The example sources, sorted by name.

    Raises:
        ValueError: If two examples share a name or a name matches no example.
    """
    sources = sorted(path for path in EXAMPLES.iterdir() if path.suffix in SUFFIXES)
    by_name = {path.stem: path for path in sources}
    if len(by_name) != len(sources):
        raise ValueError(f"Two examples in {EXAMPLES} share a file stem.")
    unknown = sorted(set(names) - set(by_name))
    if unknown:
        raise ValueError(f"Unknown examples {unknown}; available: {sorted(by_name)}.")
    return [by_name[name] for name in sorted(set(names))] if names else sources


def source_url(source: Path) -> str:
    """Returns the URL of an example source on the repository's main branch."""
    with open(ROOT / "zensical.toml", "rb") as file:
        repo_url = tomllib.load(file)["project"]["repo_url"]
    return f"{repo_url.rstrip('/')}/blob/main/{source.relative_to(ROOT).as_posix()}"


def clear_page(name: str) -> None:
    """Removes an example's Markdown page and figures from the documentation."""
    (PAGES / f"{name}.md").unlink(missing_ok=True)
    shutil.rmtree(PAGES / f"{name}_files", ignore_errors=True)


def run_script(source: Path, *, timeout: int) -> None:
    """Runs an example script and writes its page with its source, output and figures.

    The script runs in a temporary working directory; image files it writes there
    are its figures.

    Args:
        source: The example script.
        timeout: Maximum seconds for the script.

    Raises:
        RuntimeError: If the script fails.
    """
    name = source.stem
    with tempfile.TemporaryDirectory(prefix=f"ilg-{name}-") as work:
        try:
            result = subprocess.run(
                [sys.executable, str(source)],
                cwd=work,
                env=os.environ | FLOAT64 | {"MPLBACKEND": "Agg"},
                stdout=subprocess.PIPE,
                text=True,
                timeout=timeout,
                check=True,
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"{source.name} failed. Its output:\n{error.stdout}") from error
        figures = sorted(path for path in Path(work).iterdir() if path.suffix in FIGURE_SUFFIXES)
        clear_page(name)
        if figures:
            (PAGES / f"{name}_files").mkdir(parents=True)
        for figure in figures:
            shutil.copyfile(figure, PAGES / f"{name}_files" / figure.name)
    docstring = ast.get_docstring(ast.parse(source.read_text(encoding="utf-8"))) or ""
    summary = docstring.split("\n")[0]
    images = "".join(
        f"![{summary or figure.stem}]({name}_files/{figure.name})\n\n" for figure in figures
    )
    page = (
        f"# {name.replace('_', ' ').capitalize()}\n\n"
        f"[View `{source.name}` on GitHub]({source_url(source)})\n\n"
        f"```python\n{source.read_text(encoding='utf-8').rstrip()}\n```\n\n"
        f"## Output\n\n```text\n{result.stdout.rstrip()}\n```\n\n"
        f"{images}"
    )
    (PAGES / f"{name}.md").write_text(page.rstrip() + "\n", encoding="utf-8")


def run_notebook(source: Path, kernels: KernelSpecManager, *, timeout: int) -> None:
    """Executes an example notebook in place and writes its page and figures.

    The notebook runs from its own directory, as when opened in Jupyter.

    Args:
        source: The example notebook.
        kernels: Kernel specifications that provide the `python3` kernel.
        timeout: Maximum seconds for a single cell.
    """
    name = source.stem
    notebook = nbformat.read(source, as_version=4)
    kernel = KernelManager(kernel_name=KERNEL, kernel_spec_manager=kernels)
    executor = ExecutePreprocessor(
        timeout=timeout, kernel_name=KERNEL, force_raise_errors=True, record_timing=False
    )
    try:
        executor.preprocess(
            notebook, resources={"metadata": {"path": str(source.parent)}}, km=kernel
        )
    finally:
        # nbconvert leaves caller-owned kernel managers alive; release the kernel.
        if executor.kc is not None:
            executor.kc.stop_channels()
        if kernel.has_kernel:
            kernel.shutdown_kernel(now=True)
        kernel.cleanup_resources()
    nbformat.write(notebook, source)
    body, resources = MarkdownExporter().from_notebook_node(
        notebook, resources={"unique_key": name, "output_files_dir": f"{name}_files"}
    )
    link = f"[View `{source.name}` on GitHub]({source_url(source)})\n\n"
    title, _, rest = body.partition("\n")
    page = f"{title}\n\n{link}{rest.lstrip()}" if title.startswith("# ") else link + body
    clear_page(name)
    FilesWriter(build_directory=str(PAGES)).write(page, resources, notebook_name=name)


def python_kernel(directory: Path) -> KernelSpecManager:
    """Returns kernel specifications whose `python3` kernel is this interpreter with float64.

    Args:
        directory: Directory that receives the kernel specification.
    """
    spec = directory / KERNEL
    spec.mkdir()
    (spec / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "Python 3",
                "language": "python",
                "env": FLOAT64 | {"MPLBACKEND": "module://matplotlib_inline.backend_inline"},
            }
        ),
        encoding="utf-8",
    )
    return KernelSpecManager(kernel_dirs=[str(directory)])


def main() -> None:
    """Runs the examples named on the command line, or all of them."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("names", nargs="*", metavar="NAME", help="Examples to run (file stems)")
    parser.add_argument(
        "--timeout", type=int, default=3600, help="Seconds allowed per script or notebook cell"
    )
    options = parser.parse_args()
    check_readme_excerpt(README, QUICKSTART)
    sources = find_examples(options.names)
    PAGES.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ilg-kernels-") as directory:
        kernels = python_kernel(Path(directory))
        for source in sources:
            print(f"Running {source.name}", flush=True)
            started = time.monotonic()
            if source.suffix == ".py":
                run_script(source, timeout=options.timeout)
            else:
                run_notebook(source, kernels, timeout=options.timeout)
            print(f"Ran {source.name} in {time.monotonic() - started:.1f} s", flush=True)


if __name__ == "__main__":
    main()
