"""Execute authoritative examples and stage disposable Zensical inputs.

Run with the installed package and docs extra. Each notebook receives its own
working directory and a kernel from this Python interpreter; source files are
never executed in place. Zensical then reads build/docs via zensical.toml.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbconvert import MarkdownExporter
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.writers import FilesWriter

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ("ibd", "mlpe_workflow", "ensemble", "multispecies")


def _render_quickstart(
    source: Path,
    output_dir: Path,
    scratch: Path,
    environment: dict[str, str],
    *,
    timeout: int,
) -> None:
    """Execute the quickstart script and render its source and output."""
    started = time.monotonic()
    print(f"Executing {source.name} on CPU with float64", flush=True)
    work = scratch / "quickstart"
    work.mkdir()
    try:
        result = subprocess.run(
            [sys.executable, str(source)],
            cwd=work,
            env=os.environ | environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=True,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"Quickstart execution failed:\n{error.stdout}") from error
    body = (
        "# Quickstart\n\n"
        f"[Download the Python script]({source.name})\n\n"
        f"```python\n{source.read_text(encoding='utf-8').rstrip()}\n```\n\n"
        f"## Output\n\n```text\n{result.stdout.rstrip()}\n```\n"
    )
    (output_dir / "quickstart.md").write_text(body, encoding="utf-8")
    shutil.copyfile(source, output_dir / source.name)
    print(f"Rendered {source.name} in {time.monotonic() - started:.1f}s", flush=True)


def prepare(example_dir: Path, output_dir: Path, *, timeout: int) -> None:
    """Execute examples and stage documentation with their source downloads.

    Args:
        example_dir: Directory containing the quickstart script and notebooks.
        output_dir: Disposable directory for generated documentation inputs.
        timeout: Maximum seconds for the quickstart or a single notebook cell.
    """
    example_dir, output_dir = example_dir.resolve(), output_dir.resolve()
    quickstart = example_dir / "quickstart.py"
    if not quickstart.is_file():
        raise ValueError(f"Quickstart script not found: {quickstart}")
    sources = sorted(example_dir.glob("*.ipynb"))
    if not sources:
        raise ValueError(f"No notebooks found in {example_dir}")
    if example_dir == ROOT / "examples":
        if {source.stem for source in sources} != set(NOTEBOOKS):
            raise ValueError(f"Expected exactly the four authoritative notebooks: {NOTEBOOKS}")
        sources = [example_dir / f"{name}.ipynb" for name in NOTEBOOKS]
    for source in (ROOT / "docs", example_dir, ROOT / "GLOSSARY.md", ROOT / "README.md"):
        if output_dir == source or output_dir in source.parents or source in output_dir.parents:
            raise ValueError("Output must be separate from documentation and notebook sources")
    for source in sources:
        notebook = nbformat.read(source, as_version=4)
        nbformat.validate(notebook)
        if any(
            cell.cell_type == "code" and (cell.outputs or cell.execution_count is not None)
            for cell in notebook.cells
        ):
            raise ValueError(
                f"Commit notebook sources without outputs or execution counts: {source}"
            )

    # This directory contains generated files only; a failed run must not leave
    # stale notebook pages that could be mistaken for successful execution.
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(ROOT / "docs", output_dir, ignore=shutil.ignore_patterns("agents", "adr"))
    shutil.copyfile(ROOT / "README.md", output_dir / "index.md")
    shutil.copyfile(ROOT / "GLOSSARY.md", output_dir / "glossary.md")
    examples = output_dir / "examples"
    examples.mkdir(exist_ok=True)
    environment = {
        "JAX_PLATFORMS": "cpu",
        "JAX_ENABLE_X64": "true",
        "MPLBACKEND": "module://matplotlib_inline.backend_inline",
    }
    with tempfile.TemporaryDirectory(prefix="ilg-docs-") as temporary:
        scratch = Path(temporary)
        _render_quickstart(quickstart, examples, scratch, environment, timeout=timeout)
        kernel_dir = scratch / "kernels" / "ilg-docs"
        kernel_dir.mkdir(parents=True)
        (kernel_dir / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                    "display_name": "ILG documentation",
                    "language": "python",
                    "env": environment,
                }
            ),
            encoding="utf-8",
        )
        manager = KernelSpecManager(kernel_dirs=[str(kernel_dir.parent)])
        for source in sources:
            started = time.monotonic()
            print(f"Executing {source.name} on CPU with float64", flush=True)
            work = scratch / source.stem
            work.mkdir()
            notebook = nbformat.read(source, as_version=4)
            kernel = KernelManager(kernel_name="ilg-docs", kernel_spec_manager=manager)
            executor = ExecutePreprocessor(
                timeout=timeout, kernel_name="ilg-docs", force_raise_errors=True
            )
            try:
                executor.preprocess(
                    notebook,
                    resources={"metadata": {"path": str(work)}},
                    km=kernel,
                )
            finally:
                # nbconvert leaves caller-owned kernel managers alive. Release
                # each kernel before the next notebook, including on failure.
                if executor.kc is not None:
                    executor.kc.stop_channels()
                if kernel.has_kernel:
                    kernel.shutdown_kernel(now=True)
                kernel.cleanup_resources()
            body, resources = MarkdownExporter().from_notebook_node(
                notebook,
                resources={
                    "unique_key": source.stem,
                    "output_files_dir": f"{source.stem}_files",
                },
            )
            download = f"[Download the unexecuted notebook]({source.name})\n\n"
            FilesWriter(build_directory=str(examples)).write(
                download + body, resources, notebook_name=source.stem
            )
            shutil.copyfile(source, examples / source.name)
            print(f"Rendered {source.name} in {time.monotonic() - started:.1f}s", flush=True)


def main() -> None:
    """Prepare documentation using command-line source and execution settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples-dir", type=Path, default=ROOT / "examples")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build" / "docs")
    parser.add_argument(
        "--timeout", type=int, default=1200, help="Script and per-cell timeout in seconds"
    )
    options = parser.parse_args()
    prepare(options.examples_dir, options.output_dir, timeout=options.timeout)


if __name__ == "__main__":
    main()
