"""Execute authoritative notebooks and stage disposable Zensical inputs.

Run with the installed package and docs extra. Each notebook receives its own
working directory and a kernel from this Python interpreter; source files are
never executed in place. Zensical then reads build/docs via zensical.toml.
"""

import argparse
import json
import shutil
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
NOTEBOOKS = ("quickstart", "ibd", "mlpe_workflow", "ensemble", "multispecies")


def prepare(notebook_dir: Path, output_dir: Path, *, timeout: int) -> None:
    """Execute sources, export rendered pages/assets, and retain source downloads."""
    notebook_dir, output_dir = notebook_dir.resolve(), output_dir.resolve()
    sources = sorted(notebook_dir.glob("*.ipynb"))
    if not sources:
        raise ValueError(f"No notebooks found in {notebook_dir}")
    if notebook_dir == ROOT / "examples":
        if {source.stem for source in sources} != set(NOTEBOOKS):
            raise ValueError(f"Expected exactly the five authoritative notebooks: {NOTEBOOKS}")
        sources = [notebook_dir / f"{name}.ipynb" for name in NOTEBOOKS]
    for source in (ROOT / "docs", notebook_dir, ROOT / "GLOSSARY.md"):
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook-dir", type=Path, default=ROOT / "examples")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build" / "docs")
    parser.add_argument("--timeout", type=int, default=1200, help="Per-cell timeout in seconds")
    options = parser.parse_args()
    prepare(options.notebook_dir, options.output_dir, timeout=options.timeout)


if __name__ == "__main__":
    main()
