"""The public docs command executes notebooks and preserves publishable outputs."""

import subprocess
import sys
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat", reason="Documentation checks require the docs extra")
pytest.importorskip("nbconvert")
pytest.importorskip("matplotlib")
COMMAND = Path(__file__).resolve().parents[2] / "scripts" / "build_docs.py"


def test_clean_cpu_notebook_exports_plot_and_download_without_changing_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                "import os\nimport jax\nimport matplotlib.pyplot as plt\n"
                'print(os.environ["JAX_PLATFORMS"], jax.config.read("jax_enable_x64"))\n'
                "plt.plot([0, 1], [1, 2]); plt.show()\n"
            )
        ]
    )
    path = source / "example.ipynb"
    nbformat.write(notebook, path)
    original = path.read_bytes()
    output = tmp_path / "rendered"
    result = subprocess.run(
        [sys.executable, str(COMMAND), "--notebook-dir", str(source), "--output-dir", str(output)],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    page = (output / "examples" / "example.md").read_text()
    assert "cpu True" in page
    assert "(example.ipynb)" in page
    assert list((output / "examples").rglob("*.png"))
    assert (output / "examples" / "example.ipynb").read_bytes() == original
    assert path.read_bytes() == original
    assert (output / "glossary.md").read_bytes() == (
        COMMAND.parents[1] / "GLOSSARY.md"
    ).read_bytes()
    assert not (output / "agents").exists()


def test_error_cell_cannot_be_published_even_when_tagged_as_expected(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                'raise RuntimeError("cannot publish a failed notebook")',
                metadata={"tags": ["raises-exception"]},
            )
        ]
    )
    nbformat.write(notebook, source / "broken.ipynb")
    output = tmp_path / "rendered"
    result = subprocess.run(
        [sys.executable, str(COMMAND), "--notebook-dir", str(source), "--output-dir", str(output)],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode != 0
    assert "cannot publish a failed notebook" in result.stderr
    assert not (output / "examples" / "broken.md").exists()
