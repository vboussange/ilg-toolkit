"""Check that the reproducible evidence command refuses incomplete reports."""

import json
import subprocess
import sys
from pathlib import Path

ANALYZER = Path(__file__).resolve().parents[1] / "benchmarks/spec52/large_registry_analysis.py"


def test_empty_evidence_directory_cannot_report_success(tmp_path):
    output = tmp_path / "analysis.json"
    result = subprocess.run(
        [sys.executable, str(ANALYZER), str(tmp_path), "--output", str(output)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "No large-registry evidence" in result.stderr
    assert not output.exists()


def test_reports_require_full_evaluation_coverage(tmp_path):
    raw = {
        "schema": "spec52-large-registry-v1",
        "family": "unet",
        "registry_size": 4,
        "species_chunk_size": 1,
        "species_batch_size": 4,
        "ragged": False,
        "events": [],
    }
    (tmp_path / "unet-s4-c1.json").write_text(json.dumps(raw))
    result = subprocess.run(
        [sys.executable, str(ANALYZER), str(tmp_path), "--output", str(tmp_path / "a.json")],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "initial evaluation coverage" in result.stderr
