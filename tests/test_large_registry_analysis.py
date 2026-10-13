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


def test_absent_encoder_and_optimizer_work_cannot_pass_the_audit():
    from benchmarks.spec52.large_registry_analysis import analyze_case

    events = []
    for kind, step, indices in (
        ("tail_evaluation", 0, range(4)),
        ("tail_forward_backward", 1, (3, 2, 1, 0)),
        ("tail_evaluation", 1, range(4)),
    ):
        for index in indices:
            events.append(
                {
                    "kind": kind,
                    "step": step,
                    "species_indices": [index],
                    "payload_signature": "one-eight-unit-group",
                    "started_ns": len(events) * 2,
                    "finished_ns": len(events) * 2 + 1,
                    "new_memory_reports": (
                        [kind + ".txt"]
                        if not any(event["kind"] == kind for event in events)
                        else []
                    ),
                }
            )
    raw = {
        "schema": "spec52-large-registry-v1",
        "family": "unet",
        "registry_size": 4,
        "species_chunk_size": 1,
        "ragged": False,
        "events": events,
        "history_steps": [0, 1],
        "calibrated_groups": 4,
        "selected_prediction_checks": {"passed": True},
        "cpu_compiled_allocation_reports": [],
        "persistent_bytes": {},
        "process_peak_rss_bytes_including_compiler": 0,
        "source_commit": "synthetic",
    }
    import pytest

    with pytest.raises(ValueError, match="encoder|optimizer"):
        analyze_case(raw)


def test_preserved_real_reports_verify_totals_reuse_and_selected_work():
    from benchmarks.spec52.large_registry_analysis import analyze_case

    raw_path = ANALYZER.parent / "results/large-registry/preliminary-d7e410b/unet-s4-c1.json"
    summary = analyze_case(json.loads(raw_path.read_text()))
    assert sorted(summary["selected_species_indices"]) == [0, 1, 2, 3]
    assert [(item["calls"], item["compiled_executables"]) for item in summary["compilation"]] == [
        (8, 1),
        (4, 1),
    ]
    assert {item["role"]: item["compiled_allocated_bytes"] for item in summary["allocations"]} == {
        "encoder_inference": 147336,
        "encoder_training_forward": 1845256,
        "encoder_backward": 2006571,
        "tail_evaluation": 25180,
        "tail_forward_backward": 52477,
    }


def test_changed_report_total_or_extra_numeric_id_compilation_is_rejected():
    import copy

    import pytest

    from benchmarks.spec52.large_registry_analysis import analyze_case

    raw_path = ANALYZER.parent / "results/large-registry/preliminary-d7e410b/unet-s4-c1.json"
    original = json.loads(raw_path.read_text())
    corrupted = copy.deepcopy(original)
    corrupted["cpu_compiled_allocation_reports"][0]["total_allocated_bytes"] += 1
    with pytest.raises(ValueError, match="inconsistent.*allocation"):
        analyze_case(corrupted)
    extra_compile = copy.deepcopy(original)
    repeated = [event for event in extra_compile["events"] if event["kind"] == "tail_evaluation"][1]
    repeated["new_memory_reports"] = ["additional_numeric_id_specialization.txt"]
    with pytest.raises(ValueError, match="one compile per compatible"):
        analyze_case(extra_compile)


def test_changed_payload_shapes_cannot_keep_the_old_reuse_signature():
    import pytest

    from benchmarks.spec52.large_registry_analysis import analyze_case

    raw_path = ANALYZER.parent / "results/large-registry/preliminary-d7e410b/unet-s4-c1.json"
    raw = json.loads(raw_path.read_text())
    tail = next(event for event in raw["events"] if event["kind"] == "tail_evaluation")
    tail["payload_shapes"][0][0][0] = 1000
    with pytest.raises(ValueError, match="payload shape.*signature"):
        analyze_case(raw)


def test_compact_shape_catalogue_preserves_the_same_audited_evidence():
    from benchmarks.spec52.large_registry_analysis import analyze_case

    raw_path = ANALYZER.parent / "results/large-registry/preliminary-d7e410b/unet-s4-c1.json"
    raw = json.loads(raw_path.read_text())
    expected = analyze_case(raw)
    raw["payload_shapes_by_signature"] = {}
    for event in raw["events"]:
        if "payload_shapes" in event:
            raw["payload_shapes_by_signature"][event["payload_signature"]] = event.pop(
                "payload_shapes"
            )
    assert analyze_case(raw) == expected
