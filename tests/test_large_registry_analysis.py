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


def test_one_matrix_cannot_mix_different_production_source_trees(tmp_path):
    samples = ANALYZER.parent / "results/large-registry/preliminary-d7e410b"
    for index, name in enumerate(("unet-s4-c1.json", "unet-s100-c1.json")):
        raw = json.loads((samples / name).read_text())
        if index:
            raw["source_tree"] = "different-production-tree"
        (tmp_path / name).write_text(json.dumps(raw))
    result = subprocess.run(
        [
            sys.executable,
            str(ANALYZER),
            str(tmp_path),
            "--allow-partial",
            "--output",
            str(tmp_path / "analysis.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "one production source tree" in result.stderr


def two_update_report():
    """A second completed update uses different IDs with the same payload shapes."""
    import copy
    from collections import Counter

    raw_path = ANALYZER.parent / "results/large-registry/preliminary-d7e410b/unet-s100-c4.json"
    raw = json.loads(raw_path.read_text())
    raw["source_commit"] = "synthetic-two-update-execution"
    raw["schema"] = "spec52-large-registry-v2"
    raw["training_config"].update(steps=2, evaluation_interval=2)
    raw["history_steps"] = [0, 2]
    repeated = copy.deepcopy(
        [
            event
            for event in raw["events"]
            if event["kind"]
            in {
                "encoder_training_forward",
                "tail_forward_backward",
                "encoder_backward",
                "optimizer_update",
            }
        ]
    )
    for event in repeated:
        event.update(step=2, new_memory_reports=[])
        if event["kind"] == "tail_forward_backward":
            event.update(species_indices=[0, 1, 2, 3], group_indices=[0, 1, 2, 3])
    position = next(i for i, e in enumerate(raw["events"]) if e["kind"] == "optimizer_update") + 1
    for event in raw["events"][position:]:
        event["step"] = 2
    raw["events"][position:position] = repeated
    for index, event in enumerate(raw["events"]):
        event.update(started_ns=index * 2, finished_ns=index * 2 + 1)
    raw["counts"] = dict(Counter(e["kind"] for e in raw["events"]))
    return raw


def test_repeated_updates_report_numeric_id_reuse_and_complete_final_evaluation(tmp_path):
    raw = two_update_report()
    (tmp_path / "unet-s100-c4.json").write_text(json.dumps(raw))
    output = tmp_path / "analysis.json"
    result = subprocess.run(
        [sys.executable, str(ANALYZER), str(tmp_path), "--allow-partial", "--output", str(output)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(output.read_text())
    assert not summary["complete_matrix"]
    case = summary["cases"][0]
    assert case["training_steps"] == 2
    gradient = next(item for item in case["compilation"] if item["role"] == "tail_forward_backward")
    assert gradient["calls"] == 2
    assert gradient["compiled_executables"] == 1
    assert gradient["distinct_numeric_species_ids"] > 4
    assert "not a complete-fit device peak" in summary["scope"]


def test_repeating_only_the_same_large_registry_selection_is_insufficient_reuse_evidence(tmp_path):
    raw = two_update_report()
    first, second = [e for e in raw["events"] if e["kind"] == "tail_forward_backward"]
    second["species_indices"] = first["species_indices"]
    second["group_indices"] = first["group_indices"]
    (tmp_path / "unet-s100-c4.json").write_text(json.dumps(raw))
    result = subprocess.run(
        [
            sys.executable,
            str(ANALYZER),
            str(tmp_path),
            "--allow-partial",
            "--output",
            str(tmp_path / "analysis.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "different numeric species" in result.stderr


def test_checked_work_without_its_allocation_evidence_cannot_pass(tmp_path):
    raw = two_update_report()
    raw["cpu_compiled_allocation_reports"] = [
        report
        for report in raw["cpu_compiled_allocation_reports"]
        if report["role"] != "encoder_training_forward"
    ]
    (tmp_path / "unet-s100-c4.json").write_text(json.dumps(raw))
    result = subprocess.run(
        [
            sys.executable,
            str(ANALYZER),
            str(tmp_path),
            "--allow-partial",
            "--output",
            str(tmp_path / "analysis.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Missing or mislabelled allocation evidence" in result.stderr


def test_omitting_all_encoder_allocation_evidence_cannot_pass(tmp_path):
    raw = two_update_report()
    raw["cpu_compiled_allocation_reports"] = [
        report
        for report in raw["cpu_compiled_allocation_reports"]
        if report["role"] != "encoder_training_forward"
    ]
    for event in raw["events"]:
        if event["kind"] == "encoder_training_forward":
            event["new_memory_reports"] = []
    (tmp_path / "unet-s100-c4.json").write_text(json.dumps(raw))
    result = subprocess.run(
        [
            sys.executable,
            str(ANALYZER),
            str(tmp_path),
            "--allow-partial",
            "--output",
            str(tmp_path / "analysis.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Missing encoder allocation evidence" in result.stderr
