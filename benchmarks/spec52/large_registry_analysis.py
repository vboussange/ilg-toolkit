"""Audit large-registry public-fit evidence without importing JAX.

This checks coverage, completed chunk bounds, compile reuse, and recorded CPU
compiler totals. It does not convert CPU RSS/allocation estimates into GPU peaks.
"""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def analyze_case(data):
    if data.get("schema") != "spec52-large-registry-v1":
        raise ValueError("Unknown large-registry evidence schema")
    size, chunk = data["registry_size"], data["species_chunk_size"]
    events = data["events"]
    for step, phase in ((0, "initial"), (1, "final")):
        indices = [
            index
            for event in events
            if event["kind"] == "tail_evaluation" and event["step"] == step
            for index in event["species_indices"]
        ]
        if sorted(indices) != list(range(size)):
            raise ValueError(f"Incomplete {phase} evaluation coverage")
    counts = Counter(event["kind"] for event in events)
    for role, expected in (
        ("encoder_inference", 2),
        ("encoder_training_forward", 1),
        ("encoder_backward", 1),
        ("optimizer_update", 1),
    ):
        if counts[role] != expected:
            raise ValueError(f"Incomplete encoder/optimizer work: {role}={counts[role]}")
    selected = [
        index
        for event in events
        if event["kind"] == "tail_forward_backward"
        for index in event["species_indices"]
    ]
    if len(selected) != 4 or len(set(selected)) != 4:
        raise ValueError("Training must select exactly four distinct species")
    if any(len(event.get("species_indices", ())) > chunk for event in events):
        raise ValueError("Observed species tail exceeds the configured chunk bound")
    if any(
        left["finished_ns"] > right["started_ns"]
        for left, right in zip(events, events[1:], strict=False)
    ):
        raise ValueError("Checked numerical calls overlap rather than completing sequentially")
    if data["history_steps"] != [0, 1] or data["calibrated_groups"] != size:
        raise ValueError("Missing initial/final evaluations or complete calibration output")
    if not data["selected_prediction_checks"]["passed"]:
        raise ValueError("Selected public prediction checks did not pass")
    compile_groups = defaultdict(lambda: {"calls": 0, "species_ids": set(), "reports": []})
    for event in events:
        if event["kind"] not in {"tail_evaluation", "tail_forward_backward"}:
            continue
        signature = event["payload_signature"]
        shapes = (
            event["payload_shapes"]
            if "payload_shapes" in event
            else data["payload_shapes_by_signature"][signature]
        )
        if hashlib.sha256(json.dumps(shapes).encode()).hexdigest() != signature:
            raise ValueError("Recorded payload shape does not match its signature")
        if any(shape[0] != len(event["species_indices"]) for shape, _ in shapes):
            raise ValueError("Recorded payload shape exceeds its declared species chunk")
        key = (event["kind"], signature)
        group = compile_groups[key]
        group["calls"] += 1
        group["species_ids"].update(event["species_indices"])
        group["reports"].extend(event["new_memory_reports"])
    compilation = []
    for (kind, signature), group in sorted(compile_groups.items()):
        if len(group["reports"]) != 1:
            raise ValueError(f"Expected one compile per compatible {kind} signature: {group}")
        compilation.append(
            {
                "role": kind,
                "payload_signature": signature,
                "calls": group["calls"],
                "distinct_numeric_species_ids": len(group["species_ids"]),
                "compiled_executables": len(group["reports"]),
            }
        )
    expected_shapes = 2 if data["ragged"] else 1
    for role in ("tail_evaluation", "tail_forward_backward"):
        if sum(item["role"] == role for item in compilation) != expected_shapes:
            raise ValueError(f"Unexpected {role} shape specialization count")
    allocations = []
    for report in data["cpu_compiled_allocation_reports"]:
        matched = re.search(r"Total bytes(?: used)?:\s*(\d+)", report["report"])
        if matched is None or int(matched[1]) != report["total_allocated_bytes"]:
            raise ValueError("Missing or inconsistent embedded XLA allocation total")
        allocations.append(
            {
                "role": report["role"],
                "compiled_allocated_bytes": int(matched[1]),
                "file": report["file"],
                "entry_layout": report["entry_layout"],
            }
        )
    return {
        "family": data["family"],
        "registry_size": size,
        "species_chunk_size": chunk,
        "ragged": data["ragged"],
        "measured_commit": data["source_commit"],
        "selected_species_indices": selected,
        "compilation": compilation,
        "allocations": allocations,
        "persistent_bytes": data["persistent_bytes"],
        "process_peak_rss_bytes_including_compiler": data[
            "process_peak_rss_bytes_including_compiler"
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    cases = []
    source_trees, driver_hashes = set(), set()
    for path in sorted(args.directory.glob("*-s*-c*.json")):
        raw = json.loads(path.read_text())
        case = analyze_case(raw)
        source_trees.add(raw["source_tree"])
        driver_hashes.add(raw["source_sha256"]["benchmarks/spec52/large_registry_probe.py"])
        case.update(source=path.name, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        cases.append(case)
    if not cases:
        raise ValueError("No large-registry evidence")
    if len(source_trees) != 1 or len(driver_hashes) != 1:
        raise ValueError("A matrix requires one production source tree and one measured driver")
    required = {
        (family, size, chunk, False)
        for family in ("unet", "resnet")
        for size in (4, 100, 1000)
        for chunk in (1, 4)
    } | {(family, 4, 1, True) for family in ("unet", "resnet")}
    actual = {
        (case["family"], case["registry_size"], case["species_chunk_size"], case["ragged"])
        for case in cases
    }
    if len(actual) != len(cases):
        raise ValueError("A matrix must not duplicate the same family/registry/chunk/shape case")
    missing = sorted(required - actual)
    if missing and not args.allow_partial:
        raise ValueError(f"Missing required large-registry cases: {missing}")
    output = {
        "schema": "spec52-large-registry-analysis-v1",
        "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "production_source_tree": next(iter(source_trees)),
        "measured_driver_sha256": next(iter(driver_hashes)),
        "complete_matrix": not missing,
        "missing_cases": missing,
        "scope": (
            "Per-executable CPU compiled allocations are not a complete-fit device peak. "
            "Encoder residuals remain live while bounded tails run. Live-array snapshots "
            "omit internal executable transients. RSS includes host/compiler/output storage."
        ),
        "cases": cases,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
