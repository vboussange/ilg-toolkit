"""Derive allocation totals from preserved public-fit XLA text reports.

The first probe revision searched for 'Total bytes used:' whereas installed XLA
writes 'Total bytes:'. Keep its raw JSON and measured source identities intact;
this separately hashed analysis parses the embedded evidence without rerunning
or relabelling numerical measurements.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = []
    for path in sorted(args.directory.glob("*-s*-c*.json")):
        data = json.loads(path.read_text())
        allocations = []
        encoder_index = 0
        for report in data["cpu_compiled_allocation_reports"]:
            matched = re.search(r"Total bytes(?: used)?:\s*(\d+)", report["report"])
            if matched is None:
                raise ValueError(f"Missing allocation total in {path}: {report['file']}")
            if "jit__encode" in report["file"]:
                role = (
                    "encoder_inference",
                    "encoder_training_forward_and_residual_outputs",
                    "encoder_backward",
                )[encoder_index]
                encoder_index += 1
            else:
                role = (
                    "tail_forward_backward"
                    if "chunk_gradient" in report["file"]
                    else "tail_evaluation_and_posterior_output"
                )
            allocations.append(
                {
                    "role": role,
                    "compiled_allocated_bytes": int(matched.group(1)),
                    "report_file": report["file"],
                }
            )
        if encoder_index != 3:
            raise ValueError(
                f"Expected inference, training forward and backward encoder compilations: {path}"
            )
        cases.append(
            {
                "source": path.name,
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "family": data["family"],
                "registry_size": data["registry_size"],
                "species_chunk_size": data["species_chunk_size"],
                "measured_commit": data["source_commit"],
                "allocations": allocations,
                "persistent_bytes": data["persistent_bytes"],
                "counts": data["counts"],
                "process_peak_rss_bytes_including_compiler": data[
                    "process_peak_rss_bytes_including_compiler"
                ],
            }
        )
    if not cases:
        raise ValueError("No raw bounded-probe files found")
    result = {
        "schema": "spec52-bounded-analysis-v1",
        "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": (
            "CPU compiler buffer allocations include arguments, outputs and temporaries. "
            "Encoder residuals remain live during tail calls; neither the maximum nor sum "
            "of these per-executable totals is an observed end-to-end peak. Process RSS "
            "includes compiler caches and persistent inputs/outputs; no GPU peak was measured."
        ),
        "cases": cases,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
