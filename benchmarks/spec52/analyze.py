"""Derive compilation-aware summaries without overwriting frozen observations."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from fixtures import summary

POLICY_PATH = Path(__file__).with_name("analysis_policy.json")


def steady_steps(blocks, policy):
    """Bootstrap complete fit blocks to retain within-fit timing dependence."""
    if not blocks or any(len(block) < 2 for block in blocks):
        raise ValueError("At least two training steps per fit are required")
    values = np.asarray([block[1:] for block in blocks], dtype=np.float64)
    confidence = policy["confidence"]
    rng = np.random.default_rng(confidence["seed"])
    indices = rng.integers(0, len(values), size=(confidence["bootstrap_resamples"], len(values)))
    sampled = values[indices].reshape(len(indices), -1)
    medians = np.median(sampled, axis=1)
    output = summary(values.flatten())
    output["median_bootstrap_95_percent_seconds"] = np.quantile(medians, [0.025, 0.975]).tolist()
    output["bootstrap_unit"] = "fit block, retaining all post-first updates together"
    output["samples_by_fit_seconds"] = values.tolist()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.input.read_text())
    policy = json.loads(POLICY_PATH.read_text())
    results = []
    for result in raw["results"]:
        blocks = result["warm_training_steps_by_fit_seconds"]
        results.append(
            dict(
                case=result["case"]["name"],
                complete_fit=result["warm_fit"],
                steady_training_step=steady_steps(blocks, policy),
                recurrent_first_update=summary([block[0] for block in blocks]),
                predict=result["warm_predict"],
                compile_plus_first_fit_seconds=result["compile_plus_first_fit_seconds"],
                process_peak_rss_after_timed_work_mib=result[
                    "process_peak_rss_after_timed_work_mib"
                ],
            )
        )
    output = dict(
        raw_input=str(args.input),
        raw_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        analysis_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        analysis_policy_sha256=hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest(),
        policy=policy,
        results=results,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
