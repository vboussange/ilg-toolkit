"""Reproducible frozen-protocol timing gates; inputs are independent fit blocks."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from fixtures import fixture, protocol_digest
from fixtures import protocol as workload_protocol


def blocks(values):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim == 1:
        result = result[:, None]
    if (
        result.ndim != 2
        or not result.size
        or len(result) < 3
        or not np.isfinite(result).all()
        or (result <= 0).any()
    ):
        raise ValueError("Timing samples require at least three positive finite fit blocks")
    return result


def sampled_medians(values, rng):
    values = blocks(values)
    indices = rng.integers(0, len(values), size=(10000, len(values)))
    return np.median(values[indices].reshape(10000, -1), axis=1)


def decision(estimate, samples, limit, repetitions) -> dict:
    low, high = np.quantile(samples, [0.025, 0.975])
    status = "pass" if high <= limit else "fail" if low > limit else "inconclusive"
    return dict(
        ratio=float(estimate),
        bootstrap_95_percent=[float(low), float(high)],
        limit=limit,
        status=status,
        fit_repetitions=repetitions,
        minimum_repetitions_required=10 if status == "inconclusive" else None,
    )


def ratio_gate(numerator, denominator, *, limit):
    """Independently resample fits, retaining each fit's step dependence."""
    first, second = blocks(numerator), blocks(denominator)
    rng = np.random.default_rng(52053)
    draws = sampled_medians(first, rng) / sampled_medians(second, rng)
    return decision(np.median(first) / np.median(second), draws, limit, [len(first), len(second)])


def marginal_gate(before, after, independent):
    """Compare one adjacent joint increment with the added species' PR fit."""
    first, second, alone = blocks(before), blocks(after), blocks(independent)
    rng = np.random.default_rng(52053)
    draws = (sampled_medians(second, rng) - sampled_medians(first, rng)) / sampled_medians(
        alone, rng
    )
    estimate = (np.median(second) - np.median(first)) / np.median(alone)
    return decision(estimate, draws, 0.5, [len(first), len(second), len(alone)])


def _indexed(reports, native):
    output = {}
    for report in reports:
        if report.get("smoke") or report.get("failures"):
            raise ValueError("Smoke or failed worker report cannot pass acceptance")
        for result in report["results"]:
            name = result["experiment"]["id"] if native else result["case"]["name"]
            if name in output:
                previous = output[name]
                repeat = (
                    not native
                    and len(previous["warm_fit"]["samples_seconds"]) < 10
                    and len(result["warm_fit"]["samples_seconds"]) >= 10
                    and previous["case"] == result["case"]
                    and previous["input_sha256"] == result["input_sha256"]
                    and all(
                        previous["machine"][field] == result["machine"][field]
                        for field in (
                            "source_tree",
                            "software",
                            "fixtures_sha256",
                            "protocol_sha256",
                        )
                    )
                )
                if not repeat:
                    raise ValueError(
                        f"Duplicate result {name}; select a documented measurement batch"
                    )
            output[name] = result
    return output


def _samples(result):
    fits = blocks(result["warm_fit"]["samples_seconds"])
    steps = blocks(result["warm_training_steps_by_fit_seconds"])
    if len(fits) < 3 or len(steps) != len(fits) or steps.shape[1] != 3:
        raise ValueError("Require at least three complete fits, each with all three updates")
    if result["scientific_eligibility"] != "passed":
        raise ValueError("Scientific eligibility failed")
    return fits, steps[:, 1:]


def timing_summary(result):
    """Describe independent fit blocks without inventing a baseline comparison."""
    fits, steps = _samples(result)
    fit_bounds = np.quantile(sampled_medians(fits, np.random.default_rng(52053)), [0.025, 0.975])
    return dict(
        case=result["experiment"]["id"],
        fit_median_seconds=float(np.median(fits)),
        fit_bootstrap_95_percent_seconds=fit_bounds.tolist(),
        steady_step_median_seconds=float(np.median(steps)),
        recurrent_first_update_median_seconds=float(
            np.median(np.asarray(result["warm_training_steps_by_fit_seconds"])[:, 0])
        ),
        compile_plus_first_fit_seconds=result["compile_plus_first_fit_seconds"],
        first_predict_seconds=result["first_predict_seconds"],
        warm_predict=result["warm_predict"],
        process_peak_rss_after_timed_work_mib=result["process_peak_rss_after_timed_work_mib"],
    )


def compare_reports(main_reports, pr_reports) -> dict:
    """Validate complete work manifests before evaluating every frozen gate."""
    protocol = workload_protocol()
    main, pr = _indexed(main_reports, False), _indexed(pr_reports, True)
    repeated_main_cases = sorted(
        name
        for name in main
        if sum(
            any(row["case"]["name"] == name for row in report["results"]) for report in main_reports
        )
        > 1
    )
    required_main = {case["name"] for case in protocol["cases"]}
    required_pr = {name + "-single" for name in required_main}
    for name in protocol["gated_representative_cases"]:
        required_pr.update(f"{name}-joint-{count}" for count in (1, 2, 3, 4))
        required_pr.update(f"{name}-independent-{index}" for index in range(4))
    required_pr.update(
        f"loglinear-{case}-joint-{count}"
        for case in ("feature-heavy", "solver-stress")
        for count in (1, 4)
    )
    if required_main - main.keys() or required_pr - pr.keys():
        raise ValueError(
            f"Missing workloads: main={sorted(required_main - main.keys())}, "
            f"PR={sorted(required_pr - pr.keys())}"
        )
    for case in protocol["cases"]:
        result = main[case["name"]]
        machine = result["machine"]
        expected_inputs = {item["name"]: item["sha256"] for item in fixture(case, protocol)}
        if (
            result["case"] != case
            or result["input_sha256"] != expected_inputs
            or machine["source_tree"] != "8487c17e3d34a683464b3093539a74c8fd276238"
            or machine["frozen_main_commit"] != protocol["frozen_main_commit"]
            or machine["protocol_sha256"] != protocol_digest()
            or machine["fixtures_sha256"]
            != hashlib.sha256(Path(__file__).with_name("fixtures.py").read_bytes()).hexdigest()
            or machine["cpu_affinity"] != protocol["execution"]["cpu_affinity"]
        ):
            raise ValueError(f"Baseline source, workload or execution mismatch: {case['name']}")
    for report in main_reports + pr_reports:
        settings = dict(report["protocol"])
        repeats = settings.pop("fit_repeats")
        frozen = dict(protocol)
        frozen.pop("fit_repeats")
        if settings != frozen or repeats < 3:
            raise ValueError("Effective protocol differs from the frozen scientific workload")
        if any(len(value["warm_fit"]["samples_seconds"]) != repeats for value in report["results"]):
            raise ValueError("Recorded repeat count does not match complete observed fits")
    expected_cases = {}
    for case in protocol["cases"]:
        name = case["name"]
        expected_cases[name + "-single"] = (case, [0])
        if name in protocol["gated_representative_cases"]:
            for count in (1, 2, 3, 4):
                expected_cases[f"{name}-joint-{count}"] = (case, list(range(count)))
            for index in range(4):
                expected_cases[f"{name}-independent-{index}"] = (case, [index])
        if name in {"resnet-feature-heavy", "resnet-solver-stress"}:
            fixed = dict(case, name=name.replace("resnet", "loglinear"), family="loglinear")
            for count in (1, 4):
                expected_cases[f"{fixed['name']}-joint-{count}"] = (fixed, list(range(count)))
    sources, drivers, chunks = set(), set(), set()
    protocol_hash = protocol_digest()
    for key, result in pr.items():
        _samples(result)
        machine = result["machine"]
        if machine.get("smoke") or machine.get("dirty_source_or_driver"):
            raise ValueError(f"Uncommitted or smoke source: {key}")
        if machine["protocol_sha256"] != protocol_hash or machine["cpu_affinity"] != [0, 1]:
            raise ValueError(f"Protocol/affinity mismatch: {key}")
        sources.add(machine["source_tree"])
        drivers.add(machine["native_driver_sha256"])
        chunks.add(result["accounting"]["species_chunk_size"])
        count = len(result["experiment"]["species_indices"])
        case = result["case"]
        if (case, result["experiment"]["species_indices"]) != expected_cases[key]:
            raise ValueError(f"Frozen workload/count mismatch: {key}")
        groups = case["regions"] * count
        expected = dict(
            species=count,
            optimizer_updates=3,
            group_update_exposure=3,
            regions=case["regions"],
            pairs_per_group=case["units"] * (case["units"] - 1) // 2,
            full_evaluations=4,
            finalized_group_calibrations=4 * groups if case["objective"] == "mlpe" else 0,
            observed=dict(
                encodings=7 * case["regions"],
                group_contributions=3 * groups,
                evaluated_groups=4 * groups,
            ),
            region_batch_size=None,
            species_batch_size=None,
        )
        if any(result["accounting"].get(field) != value for field, value in expected.items()):
            raise ValueError(f"Scientific work mismatch: {key}")
        if len(result["checks"]) != groups or any(
            check["pairs"] != expected["pairs_per_group"] for check in result["checks"]
        ):
            raise ValueError(f"Incomplete scientific validation: {key}")
    if len(sources) != 1 or len(drivers) != 1 or len(chunks) != 1:
        raise ValueError(
            "Use one production source, native driver and declared chunk per comparison batch"
        )
    single, joint, disclosures = [], [], []
    for case in protocol["cases"]:
        name = case["name"]
        baseline, native = main[name], pr[name + "-single"]
        main_fit, main_steps = _samples(baseline)
        pr_fit, pr_steps = _samples(native)
        if name in repeated_main_cases and len(pr_fit) < 10:
            raise ValueError("A repeated baseline requires a fresh PR batch with at least ten fits")
        if (
            native["case"] != case
            or native["input_sha256"]["species-0"] != baseline["input_sha256"]
        ):
            raise ValueError(f"Frozen input mismatch: {name}")
        if native["machine"]["software"] != baseline["machine"]["software"]:
            raise ValueError(f"Software mismatch: {name}")
        fitted = ratio_gate(pr_fit, main_fit, limit=1.1)
        steady = ratio_gate(pr_steps, main_steps, limit=1.1)
        single.append(
            dict(
                case=name,
                gated=name in protocol["gated_representative_cases"],
                complete_fit=fitted,
                steady_training_step=steady,
                median_seconds=dict(
                    main_fit=float(np.median(main_fit)),
                    pr_fit=float(np.median(pr_fit)),
                    main_steady_step=float(np.median(main_steps)),
                    pr_steady_step=float(np.median(pr_steps)),
                    main_recurrent_first_update=float(
                        np.median(np.asarray(baseline["warm_training_steps_by_fit_seconds"])[:, 0])
                    ),
                    pr_recurrent_first_update=float(
                        np.median(np.asarray(native["warm_training_steps_by_fit_seconds"])[:, 0])
                    ),
                ),
            )
        )
        for field, threshold in (
            ("compile_plus_first_fit_seconds", 2.0),
            ("process_peak_rss_after_timed_work_mib", 1.5),
        ):
            ratio = native[field] / baseline[field]
            disclosures.append(
                dict(case=name, metric=field, ratio=ratio, material_regression=ratio > threshold)
            )
        prediction_ratio = (
            native["warm_predict"]["median_seconds"] / baseline["warm_predict"]["median_seconds"]
        )
        disclosures.append(
            dict(
                case=name,
                metric="warm_predict",
                ratio=prediction_ratio,
                material_regression=prediction_ratio > 1.5,
            )
        )
    for name in protocol["gated_representative_cases"]:
        prefix = [pr[f"{name}-joint-{count}"] for count in (1, 2, 3, 4)]
        independent = [pr[f"{name}-independent-{index}"] for index in range(4)]
        canonical = prefix[-1]["initialization"]
        for result in prefix + independent:
            initialization = result["initialization"]
            if (
                not initialization.get("matched_exactly")
                or initialization["backbone"] != canonical["backbone"]
            ):
                raise ValueError(
                    f"Joint/independent backbone mismatch: {result['experiment']['id']}"
                )
            for species, row in initialization["projection_rows"].items():
                if row != canonical["projection_rows"][species]:
                    raise ValueError("Joint/independent projection row mismatch")
                if result["input_sha256"][species] != prefix[-1]["input_sha256"][species]:
                    raise ValueError("Joint/independent species input mismatch")
        increments = [
            dict(
                before=count,
                after=count + 1,
                **marginal_gate(
                    _samples(prefix[count - 1])[0],
                    _samples(prefix[count])[0],
                    _samples(independent[count])[0],
                ),
            )
            for count in (1, 2, 3)
        ]
        totals = [
            dict(
                species=count,
                joint_over_sum_independent=float(
                    np.median(_samples(prefix[count - 1])[0])
                    / sum(np.median(_samples(value)[0]) for value in independent[:count])
                ),
            )
            for count in (1, 2, 3, 4)
        ]
        joint.append(dict(case=name, adjacent_increments=increments, total_comparisons=totals))
    required = [
        value[metric]["status"]
        for value in single
        if value["gated"]
        for metric in ("complete_fit", "steady_training_step")
    ]
    required.extend(
        increment["status"] for value in joint for increment in value["adjacent_increments"]
    )
    status = (
        "fail" if "fail" in required else "inconclusive" if "inconclusive" in required else "pass"
    )
    return dict(
        status=status,
        repeated_main_cases=repeated_main_cases,
        production_source_tree=next(iter(sources)),
        single_species=single,
        joint_species=joint,
        disclosures=disclosures,
        loglinear=[
            timing_summary(result) for name, result in pr.items() if name.startswith("loglinear-")
        ],
        measured_commits=sorted({value["machine"]["measured_commit"] for value in pr.values()}),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", type=Path, action="append", required=True)
    parser.add_argument("--pr", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in args.main + args.pr
    }
    result: dict
    try:
        result = compare_reports(
            [json.loads(path.read_text()) for path in args.main],
            [json.loads(path.read_text()) for path in args.pr],
        )
    except (KeyError, ValueError) as error:
        result = dict(status="invalid", reason=str(error))
    result.update(
        raw_inputs=evidence,
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        protocol_sha256=protocol_digest(),
    )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result))
    raise SystemExit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
