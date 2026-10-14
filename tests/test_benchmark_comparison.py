"""Acceptance analysis examples independent of measured performance."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.spec52.compare import compare_reports as analyze_reports
from benchmarks.spec52.compare import marginal_gate, ratio_gate


def baseline_report():
    """Authored identity and work examples; none of these values are measurements."""
    directory = Path(__file__).parents[1] / "benchmarks/spec52"
    protocol_bytes = (directory / "protocol.json").read_bytes()
    protocol = json.loads(protocol_bytes)
    protocol["fit_repeats"] = 10
    rows = []
    for case in protocol["cases"]:
        pairs = case["units"] * (case["units"] - 1) // 2
        rows.append(
            dict(
                case=case,
                input_sha256=f"synthetic-input-{case['name']}",
                machine=dict(
                    source_tree="synthetic-main",
                    frozen_main_commit="synthetic-main-commit",
                    protocol_sha256=hashlib.sha256(protocol_bytes).hexdigest(),
                    fixtures_sha256="synthetic-fixtures",
                    cpu_affinity=[0, 1],
                    software={"synthetic": "test"},
                ),
                scientific_eligibility="passed",
                warm_fit={"samples_seconds": [10.0] * 10},
                warm_training_steps_by_fit_seconds=[[10.0] * 3 for _ in range(10)],
                compile_plus_first_fit_seconds=50.0,
                first_predict_seconds=2.0,
                warm_predict={"median_seconds": 1.0},
                process_peak_rss_after_timed_work_mib=128.0,
                checks=[{"pairs": pairs} for _ in range(case["regions"])],
                accounting=dict(
                    optimizer_updates=3,
                    group_update_exposure=3,
                    regions=case["regions"],
                    pairs_per_group=pairs,
                    full_evaluations=4,
                ),
            )
        )
    return dict(protocol=protocol, results=rows, failures=[], smoke=False)


def compare_reports(main_reports, pr_reports):
    # A fresh reference must not inherit corruptions applied to the test inputs.
    return analyze_reports(main_reports, pr_reports, frozen_main_report=baseline_report())


def test_ratio_gate_separates_pass_failure_and_unresolved_noise():
    assert ratio_gate([1.0] * 10, [1.0] * 10, limit=1.1)["status"] == "pass"
    assert ratio_gate([1.2] * 10, [1.0] * 10, limit=1.1)["status"] == "fail"
    ambiguous = ratio_gate([0.8, 1.2, 1.4], [1.0] * 3, limit=1.1)
    assert ambiguous["status"] == "inconclusive"
    assert ambiguous["minimum_repetitions_required"] == 10


@pytest.mark.parametrize("samples", [[], [float("nan")], [0.0], [-1.0], [1.0], [1.0, 1.0]])
def test_invalid_samples_cannot_yield_a_passing_gate(samples):
    with pytest.raises(ValueError):
        ratio_gate(samples, [1.0] * 3, limit=1.1)


def test_each_adjacent_increment_is_measured_and_signed_noise_is_retained():
    first = marginal_gate([10.0] * 10, [12.0] * 10, [10.0] * 10)
    second = marginal_gate([12.0] * 10, [18.0] * 10, [10.0] * 10)
    negative = marginal_gate([10.0] * 10, [9.0] * 10, [10.0] * 10)
    assert first["ratio"] == pytest.approx(0.2)
    assert first["status"] == "pass"
    assert second["ratio"] == pytest.approx(0.6)
    assert second["status"] == "fail"
    assert negative["ratio"] == pytest.approx(-0.1)


def test_unpaired_fits_do_not_invent_exact_correlation_between_revisions():
    result = ratio_gate([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], [[1.0], [2.0], [3.0]], limit=1.1)
    low, high = result["bootstrap_95_percent"]
    assert low < 1.0 < high
    assert result["status"] == "inconclusive"


def test_missing_workloads_cannot_become_a_pass():
    with pytest.raises(ValueError, match="Missing"):
        compare_reports([{"results": []}], [{"results": [], "failures": []}])


def reports():
    """Literal 1.0 parity and 0.2 marginal examples with synthetic work manifests."""
    baseline = baseline_report()
    native = dict(protocol=baseline["protocol"], results=[], failures=[], smoke=False)
    for original in baseline["results"]:
        original["warm_fit"]["samples_seconds"] = [10.0] * 10
        original["warm_training_steps_by_fit_seconds"] = [[10.0] * 3] * 10
        variants = [("single", [0])]
        if original["case"]["name"] in baseline["protocol"]["gated_representative_cases"]:
            variants += [("joint", list(range(count))) for count in (1, 2, 3, 4)]
            variants += [("independent", [index]) for index in range(4)]
        if original["case"]["name"] in {"resnet-feature-heavy", "resnet-solver-stress"}:
            variants += [("loglinear", list(range(count))) for count in (1, 4)]
        for kind, indices in variants:
            row = copy.deepcopy(original)
            if kind == "loglinear":
                row["case"]["name"] = row["case"]["name"].replace("resnet", "loglinear")
                row["case"]["family"] = "loglinear"
            suffix = (
                "single"
                if kind == "single"
                else f"independent-{indices[0]}"
                if kind == "independent"
                else f"joint-{len(indices)}"
            )
            row["experiment"] = dict(
                id=row["case"]["name"] + "-" + suffix,
                case=row["case"],
                kind=kind,
                species_indices=indices,
                chunk=1,
            )
            row["warm_fit"]["samples_seconds"] = [10.0 + 2 * (len(indices) - 1)] * 10
            row["machine"].update(
                measured_commit="native",
                source_tree="native-src",
                native_driver_sha256="driver",
                smoke=False,
                dirty_source_or_driver="",
            )
            row["initialization"] = dict(
                backbone="shared-backbone",
                matched_exactly=True,
                projection_rows={f"species-{index}": f"row-{index}" for index in indices},
            )
            row["input_sha256"] = {
                f"species-{index}": original["input_sha256"] for index in indices
            }
            groups = len(indices) * row["case"]["regions"]
            row["checks"] = [
                dict(check, species=f"species-{index}")
                for index in indices
                for check in original["checks"]
            ]
            row["accounting"].update(
                species=len(indices),
                species_chunk_size=1,
                region_batch_size=None,
                species_batch_size=None,
                finalized_group_calibrations=4 * groups
                if row["case"]["objective"] == "mlpe"
                else 0,
                observed=dict(
                    encodings=7 * row["case"]["regions"],
                    group_contributions=3 * groups,
                    evaluated_groups=4 * groups,
                ),
            )
            native["results"].append(row)
    return baseline, native


def test_complete_equivalent_report_passes_every_required_gate():
    baseline, native = reports()
    result = compare_reports([baseline], [native])
    assert result["status"] == "pass"
    assert len(result["single_species"]) == 12
    assert [
        row["ratio"] for family in result["joint_species"] for row in family["adjacent_increments"]
    ] == pytest.approx([0.2] * 6)


def test_fresh_ten_repeat_main_batch_supersedes_preserved_initial_samples():
    baseline, native = reports()
    repeated = copy.deepcopy(baseline)
    repeated["results"] = [
        row
        for row in repeated["results"]
        if row["case"]["name"] in repeated["protocol"]["gated_representative_cases"]
    ]
    native["protocol"] = copy.deepcopy(native["protocol"])
    baseline["protocol"]["fit_repeats"] = 3
    for row in baseline["results"]:
        row["warm_fit"]["samples_seconds"] = [10.0] * 3
        row["warm_training_steps_by_fit_seconds"] = [[10.0] * 3] * 3
    result = compare_reports([baseline, repeated], [native])
    assert result["status"] == "pass"
    assert result["repeated_main_cases"] == ["resnet-feature-heavy", "unet-feature-heavy"]


def test_duplicate_full_batches_cannot_select_favorable_samples():
    baseline, native = reports()
    with pytest.raises(ValueError, match="Duplicate"):
        compare_reports([baseline, copy.deepcopy(baseline)], [native])


@pytest.mark.parametrize("corruption", ["source", "input", "affinity", "fixture"])
def test_baseline_must_match_the_frozen_measured_workload(corruption):
    baseline, native = reports()
    row = baseline["results"][0]
    if corruption == "source":
        row["machine"]["source_tree"] = "different-source"
    elif corruption == "input":
        row["input_sha256"] = "different-input"
        native["results"][0]["input_sha256"]["species-0"] = "different-input"
    elif corruption == "affinity":
        row["machine"]["cpu_affinity"] = [2, 3]
    else:
        row["machine"]["fixtures_sha256"] = "different-fixture"
    with pytest.raises(ValueError, match="Frozen baseline"):
        compare_reports([baseline], [native])


def test_loglinear_timings_are_reported_without_a_main_ratio():
    baseline, native = reports()
    result = compare_reports([baseline], [native])
    assert len(result["loglinear"]) == 4
    four = next(row for row in result["loglinear"] if row["case"].endswith("heavy-joint-4"))
    assert four["fit_median_seconds"] == 16.0
    assert four["fit_bootstrap_95_percent_seconds"] == [16.0, 16.0]
    assert four["steady_step_median_seconds"] == 10.0
    assert "main_ratio" not in four


@pytest.mark.parametrize(
    "corruption", ["work", "initialization", "shape", "repeats", "protocol", "ids"]
)
def test_report_rejects_reduced_work_unmatched_initialization_and_fake_count(corruption):
    baseline, native = reports()
    row = next(
        value
        for value in native["results"]
        if value["experiment"]["id"] == "unet-feature-heavy-joint-4"
    )
    if corruption == "work":
        row["accounting"]["group_update_exposure"] = 2
    elif corruption == "initialization":
        row["initialization"]["projection_rows"]["species-0"] = "different"
    elif corruption == "shape":
        row["case"]["channels"] = 1
    elif corruption == "protocol":
        native["protocol"]["solver"]["rtol"] = 1e-3
    elif corruption == "ids":
        row["experiment"]["species_indices"] = [0]
    else:
        row["warm_fit"]["samples_seconds"] = [1.0]
    with pytest.raises(ValueError):
        compare_reports([baseline], [native])
