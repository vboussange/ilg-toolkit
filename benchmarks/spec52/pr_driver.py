"""Native public-workflow measurements under the immutable spec52 protocol.

Each experiment runs in a fresh process. Results retain failed workers and exact
source/initialization identities. See README.md for scientific and timing scope.
"""

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

from fixtures import protocol, summary
from main_driver import dense_resistance, environment_report, git
from multispecies_fixtures import species_fixture


def array_digest(tree):
    import equinox as eqx
    import jax
    import numpy as np

    digest = hashlib.sha256()
    for leaf in jax.tree.leaves(eqx.filter(tree, eqx.is_array)):
        value = np.asarray(leaf)
        digest.update(str((value.shape, value.dtype)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def initial_model(case, settings, indices):
    """Copy exact joint-backbone/output rows into each native experiment."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from ilg_toolkit import ResistanceSolverConfig
    from ilg_toolkit.models import LogLinearConductance, ResNet9Conductance, UNetEmbeddingDistance

    species = tuple(f"species-{index}" for index in indices)
    common = dict(in_channels=case["channels"], patch_size=case["patch_size"])
    if case["family"] == "loglinear":
        model = LogLinearConductance(
            **common, species=species, solver_config=ResistanceSolverConfig(**settings["solver"])
        )
        return model, {"fixed_features": True, "model": array_digest(model)}
    constructor = ResNet9Conductance if case["family"] == "resnet" else UNetEmbeddingDistance
    common.update(key=jax.random.key(settings["model_seed"]))
    common.update(settings[case["family"]])
    if case["family"] == "resnet":
        common["solver_config"] = ResistanceSolverConfig(**settings["solver"])
    canonical = constructor(**common, species=tuple(f"species-{index}" for index in range(4)))
    selected = jnp.asarray(indices)
    model = constructor(**common, species=species)
    model = eqx.tree_at(
        lambda value: (value.backbone, value.projection_weight, value.projection_bias),
        model,
        (
            canonical.backbone,
            canonical.projection_weight[selected],
            canonical.projection_bias[selected],
        ),
    )
    backbone = array_digest(model.backbone)
    if backbone != array_digest(canonical.backbone):
        raise AssertionError("Independent/joint backbone initialization mismatch")
    rows = {}
    for row, index in enumerate(indices):
        actual = array_digest((model.projection_weight[row], model.projection_bias[row]))
        expected = array_digest(
            (canonical.projection_weight[index], canonical.projection_bias[index])
        )
        if actual != expected:
            raise AssertionError("Independent/joint projection initialization mismatch")
        rows[f"species-{index}"] = actual
    return model, dict(backbone=backbone, projection_rows=rows, matched_exactly=True)


def worker(experiment, settings, *, smoke=False):
    os.sched_setaffinity(0, settings["execution"]["cpu_affinity"])
    import coordax as cx
    import equinox as eqx
    import jax
    import numpy as np

    import ilg_toolkit
    from ilg_toolkit import (
        PairwiseObservations,
        RegionBatch,
        SamplingUnits,
        TargetSpec,
        TrainingConfig,
        fit,
        training,
    )
    from ilg_toolkit import _training_execution as execution
    from ilg_toolkit.calibration import UnpooledMLPECalibrationModule

    source = Path(ilg_toolkit.__file__).resolve().parents[2]
    dirty = git("status", "--porcelain", "--", "src", "benchmarks/spec52", cwd=source)
    if dirty and not smoke:
        raise RuntimeError("Commit source/driver changes before timed measurements")
    case = experiment["case"]
    indices = experiment["species_indices"]
    machine: dict = environment_report(source, settings)
    machine.update(
        measured_commit=git("rev-parse", "HEAD", cwd=source),
        native_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        multispecies_fixtures_sha256=hashlib.sha256(
            Path(__file__).with_name("multispecies_fixtures.py").read_bytes()
        ).hexdigest(),
        smoke=smoke,
        dirty_source_or_driver=dirty,
    )
    started = time.perf_counter()
    data = {f"species-{index}": species_fixture(case, settings, index) for index in indices}
    names = tuple(data)
    feature_names = tuple(f"covariate-{index}" for index in range(case["channels"]))
    regions, observations = [], {}
    for region_index, item in enumerate(data[names[0]]):
        regions.append(
            RegionBatch(
                item["name"],
                cx.field(item["features"], "row", "column", "feature"),
                {
                    name: SamplingUnits(
                        items[region_index]["ids"], items[region_index]["positions"]
                    )
                    for name, items in data.items()
                },
                feature_names=feature_names,
            )
        )
        observations[item["name"]] = {
            name: PairwiseObservations.from_pairs(
                items[region_index]["pairs"],
                items[region_index]["targets"],
                target=TargetSpec("synthetic genetic dissimilarity", units="index"),
            )
            for name, items in data.items()
        }
    model, initialization = initial_model(case, settings, indices)
    jax.block_until_ready(eqx.filter(model, eqx.is_array))
    prepare_seconds = time.perf_counter() - started
    parameter_leaves = jax.tree.leaves(eqx.filter(model, eqx.is_inexact_array))
    config = TrainingConfig(
        objective=case["objective"],
        steps=settings["steps"],
        region_batch_size=None,
        species_batch_size=None,
        species_chunk_size=experiment["chunk"],
        evaluation_interval=1,
        learning_rate=settings["learning_rate"],
        seed=settings["training_seed"],
        jit=True,
        mlpe_variance_floor=settings["mlpe"]["variance_floor"],
        mlpe_jitter=settings["mlpe"]["jitter"],
        mlpe_initial_variances=tuple(settings["mlpe"]["initial_variances"]),
    )
    checked = training._checked_call
    step_started, step_samples, counts = {}, [], {}

    def observe(function, *args, **kwargs):
        step = kwargs["step"]
        fused = function is execution.compiled_group_gradient
        if fused:
            counts["encodings"] += 1
            if step not in step_started:
                step_started[step] = time.perf_counter()
        if function in (execution.pooled_region, execution.compiled_pooled_region):
            counts["encodings"] += 1
            if not kwargs.get("inference", True) and step not in step_started:
                step_started[step] = time.perf_counter()
        result = checked(function, *args, **kwargs)
        if fused:
            counts["group_contributions"] += len(args[2])
        elif function is execution.compiled_chunk_gradient:
            counts["group_contributions"] += args[1].species_index.shape[0]
        elif function is execution.compiled_chunk_evaluate:
            counts["evaluated_groups"] += args[3].species_index.shape[0]
        elif function is training._compiled_update:
            step_samples.append(time.perf_counter() - step_started.pop(step))
        return result

    def fit_once():
        step_samples.clear()
        step_started.clear()
        counts.update(encodings=0, group_contributions=0, evaluated_groups=0)
        intervals = []
        started_fit = previous = time.perf_counter()

        def record(state):
            nonlocal previous
            jax.block_until_ready(eqx.filter(state, eqx.is_array))
            now = time.perf_counter()
            intervals.append(dict(step=state.step, seconds=now - previous))
            previous = now

        state = fit(regions, observations, distance_model=model, config=config, on_step=record)
        jax.block_until_ready(eqx.filter(state, eqx.is_array))
        elapsed = time.perf_counter() - started_fit
        groups = case["regions"] * len(indices)
        if len(step_samples) != settings["steps"] or step_started:
            raise AssertionError("Every complete optimizer step must be observed exactly once")
        expected = dict(
            encodings=case["regions"] * (2 * settings["steps"] + 1),
            group_contributions=groups * settings["steps"],
            evaluated_groups=groups * (settings["steps"] + 1),
        )
        if counts != expected or [record.step for record in state.history] != list(
            range(settings["steps"] + 1)
        ):
            raise AssertionError(f"Scientific work changed: {counts}, expected {expected}")
        return state, elapsed, intervals, list(step_samples), dict(counts)

    training._checked_call = observe
    try:
        state, first_fit, cold_intervals, cold_steps, work = fit_once()
        for _ in range(settings["warm_fit_calls"]):
            fit_once()
        fit_samples, warm_steps = [], []
        for _ in range(settings["fit_repeats"]):
            state, elapsed, _, steps, work = fit_once()
            fit_samples.append(elapsed)
            warm_steps.append(steps)
    finally:
        training._checked_call = checked

    def predict_once():
        started_predict = time.perf_counter()
        predictions = [
            state.model.predict(
                region,
                species=names,
                pairs={name: data[name][region_index]["pairs"] for name in names},
            )
            for region_index, region in enumerate(regions)
        ]
        jax.block_until_ready([result.values for group in predictions for result in group.values()])
        return predictions, time.perf_counter() - started_predict

    predictions, first_predict = predict_once()
    predict_once()
    prediction_samples = [predict_once()[1] for _ in range(settings["prediction_repeats"])]
    rss_divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    timed_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rss_divisor
    checks, thresholds = [], settings["thresholds"]
    for region_index, region in enumerate(regions):
        matrices = state.model.ecological_distances(region, species=names)
        surfaces = (
            state.model.conductance_surface(region, species=names)
            if case["family"] in {"resnet", "loglinear"}
            else {}
        )
        for name in names:
            item, prediction = data[name][region_index], predictions[region_index][name]
            raw = np.asarray(jax.block_until_ready(matrices[name]))
            np.testing.assert_allclose(raw, raw.T, atol=thresholds["raw_symmetry_atol"], rtol=0)
            np.testing.assert_allclose(
                np.diag(raw), 0, atol=thresholds["raw_diagonal_atol"], rtol=0
            )
            if not np.isfinite(raw).all() or raw.min() < -thresholds["raw_nonnegative_atol"]:
                raise AssertionError("Raw distances must be finite and nonnegative")
            if not np.isfinite(prediction.values).all() or tuple(prediction.pairs) != item["pairs"]:
                raise AssertionError(
                    "Prediction changed requested pairs or produced nonfinite values"
                )
            left, right = np.triu_indices(case["units"], 1)
            eligible: dict = dict(
                region=region.name,
                species=name,
                pairs=len(item["pairs"]),
                raw_dtype=str(raw.dtype),
                raw_min=float(raw.min()),
                raw_max=float(raw.max()),
                raw_pair_std=float(raw[left, right].std()),
            )
            if surfaces:
                expected = dense_resistance(
                    np.asarray(surfaces[name]), item["positions"], case["patch_size"]
                )
                np.testing.assert_allclose(
                    raw,
                    expected,
                    rtol=thresholds["dense_resistance_rtol"],
                    atol=thresholds["dense_resistance_atol"],
                )
                eligible["dense_resistance_max_absolute_error"] = float(np.max(abs(raw - expected)))
            if case["objective"] == "mlpe":
                if eligible["raw_pair_std"] <= thresholds["raw_mlpe_min_standard_deviation"]:
                    raise AssertionError("Scientifically degenerate MLPE distances")
                assert isinstance(state.model.calibration, UnpooledMLPECalibrationModule)
                head = state.model.calibration.for_group(region.name, name)
                if (
                    len(head.calibration_pairs) != len(item["pairs"])
                    or min(head.unit_variance, head.residual_variance) <= 0
                ):
                    raise AssertionError("Incomplete or invalid MLPE calibration")
                eligible.update(
                    final_calibration_pairs=len(head.calibration_pairs),
                    intercept=float(head.intercept),
                    slope=float(head.slope),
                    variances=[head.unit_variance, head.residual_variance],
                )
            checks.append(eligible)
    for leaf in jax.tree.leaves(eqx.filter(state, eqx.is_array)):
        if not np.isfinite(np.asarray(leaf)).all():
            raise AssertionError("Nonfinite fitted state")
    history = [float(record.training_loss) for record in state.history]
    if not np.isfinite(history).all():
        raise AssertionError("Nonfinite history")
    return dict(
        experiment=experiment,
        case=case,
        machine=machine,
        scientific_eligibility="passed",
        checks=checks,
        initialization=initialization,
        input_sha256={
            name: {item["name"]: item["sha256"] for item in items} for name, items in data.items()
        },
        prepare_and_model_init_seconds=prepare_seconds,
        parameter_count=sum(leaf.size for leaf in parameter_leaves),
        parameter_bytes=sum(leaf.size * leaf.dtype.itemsize for leaf in parameter_leaves),
        compile_plus_first_fit_seconds=first_fit,
        cold_callback_intervals=cold_intervals,
        cold_training_steps_seconds=cold_steps,
        warm_fit=summary(fit_samples),
        warm_training_steps_by_fit_seconds=warm_steps,
        first_predict_seconds=first_predict,
        warm_predict=summary(prediction_samples),
        process_peak_rss_after_timed_work_mib=timed_peak,
        process_peak_rss_after_validation_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / rss_divisor,
        device_allocator_statistics=[device.memory_stats() for device in jax.devices()],
        training_history=history,
        accounting=dict(
            species=len(names),
            optimizer_updates=settings["steps"],
            group_update_exposure=settings["steps"],
            regions=case["regions"],
            pairs_per_group=case["units"] * (case["units"] - 1) // 2,
            full_evaluations=settings["steps"] + 1,
            finalized_group_calibrations=(
                case["regions"] * len(names) * (settings["steps"] + 1)
                if case["objective"] == "mlpe"
                else 0
            ),
            observed=work,
            species_chunk_size=experiment["chunk"],
            region_batch_size=None,
            species_batch_size=None,
        ),
    )


def experiments(settings, suite, chunk):
    output = []
    for case in settings["cases"]:
        if suite in {"single", "all"}:
            output.append(
                dict(
                    id=case["name"] + "-single",
                    case=case,
                    kind="single",
                    species_indices=[0],
                    chunk=chunk,
                )
            )
        if suite in {"joint", "all"} and case["name"] in settings["gated_representative_cases"]:
            for count in settings["joint_species_counts"]:
                output.append(
                    dict(
                        id=f"{case['name']}-joint-{count}",
                        case=case,
                        kind="joint",
                        species_indices=list(range(count)),
                        chunk=chunk,
                    )
                )
            for index in range(4):
                output.append(
                    dict(
                        id=f"{case['name']}-independent-{index}",
                        case=case,
                        kind="independent",
                        species_indices=[index],
                        chunk=chunk,
                    )
                )
        if suite in {"loglinear", "all"} and case["name"] in {
            "resnet-feature-heavy",
            "resnet-solver-stress",
        }:
            new_case = dict(
                case, name=case["name"].replace("resnet", "loglinear"), family="loglinear"
            )
            for count in (1, 4):
                output.append(
                    dict(
                        id=f"{new_case['name']}-joint-{count}",
                        case=new_case,
                        kind="loglinear",
                        species_indices=list(range(count)),
                        chunk=chunk,
                    )
                )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--suite", choices=("single", "joint", "loglinear", "all"), default="single"
    )
    parser.add_argument("--case", action="append", help="Exact experiment ID")
    parser.add_argument("--fit-repeats", type=int, default=3)
    parser.add_argument("--chunk", type=int, default=1)
    parser.add_argument(
        "--smoke", action="store_true", help="Diagnostic dirty-tree run, never gate evidence"
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.output.exists() and not args.worker:
        parser.error("Preserve existing measurements; choose a new output filename")
    if args.fit_repeats < 3 or args.chunk < 1:
        parser.error("Require at least three fit repetitions and positive chunk size")
    settings = protocol()
    settings["fit_repeats"] = args.fit_repeats
    selected = experiments(settings, args.suite, args.chunk)
    if args.case:
        selected = [value for value in selected if value["id"] in args.case]
        if set(args.case) - {value["id"] for value in selected}:
            parser.error("Unknown experiment ID")
    if args.worker:
        print(json.dumps(worker(selected[0], settings, smoke=args.smoke), allow_nan=False))
        return
    environment = os.environ.copy()
    environment.update(
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        JAX_PLATFORMS="cpu",
        JAX_ENABLE_X64="true",
        JAX_ENABLE_COMPILATION_CACHE="false",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
    )
    report = dict(protocol=settings, results=[], failures=[], command=sys.argv, smoke=args.smoke)
    for experiment in selected:
        print(f"Running {experiment['id']}", file=sys.stderr, flush=True)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--suite",
            args.suite,
            "--case",
            experiment["id"],
            "--fit-repeats",
            str(args.fit_repeats),
            "--chunk",
            str(args.chunk),
            "--output",
            str(args.output),
        ]
        if args.smoke:
            command.append("--smoke")
        result = subprocess.run(command, env=environment, text=True, capture_output=True)
        if result.returncode:
            report["failures"].append(
                dict(
                    experiment=experiment,
                    returncode=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                )
            )
            print(result.stderr, file=sys.stderr)
        else:
            report["results"].append(json.loads(result.stdout))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
