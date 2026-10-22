"""Frozen-main public fit/predict benchmark; see README.md for scope and gates.

Run with frozen-main/src on PYTHONPATH. Each case uses its own child process;
never import this driver as a compatibility adapter in the production package.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import cast

from fixtures import fixture, protocol, protocol_digest, summary


def git(*args, cwd):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def environment_report(source, settings):
    import jax

    variables = (
        "PYTHONPATH",
        "JAX_ENABLE_X64",
        "JAX_PLATFORMS",
        "JAX_ENABLE_COMPILATION_CACHE",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "XLA_PYTHON_CLIENT_ALLOCATOR",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "XLA_FLAGS",
    )
    cpu = next(
        (
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        platform.processor(),
    )
    return dict(
        source_path=str(source),
        source_tree=git("rev-parse", "HEAD:src", cwd=source),
        frozen_main_commit=settings["frozen_main_commit"],
        benchmark_commit=git("rev-parse", "HEAD", cwd=Path(__file__).parent),
        protocol_sha256=protocol_digest(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        fixtures_sha256=hashlib.sha256(
            Path(__file__).with_name("fixtures.py").read_bytes()
        ).hexdigest(),
        platform=platform.platform(),
        python=platform.python_version(),
        interpreter=sys.executable,
        cpu=cpu,
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        mem_total=Path("/proc/meminfo").read_text().splitlines()[0],
        devices=[str(device) for device in jax.devices()],
        backend=jax.default_backend(),
        software={
            name: importlib.metadata.version(name)
            for name in (
                "jax",
                "jaxlib",
                "numpy",
                "scipy",
                "equinox",
                "optax",
                "lineax",
                "jaxscape",
                "coordax",
                "chex",
            )
        },
        environment={name: os.environ.get(name) for name in variables},
    )


def dense_resistance(surface, positions, patch_size):
    """Independent host pseudoinverse of the undirected arithmetic-edge grid."""
    import numpy as np

    rows, columns = surface.shape
    laplacian = np.zeros((surface.size, surface.size), dtype=np.float64)
    for row in range(rows):
        for column in range(columns):
            first = row * columns + column
            for r, c in ((row + 1, column), (row, column + 1)):
                if r < rows and c < columns:
                    second = r * columns + c
                    weight = (surface[row, column] + surface[r, c]) / 2
                    laplacian[first, first] += weight
                    laplacian[second, second] += weight
                    laplacian[first, second] -= weight
                    laplacian[second, first] -= weight
    inverse = np.linalg.pinv(laplacian, hermitian=True)
    distances = np.diag(inverse)[:, None] + np.diag(inverse)[None, :] - 2 * inverse
    cells = positions // patch_size
    nodes = cells[:, 0] * columns + cells[:, 1]
    return distances[np.ix_(nodes, nodes)]


def worker(case, settings):
    # Affinity is set before any JAX import/backend initialization.
    os.sched_setaffinity(0, settings["execution"]["cpu_affinity"])
    import coordax as cx
    import equinox as eqx
    import jax
    import numpy as np

    import ilg_toolkit
    import ilg_toolkit.training as training
    from ilg_toolkit import (
        PairwiseObservations,
        RegionBatch,
        ResistanceSolverConfig,
        TargetSpec,
        TrainingConfig,
        fit,
    )
    from ilg_toolkit.calibration import MLPECalibrationModule
    from ilg_toolkit.models import ResNet9Conductance, UNetEmbeddingDistance

    source = Path(ilg_toolkit.__file__).resolve().parents[2]
    expected = git("rev-parse", f"{settings['frozen_main_commit']}:src", cwd=source)
    actual = git("rev-parse", "HEAD:src", cwd=source)
    if actual != expected or git("status", "--porcelain", "--", "src", cwd=source):
        raise RuntimeError("Baseline driver requires unmodified frozen-main production sources")
    machine = environment_report(source, settings)
    started = time.perf_counter()
    data = fixture(case, settings)
    feature_names = tuple(f"covariate-{index}" for index in range(case["channels"]))
    regions = [
        RegionBatch(
            item["name"],
            cx.field(item["features"], "row", "column", "feature"),
            item["ids"],
            item["positions"],
            feature_names=feature_names,
        )
        for item in data
    ]
    observations = [
        PairwiseObservations.from_pairs(
            item["pairs"],
            item["targets"],
            target=TargetSpec("synthetic genetic dissimilarity", units="index"),
        )
        for item in data
    ]
    constructor = dict(
        in_channels=case["channels"],
        patch_size=case["patch_size"],
        key=jax.random.key(settings["model_seed"]),
    )
    if case["family"] == "resnet":
        model = ResNet9Conductance(
            **constructor,
            **settings["resnet"],
            solver_config=ResistanceSolverConfig(**settings["solver"]),
        )
    else:
        model = UNetEmbeddingDistance(**constructor, **settings["unet"])
    jax.block_until_ready(eqx.filter(model, eqx.is_array))
    prepare_seconds = time.perf_counter() - started
    parameter_leaves = jax.tree.leaves(eqx.filter(model, eqx.is_inexact_array))
    config = TrainingConfig(
        objective=case["objective"],
        epochs=settings["steps"],
        learning_rate=settings["learning_rate"],
        seed=settings["training_seed"],
        jit=True,
        mlpe_variance_floor=settings["mlpe"]["variance_floor"],
        mlpe_jitter=settings["mlpe"]["jitter"],
        mlpe_initial_variances=tuple(settings["mlpe"]["initial_variances"]),
    )
    # The old public API has no pre-evaluation update event. Instrument its
    # synchronized host boundary, never compiled computation or numerical data.
    checked_call = training._checked_call
    step_started = {}
    step_samples = []

    def timed_checked_call(function, *args, **kwargs):
        epoch = kwargs["epoch"]
        if function is training._compiled_joint_gradient and epoch not in step_started:
            step_started[epoch] = time.perf_counter()
        result = checked_call(function, *args, **kwargs)
        if "optimizer" in kwargs:
            step_samples.append(time.perf_counter() - step_started.pop(epoch))
        return result

    training._checked_call = timed_checked_call

    def fit_once():
        intervals = []
        step_samples.clear()
        previous = time.perf_counter()
        started_fit = previous

        def record(state):
            nonlocal previous
            jax.block_until_ready(eqx.filter(state, eqx.is_array))
            now = time.perf_counter()
            intervals.append(dict(epoch=state.epoch, seconds=now - previous))
            previous = now

        state = fit(regions, observations, distance_model=model, config=config, on_epoch=record)
        jax.block_until_ready(eqx.filter(state, eqx.is_array))
        elapsed = time.perf_counter() - started_fit
        if len(step_samples) != settings["steps"]:
            raise AssertionError("Training-step instrumentation did not observe every update")
        return state, elapsed, intervals, list(step_samples)

    state, first_fit, cold_intervals, cold_steps = fit_once()
    for _ in range(settings["warm_fit_calls"]):
        fit_once()
    fit_samples, warm_steps = [], []
    for _ in range(settings["fit_repeats"]):
        state, elapsed, _, steps = fit_once()
        fit_samples.append(elapsed)
        warm_steps.append(steps)
    training._checked_call = checked_call

    def predict_once():
        started_predict = time.perf_counter()
        predictions = [
            state.model.predict(region, pairs=item["pairs"])
            for region, item in zip(regions, data, strict=True)
        ]
        jax.block_until_ready([prediction.values for prediction in predictions])
        return predictions, time.perf_counter() - started_predict

    predictions, first_predict = predict_once()
    predict_once()
    prediction_samples = [predict_once()[1] for _ in range(settings["prediction_repeats"])]
    rss_divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    timed_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rss_divisor
    device_stats = [device.memory_stats() for device in jax.devices()]

    checks = []
    thresholds = settings["thresholds"]
    for region, item, prediction in zip(regions, data, predictions, strict=True):
        raw = np.asarray(jax.block_until_ready(state.model.ecological_distances(region)))
        np.testing.assert_allclose(raw, raw.T, atol=thresholds["raw_symmetry_atol"], rtol=0)
        np.testing.assert_allclose(np.diag(raw), 0, atol=thresholds["raw_diagonal_atol"], rtol=0)
        if not np.isfinite(raw).all() or raw.min() < -thresholds["raw_nonnegative_atol"]:
            raise AssertionError("Raw ecological distances must be finite and nonnegative")
        if not np.isfinite(prediction.values).all() or len(prediction.pairs) != len(item["pairs"]):
            raise AssertionError("Prediction omitted pairs or produced nonfinite values")
        left, right = np.triu_indices(case["units"], 1)
        eligible = dict(
            region=region.name,
            pairs=len(item["pairs"]),
            raw_min=float(raw.min()),
            raw_max=float(raw.max()),
            raw_pair_std=float(raw[left, right].std()),
            prediction_min=float(np.min(prediction.values)),
            prediction_max=float(np.max(prediction.values)),
        )
        if case["family"] == "resnet":
            surface = np.asarray(state.model.conductance_surface(region))
            expected = dense_resistance(surface, item["positions"], case["patch_size"])
            np.testing.assert_allclose(
                raw,
                expected,
                rtol=thresholds["dense_resistance_rtol"],
                atol=thresholds["dense_resistance_atol"],
            )
            eligible["dense_resistance_max_absolute_error"] = float(np.max(abs(raw - expected)))
        if case["objective"] == "mlpe":
            if eligible["raw_pair_std"] <= thresholds["raw_mlpe_min_standard_deviation"]:
                raise AssertionError("MLPE distances are scientifically degenerate")
            head = cast(MLPECalibrationModule, state.model.calibration).for_region(region.name)
            if len(head.calibration_pairs) != len(item["pairs"]):
                raise AssertionError("Final calibration omitted observations")
            eligible["final_calibration_pairs"] = len(head.calibration_pairs)
            eligible["intercept"] = float(head.intercept)
            eligible["slope"] = float(head.slope)
        checks.append(eligible)
    for leaf in jax.tree.leaves(eqx.filter(state, eqx.is_array)):
        if not np.isfinite(np.asarray(leaf)).all():
            raise AssertionError("Fitted state contains nonfinite values")
    if not np.isfinite([record.training_loss for record in state.history]).all():
        raise AssertionError("History contains nonfinite objectives")
    return dict(
        case=case,
        machine=machine,
        scientific_eligibility="passed",
        checks=checks,
        input_sha256={item["name"]: item["sha256"] for item in data},
        prepare_and_model_init_seconds=prepare_seconds,
        parameter_count=sum(leaf.size for leaf in parameter_leaves),
        parameter_bytes=sum(leaf.size * leaf.dtype.itemsize for leaf in parameter_leaves),
        compile_plus_first_fit_seconds=first_fit,
        cold_callback_intervals=cold_intervals,
        cold_training_steps_seconds=cold_steps,
        warm_fit=summary(fit_samples),
        warm_training_step=summary([value for steps in warm_steps for value in steps]),
        warm_training_steps_by_fit_seconds=warm_steps,
        first_predict_seconds=first_predict,
        warm_predict=summary(prediction_samples),
        process_peak_rss_after_timed_work_mib=timed_peak,
        process_peak_rss_after_validation_mib=(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rss_divisor
        ),
        device_allocator_statistics=device_stats,
        memory_scope="Whole-process high-water RSS, includes JIT/compiler and allocator caches; "
        "not live or per-step memory. CPU device stats unavailable when null.",
        training_history=[float(record.training_loss) for record in state.history],
        accounting=dict(
            species=1,
            optimizer_updates=settings["steps"],
            group_update_exposure=settings["steps"],
            regions=case["regions"],
            pairs_per_group=case["units"] * (case["units"] - 1) // 2,
            full_evaluations=settings["steps"] + 1,
            finalized_group_calibrations=(
                case["regions"] * (settings["steps"] + 1) if case["objective"] == "mlpe" else 0
            ),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case", action="append")
    parser.add_argument(
        "--fit-repeats", type=int, help="Override repetitions, preserving the frozen workload"
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    settings = protocol()
    if args.fit_repeats is not None:
        if args.fit_repeats < 3:
            parser.error("fit repetitions must be at least three")
        settings["fit_repeats"] = args.fit_repeats
    cases = [case for case in settings["cases"] if not args.case or case["name"] in args.case]
    if not cases or (args.case and set(args.case) - {case["name"] for case in cases}):
        parser.error("unknown or empty case selection")
    if args.list:
        print("\n".join(case["name"] for case in cases))
        return
    if args.worker:
        print(json.dumps(worker(cases[0], settings), allow_nan=False))
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
    report = dict(
        protocol=settings,
        results=[],
        command=sys.argv,
        scope="Frozen-main, fixed-budget, synchronized CPU public workflows",
    )
    for case in cases:
        print(f"Running {case['name']}", file=sys.stderr, flush=True)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--case",
            case["name"],
        ]
        if args.fit_repeats is not None:
            command.extend(["--fit-repeats", str(args.fit_repeats)])
        result = subprocess.run(command, env=environment, text=True, capture_output=True)
        if result.returncode:
            print(result.stderr, file=sys.stderr)
            raise RuntimeError(f"Benchmark case {case['name']} failed: {result.returncode}")
        report["results"].append(json.loads(result.stdout))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"Completed {case['name']}", file=sys.stderr, flush=True)
    if not args.output:
        print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
