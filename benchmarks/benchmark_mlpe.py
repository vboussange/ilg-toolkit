"""Compare exact host calibration kernels in isolated CPU processes.

Run after installing the package, or with PYTHONPATH=src from the repository:
  python benchmarks/benchmark_mlpe.py --sizes 12 24 48 72 --repeats 5 \
      --output /tmp/mlpe-scaling.json

Both implementations use float64 Cholesky and precompute their covariance
crossproduct once, outside timings. This measures host numerical work, including
factorization and signed GLS; it excludes encoder/solver time and optimization
iterations. ru_maxrss is whole-process CPU peak RSS, including imports, native
BLAS workspace, and allocator effects. It is not GPU/device peak memory.
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path


def _worker(backend, n_populations, repeats):
    import resource

    import numpy as np
    import scipy
    from scipy.linalg import cho_solve

    from ilg_toolkit.mlpe.fit import _profile
    from ilg_toolkit.mlpe.numpy_system import NumpyPopulationSystem, endpoint_gram

    rss_divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    baseline_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rss_divisor
    left, right = np.triu_indices(n_populations, 1)
    rng = np.random.default_rng(37)
    scores = rng.normal(size=len(left))
    targets = 0.4 - 0.2 * scores + rng.normal(scale=0.3, size=len(left))
    design = np.column_stack((np.ones(len(left)), (scores - scores.mean()) / scores.std(ddof=1)))
    unit, residual = 0.5, 0.2
    if backend == "dense":
        incidence = np.zeros((len(left), n_populations))
        incidence[np.arange(len(left)), left] = incidence[np.arange(len(left)), right] = 1
        crossproduct = incidence @ incidence.T
        del incidence

        def evaluate():
            covariance = unit * crossproduct + residual * np.eye(len(left))
            factor = np.linalg.cholesky(covariance)
            vinv_x = cho_solve((factor, True), design)
            beta = np.linalg.solve(design.T @ vinv_x, design.T @ cho_solve((factor, True), targets))
            r = targets - design @ beta
            nll = 0.5 * (
                len(left) * np.log(2 * np.pi)
                + 2 * np.log(np.diag(factor)).sum()
                + r @ cho_solve((factor, True), r)
            )
            return nll, beta
    else:
        gram = endpoint_gram(left, right, n_populations)

        def evaluate():
            system = NumpyPopulationSystem(left, right, gram, unit, residual)
            return _profile(design, targets, system)

    nll, beta = evaluate()  # Warm native-library setup and allocator reuse.
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        evaluate()  # NumPy/SciPy host operations are synchronous.
        samples.append(time.perf_counter() - start)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rss_divisor
    return dict(
        backend=backend,
        populations=n_populations,
        pairs=len(left),
        dtype="float64",
        median_seconds=statistics.median(samples),
        samples_seconds=samples,
        baseline_peak_rss_mib=baseline_rss,
        process_peak_rss_mib=peak_rss,
        increase_peak_rss_mib=max(0, peak_rss - baseline_rss),
        nll=float(nll),
        beta=beta.tolist(),
        numpy=np.__version__,
        scipy=scipy.__version__,
    )


def _cpu_name():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[12, 24, 48, 72])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", choices=["dense", "endpoint"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1 or min(args.sizes) < 3:
        parser.error("sizes must be >=3 and repeats positive")
    if args.worker:
        print(json.dumps(_worker(args.worker, args.sizes[0], args.repeats)))
        return
    environment = os.environ.copy()
    environment.update(
        OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", JAX_PLATFORMS="cpu"
    )
    results = []
    for size in args.sizes:
        for backend in ("dense", "endpoint"):
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                backend,
                "--sizes",
                str(size),
                "--repeats",
                str(args.repeats),
            ]
            result = subprocess.run(
                command, env=environment, check=True, text=True, capture_output=True
            )
            results.append(json.loads(result.stdout))
    report = dict(
        platform=platform.platform(),
        processor=_cpu_name(),
        python=platform.python_version(),
        scope="host CPU calibration numerical kernel; whole-process ru_maxrss; no device memory",
        repeats=args.repeats,
        native_threads=1,
        results=results,
    )
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()
