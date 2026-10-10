"""Measure the maintained JAX endpoint likelihood on CPU with float64 inputs.

Run PYTHONPATH=src python benchmarks/benchmark_mlpe.py --sizes 12 24 48 \\
    --output benchmarks/results/mlpe.json

Compilation and one warm call precede synchronized timings. Each size runs in a
fresh process. Whole-process peak RSS includes imports, compilation and allocator
workspace; it does not measure individual buffers or accelerator memory.
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


def _worker(n_populations, repeats):
    import resource

    import jax
    import jax.numpy as jnp
    import numpy as np

    from ilg_toolkit.mlpe import profiled_mlpe_ml_fit

    with jax.enable_x64():
        left, right = np.triu_indices(n_populations, 1)
        rng = np.random.default_rng(37)
        scores = jnp.asarray(rng.normal(size=len(left)))
        targets = 0.4 - 0.2 * scores + jnp.asarray(rng.normal(scale=0.3, size=len(left)))
        raw = jnp.log(jnp.expm1(jnp.asarray([0.5, 0.2]) - 1e-10))

        @jax.jit
        def evaluate(scores, targets, raw):
            return profiled_mlpe_ml_fit(
                scores, targets, left, right, n_populations=n_populations, raw_variances=raw
            )

        started = time.perf_counter()
        nll, beta = jax.block_until_ready(evaluate(scores, targets, raw))
        compile_and_first_call = time.perf_counter() - started
        jax.block_until_ready(evaluate(scores, targets, raw))
        samples = []
        for _ in range(repeats):
            started = time.perf_counter()
            jax.block_until_ready(evaluate(scores, targets, raw))
            samples.append(time.perf_counter() - started)
    rss_divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return dict(
        backend=jax.default_backend(),
        populations=n_populations,
        pairs=len(left),
        dtype=str(scores.dtype),
        compile_and_first_call_seconds=compile_and_first_call,
        warm_calls=1,
        median_seconds=statistics.median(samples),
        samples_seconds=samples,
        process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rss_divisor,
        nll=float(nll),
        beta=beta.tolist(),
        jax=jax.__version__,
        numpy=np.__version__,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[12, 24, 48])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1 or min(args.sizes) < 3:
        parser.error("sizes must be >=3 and repeats positive")
    if args.worker:
        print(json.dumps(_worker(args.sizes[0], args.repeats)))
        return
    environment = os.environ.copy()
    environment.update(
        OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", JAX_PLATFORMS="cpu"
    )
    results = []
    for size in args.sizes:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
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
        python=platform.python_version(),
        scope="JAX endpoint likelihood and signed GLS; synchronized CPU; whole-process peak RSS",
        repeats=args.repeats,
        requested_native_threads=1,
        results=results,
    )
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()
