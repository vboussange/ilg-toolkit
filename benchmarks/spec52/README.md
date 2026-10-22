# Spec #52 benchmarks

`perf/spec52-multispecies` preserves the runtime and memory benchmarks for the
multispecies implementation in `codex/spec52-integration`. This branch is never
merged. [RESULTS.md](RESULTS.md) contains the measurements and their limits.

The scripts use synthetic inputs and the toolkit source on this branch. Runtime
settings live in `fixtures.py`; `multispecies_fixtures.py` supplies matched joint
and separate species inputs. `main_driver.py` measures the single-species
baseline, `pr_driver.py` measures this branch, and `compare.py` evaluates their
runtime ratios. `large_registry_probe.py` and `large_registry_analysis.py`
measure and summarize memory for up to 1,000 species.

## Runtime

Run from a clean, committed checkout on Linux with CPU cores 0 and 1 available.
Install Python 3.12 or newer and the development dependencies with `uv sync
--extra dev`. Use the same dependency environment for both revisions; each
measurement records package versions, source identities and CPU affinity.
The drivers select CPU, enable float64 arithmetic, disable persistent compilation
caching, use one numerical-library thread and start a fresh process per case.

```sh
uv sync --extra dev
BENCH_PYTHON="$PWD/.venv/bin/python"
BENCH_BASELINE=$(mktemp -d)
BENCH_RESULTS=$(mktemp -d)
git clone --no-hardlinks . "$BENCH_BASELINE"
git -C "$BENCH_BASELINE" checkout 8439f99ea37f04273a1868b8009e1076239ad316

PYTHONPATH="$BENCH_BASELINE/src" "$BENCH_PYTHON" benchmarks/spec52/main_driver.py \
  --fit-repeats 10 --output "$BENCH_RESULTS/main.json"
PYTHONPATH=src "$BENCH_PYTHON" benchmarks/spec52/pr_driver.py \
  --suite all --chunk 4 --fit-repeats 10 --output "$BENCH_RESULTS/candidate.json"
"$BENCH_PYTHON" benchmarks/spec52/compare.py \
  --main "$BENCH_RESULTS/main.json" --pr "$BENCH_RESULTS/candidate.json" \
  --output "$BENCH_RESULTS/comparison.json"
```

The matrix has 12 baseline cases and 32 candidate cases: the same 12 cases,
16 joint/separate fits on two feature-heavy cases, and four LogLinear fits.
Each fit takes three optimizer steps with evaluation at initialization and after
every step. After one warm-up fit, ten repetitions measure the public fit;
updates 2 and 3 measure steady training. Predictions have five warmed repetitions.
All species use all their observation pairs at every update. Joint and separate
fits share initial pooled-backbone parameters and matching species projections.

Ratios use 10,000 bootstrap draws with seed 52053 and 95% intervals, resampling
whole fits separately for each revision. The upper interval must be at most
1.10 for feature-heavy fit/update parity and 0.50 for each added-species cost
relative to its separate fit. Inconclusive comparisons require more repetitions;
`compare.py` exits nonzero for missing, invalid, failed or inconclusive results.
Scientific checks include finite predictions, full calibration and dense-reference
resistance. Use `main_driver.py --list` to list cases. For a quick development
check, select `pr_driver.py --case unet-direct-one-single --chunk 4 --smoke` with
an output path; smoke results cannot pass the comparison.

## Memory

Commit changes before measuring. Each case needs a fresh process and empty
compiler-output directory. Keep these runs separate from timing measurements.
The loop measures both neural families, species counts 4/100/1,000, microbatch
sizes 1/4, and two cases with unequal sampling-unit counts.

```sh
BENCH_MEMORY=$(mktemp -d)
for family in resnet unet; do
  for size in 4 100 1000; do
    for batch in 1 4; do
      name="$family-s$size-c$batch"
      XLA_FLAGS="--xla_dump_to=$BENCH_MEMORY/$name --xla_dump_hlo_as_text" \
        JAX_ENABLE_X64=true JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 \
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
        PYTHONPATH=src taskset -c 0,1 "$BENCH_PYTHON" \
        benchmarks/spec52/large_registry_probe.py --family "$family" \
        --species "$size" --chunk "$batch" --xla-directory "$BENCH_MEMORY/$name" \
        --output "$BENCH_MEMORY/$name.json"
    done
  done
  name="$family-s4-c1-ragged"
  XLA_FLAGS="--xla_dump_to=$BENCH_MEMORY/$name --xla_dump_hlo_as_text" \
    JAX_ENABLE_X64=true JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    PYTHONPATH=src taskset -c 0,1 "$BENCH_PYTHON" \
    benchmarks/spec52/large_registry_probe.py --family "$family" \
    --species 4 --chunk 1 --ragged --xla-directory "$BENCH_MEMORY/$name" \
    --output "$BENCH_MEMORY/$name.json"
done
"$BENCH_PYTHON" benchmarks/spec52/large_registry_analysis.py "$BENCH_MEMORY" \
  --output "$BENCH_MEMORY/analysis.json"
```

Memory fits use two updates, four selected species per update, and initial/final
evaluations across every species. The analyzer requires all 14 cases and checks
calibration coverage, update counts, microbatch sizes and compilation reuse.
Generated JSON, arrays and compiler output stay local; `benchmarks/**/results/`
is ignored. The report contains CPU observations, with no GPU-memory estimate.
