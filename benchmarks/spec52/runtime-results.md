# Runtime acceptance for native multispecies workflows

Status: **PASS**. All 32 PR workloads completed with ten synchronized full-fit repetitions each on clean commit `df7aea21ec55ce39bf34d78e53668ab8d4c86f44`; production source tree `8d92e023e4d4b29beca50e0c7028572ef7dd78ba`. The final candidate uniformly uses species chunk 4. The frozen protocol, scientific work and thresholds are unchanged.

These are CPU measurements with two-core affinity, float32 neural parameters/features and float64 solver/MLPE arithmetic. The main target remains `8439f99ea37f04273a1868b8009e1076239ad316`. Returned conductance distances retain the surface dtype, as on main. Raw inputs, initialization, source/driver hashes and software are recorded in each worker result.

## Required gates

Intervals are seeded 95% independent bootstrap intervals over complete fit blocks, with 10,000 draws. Steady steps are updates 2/3, retained together within each fit. Passing requires the entire interval to be at or below the frozen limit; a missing, failed or inconclusive gate exits nonzero.

| Feature-heavy family | Complete fit / main (95% interval; limit 1.10) | Steady update / main (95% interval; limit 1.10) |
| --- | ---: | ---: |
| resnet-feature-heavy | 0.5442 [0.5378, 0.5483] pass | 1.0190 [1.0127, 1.0260] pass |
| unet-feature-heavy | 0.0437 [0.0423, 0.0442] pass | 0.6256 [0.4329, 0.8328] pass |

| Family | Added species | Increment / matching added independent fit (95% interval; limit 0.50) |
| --- | ---: | ---: |
| resnet-feature-heavy | 1→2 | 0.0185 [0.0068, 0.0384] pass |
| resnet-feature-heavy | 2→3 | 0.0285 [0.0095, 0.0363] pass |
| resnet-feature-heavy | 3→4 | 0.0197 [0.0086, 0.0501] pass |
| unet-feature-heavy | 1→2 | 0.4449 [0.4201, 0.4665] pass |
| unet-feature-heavy | 2→3 | 0.2110 [0.1915, 0.2570] pass |
| unet-feature-heavy | 3→4 | 0.2257 [0.1762, 0.2695] pass |

Every independent fit copies the exact joint backbone and selected output row; each shared input array and initialization row is checked by hash. Signed marginal increments are retained. Each species receives its complete pair set on every one of the three updates, and all groups receive all four evaluations, with complete returned calibrations for MLPE workloads.

| Family | Species | Joint median(s) | Sum independent medians(s) | Joint / sum |
| --- | ---: | ---: | ---: | ---: |
| resnet-feature-heavy | 1 | 1.435773 | 1.419982 | 1.0111 |
| resnet-feature-heavy | 2 | 1.462061 | 2.843287 | 0.5142 |
| resnet-feature-heavy | 3 | 1.503219 | 4.285217 | 0.3508 |
| resnet-feature-heavy | 4 | 1.531203 | 5.705046 | 0.2684 |
| unet-feature-heavy | 1 | 0.072908 | 0.071234 | 1.0235 |
| unet-feature-heavy | 2 | 0.104960 | 0.143273 | 0.7326 |
| unet-feature-heavy | 3 | 0.120118 | 0.215096 | 0.5584 |
| unet-feature-heavy | 4 | 0.136172 | 0.286235 | 0.4757 |

## All frozen single-species workloads

| Case | Main fit(s) | PR fit(s) | Main steady(ms) | PR steady(ms) | Fit ratio | Step ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| resnet-direct-one | 1.367403 | 0.128051 | 24.248 | 21.590 | 0.0936 | 0.8904 |
| resnet-direct-multi | 1.437448 | 0.205308 | 38.185 | 39.611 | 0.1428 | 1.0374 |
| resnet-mlpe-one | 1.448814 | 0.135235 | 27.989 | 23.527 | 0.0933 | 0.8406 |
| resnet-mlpe-multi | 1.487203 | 0.242693 | 45.229 | 43.902 | 0.1632 | 0.9707 |
| unet-direct-one | 1.770295 | 0.062304 | 13.541 | 4.837 | 0.0352 | 0.3572 |
| unet-direct-multi | 1.758729 | 0.095040 | 13.188 | 8.737 | 0.0540 | 0.6625 |
| unet-mlpe-one | 1.753563 | 0.075811 | 11.233 | 5.162 | 0.0432 | 0.4596 |
| unet-mlpe-multi | 1.815819 | 0.118885 | 11.330 | 9.434 | 0.0655 | 0.8327 |
| resnet-feature-heavy | 2.613931 | 1.422631 | 360.269 | 367.119 | 0.5442 | 1.0190 |
| unet-feature-heavy | 1.656364 | 0.072347 | 10.516 | 6.578 | 0.0437 | 0.6256 |
| resnet-solver-stress | 2.272584 | 1.184184 | 203.899 | 204.041 | 0.5211 | 1.0007 |
| unet-mlpe-stress | 2.577926 | 1.606350 | 16.076 | 14.062 | 0.6231 | 0.8747 |

Only the two predeclared feature-heavy cases impose parity gates; the factorial direct/MLPE and one/two-region cases plus solver/MLPE stress cases remain mandatory completed scientific workloads. The large short-fit improvements include removal of main’s recurring first-update overhead when public fit recreates its optimizer. They do not imply the same speedup for long training or convergence.

Two non-gated steady-step diagnostics are inconclusive at 1.10 under the original three-fit main baseline: resnet-direct-multi 1.0374 [0.8460, 1.2755]; resnet-mlpe-multi 0.9707 [0.8437, 1.1326]. These intervals do not establish parity for those workloads; all required scientific work completed, and the predeclared feature-heavy gates remain unchanged.

| Feature-heavy family | Main recurring first update(s) | PR recurring first update(s) |
| --- | ---: | ---: |
| resnet-feature-heavy | 1.569124 | 0.366085 |
| unet-feature-heavy | 1.581675 | 0.006989 |

## Fixed-feature LogLinear

These fits have fixed patch summaries and learned species-specific coefficients/intercepts. They have no learned backbone and no main baseline. Complete pair/solver/MLPE/calibration work is retained.

| Workload | Fit median(s),95% interval | Steady update(ms) | Compile+first fit(s) | Warm predict(ms) | Peak process RSS(MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| loglinear-feature-heavy-joint-1 | 0.039246 [0.038807, 0.041656] | 3.285 | 5.768 | 14.509 | 711.8 |
| loglinear-feature-heavy-joint-4 | 0.106417 [0.105403, 0.133072] | 6.798 | 6.727 | 54.697 | 756.3 |
| loglinear-solver-stress-joint-1 | 0.743684 [0.698323, 0.769308] | 76.121 | 10.897 | 74.220 | 916.7 |
| loglinear-solver-stress-joint-4 | 1.885762 [1.856109, 1.926065] | 201.604 | 13.072 | 318.670 | 997.0 |

## Startup, prediction and memory disclosures

The raw report records preparation, compile plus first complete fit, first and warmed prediction, all raw update/fit samples, and process high-water RSS separately. Cold startup has one observation per fresh worker; it is descriptive rather than a compiler-only duration or confidence interval. Material disclosure thresholds are startup >2×, warmed prediction >1.5× and RSS >1.5× main.

| Case | Compile+first-fit ratio | Warm-prediction ratio | Peak-RSS ratio |
| --- | ---: | ---: | ---: |
| resnet-direct-one | 0.965 | 0.877 | 1.046 |
| resnet-direct-multi | 0.719 | 1.009 | 1.029 |
| resnet-mlpe-one | 1.109 | 0.994 | 1.005 |
| resnet-mlpe-multi | 0.742 | 0.985 | 1.021 |
| unet-direct-one | 1.011 | 0.983 | 0.987 |
| unet-direct-multi | 0.742 | 0.996 | 0.965 |
| unet-mlpe-one | 1.100 | 0.992 | 0.970 |
| unet-mlpe-multi | 0.716 | 1.012 | 0.939 |
| resnet-feature-heavy | 0.992 | 0.996 | 1.050 |
| unet-feature-heavy | 0.951 | 1.007 | 0.885 |
| resnet-solver-stress | 1.047 | 0.877 | 1.063 |
| unet-mlpe-stress | 1.071 | 0.987 | 1.009 |

CPU allocator statistics are unavailable; process RSS includes imports, compiler state, host objects and caches. The separate [large-registry report](large-registry-memory.md) covers actual differentiated executable allocations, synchronized live-array observations, full evaluation/finalization, persistent parameter/optimizer/output growth, compatible numeric-ID reuse, and legitimate ragged-shape specialization through 1,000 species. Neither executable allocation totals nor RSS is a measured complete-fit GPU/device peak.

## Scientific checks and preserved failures

Every workload passes post-timing checks of the returned state/history, exact requested pair order/cardinality, and complete MLPE calibration with positive nuisance quantities where applicable. Conductance distances pass an independent dense Laplacian reference. Full diagnostic matrices remain symmetric, nonnegative and zero-diagonal within frozen tolerances. Effective work manifests observe every update, encoding, group contribution and full evaluation; comparison rejects mismatched source/work/initialization or incomplete cases.

Scoped fixes fuse the unchanged one-group chain rule and cache multigroup encoder-VJP preparation. Public directional likelihood references, every Adam moment/update identity, exact continuation, species-ID compilation reuse, strict dropout chunk equivalence, LogLinear and conductance tests pass. Float32 rounding diagnostics and the rejected broader chunk fusion are retained in [scientific diagnostics](results/singleton-scientific/README.md).

The initial U-Net steady-step failure, the default chunk 1 marginal failure, and the unsuccessful chunk 4 diagnostic remain in [all attempts](results/runtime-attempts/README.md). The original main observations and fresh ten-repeat feature-heavy baseline both remain intact; the comparison records which two baseline cases were repeated. No unfavorable sample was deleted and no scientific workload, accuracy or threshold was reduced.

## Reproduction and final integration gate

Run against this committed source with the recorded dependencies and isolated CPU 0–1; raw output paths must not already exist:

```sh
PYTHONPATH=src python benchmarks/spec52/pr_driver.py --suite all --fit-repeats 10 --chunk 4 --output /tmp/spec52-pr-final.json
python benchmarks/spec52/compare.py --main benchmarks/spec52/results/main.json --main benchmarks/spec52/results/main-feature-repeat10.json --pr /tmp/spec52-pr-final.json --output /tmp/spec52-comparison.json

# Reproduce the archived comparison without timing or importing JAX.
python benchmarks/spec52/compare.py --main benchmarks/spec52/results/main.json --main benchmarks/spec52/results/main-feature-repeat10.json --pr benchmarks/spec52/results/pr-final-df7aea21.json --output /tmp/spec52-archived-comparison.json
```

The unchanged [final raw measurements](results/pr-final-df7aea21.json) retain their actual measured command and environment. The machine-readable [comparison](results/runtime-comparison.json) includes its own analysis/policy/raw-file hashes. Final integration must revalidate the complete source/driver identity and rerun implicated gates after performance-relevant production changes; reverify target main and explicitly refresh its baseline if it changes. Required gates remain merge acceptance conditions.
