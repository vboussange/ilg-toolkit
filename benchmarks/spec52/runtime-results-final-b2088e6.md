# Final source runtime acceptance

Status: **PASS**. All 32 workloads completed with ten synchronized full-fit repetitions and chunk 4 on clean commit `b2088e6c390e943ce90f2d14e678358c9f33afa0`, production source tree `9d8c31b784f4d716bc00898838220382bc2b5da7`. All ten required gates pass the unchanged frozen protocol and bootstrap policy. The frozen main target remains `8439f99ea37f04273a1868b8009e1076239ad316`.

[Raw final measurements](results/pr-final-b2088e6.json), [official comparison](results/runtime-comparison-final-b2088e6.json), [independent recomputation audit](results/runtime-audit-final-b2088e6.json) and [wrapper identity](results/runtime-wrapper-final-b2088e6.json) preserve exact source, driver, work, inputs and initialization evidence. Raw SHA256: `aabe422a5847e5a802fbb9d4ae0ad52c9d5c29105aff421e4c4dc8c493414640`.

## Required gates

Ratios compare native / main. Intervals are seeded 95% independent bootstrap intervals over complete fit blocks (10,000 draws); updates 2/3 remain together within each block. A gate passes only when the entire interval is below its limit.

| Family | Complete fit ratio [95% interval] | Steady update ratio [95% interval] |
| --- | ---: | ---: |
| resnet-feature-heavy | 0.5421115 [0.5357457, 0.5456540] | 1.0146009 [1.0074297, 1.0200617] |
| unet-feature-heavy | 0.0429303 [0.0415549, 0.0434509] | 0.6221370 [0.4232513, 0.8254119] |

Both parity limits are 1.10. Each adjacent species increment below is normalized by the matching added independent fit; its limit is 0.50.

| Family | Added species | Increment ratio [95% interval] |
| --- | ---: | ---: |
| resnet-feature-heavy | 1→2 | 0.0230226 [0.0121142, 0.0331692] |
| resnet-feature-heavy | 2→3 | 0.0156971 [0.0077732, 0.0261128] |
| resnet-feature-heavy | 3→4 | 0.0203629 [0.0127401, 0.0351966] |
| unet-feature-heavy | 1→2 | 0.4564803 [0.4454445, 0.4769167] |
| unet-feature-heavy | 2→3 | 0.2145529 [0.1951427, 0.2294081] |
| unet-feature-heavy | 3→4 | 0.2281815 [0.2127888, 0.2370833] |

The non-gated `resnet-direct-multi` steady-update diagnostic remains **inconclusive**: ratio 0.8898089978, 95% interval [0.7245005466, 1.1465887366], spanning the 1.10 diagnostic limit. This does not change the ten required gate outcomes.

## LogLinear and disclosures

LogLinear has no legacy main analogue and is reported separately, without a fabricated main ratio.

| Workload | Complete fit median (s) | Steady update median (s) | Compile + first fit (s) | Warm prediction median (s) | Peak RSS (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| loglinear-feature-heavy-joint-1 | 0.037458 | 0.003255 | 5.777078 | 0.016338 | 709.74 |
| loglinear-feature-heavy-joint-4 | 0.124602 | 0.012127 | 6.968215 | 0.055007 | 717.90 |
| loglinear-solver-stress-joint-1 | 0.701214 | 0.076597 | 11.191703 | 0.081067 | 913.69 |
| loglinear-solver-stress-joint-4 | 1.870770 | 0.194101 | 13.417640 | 0.316247 | 929.84 |

No startup, warm-prediction or RSS disclosure crosses the frozen material-regression policy. The comparison retains every measured disclosure and LogLinear interval. Cold compile/first-fit and first-prediction times include compilation; warmed complete fits and updates measure recurrent work. They are separate quantities and should not be interpreted as interchangeable speedups. The three-update complete-fit gains include removal of main’s recurring first-update compilation. They do not estimate long-training or convergence speedups. CPU RSS includes compiler/host storage and is not a GPU device peak.

Workers use CPU affinity 0–1, x64 CPU arithmetic, float32 neural parameters/features, float64 solver/MLPE arithmetic, one native thread and no persistent compilation cache. Project JAX workers were stopped before timing. This was a shared host: an unrelated JAX process remained active with unrestricted affinity and was left untouched. The [10-second CPU preflight](results/final-review-refactor/runtime65-host-preflight.json) recorded cores 0/1 at 3.85%/2.90% busy and sibling cores 64/65 at 5.73%/4.22%; no exclusive host reservation or whole-host idleness is claimed.

The [original report](runtime-results.md) retains its measurement body and now adds a historical-source banner. Original 32-case raw data, earlier unsuccessful attempts, frozen main data and the ten-fit main feature repeat remain byte-identical. Final-source rows are not mixed with historical native rows. The [compiler equivalence audit](results/final-review-refactor/README.md) supports the private extraction's numerical equivalence; this fresh matrix supplies its actual performance evidence. The [final memory report](large-registry-memory-final-b2088e6.md) records the separate 14-case rerun.
