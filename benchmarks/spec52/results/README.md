# Frozen main results

Source: `8439f99ea37f04273a1868b8009e1076239ad316`. Measured driver: `59631944a75a207ca3bd7c59ba5ab2ecd707ad00`.

Hardware: AMD EPYC 7742 64-Core Processor; CPU affinity [0, 1]; MemTotal:       528174084 kB. Python 3.13.13, JAX 0.10.2, jaxlib 0.10.2. Complete package versions and allocator/thread settings are in `main.json`.

All 12 declared cases passed finite-state/prediction checks. All six conductance cases also passed the independent dense graph pseudoinverse comparison; all six MLPE cases retained every calibration pair and satisfied the public calibration validity checks. These are fixed-budget, synthetic CPU observations, not convergence or accelerator-memory results.

Each row has three repeated fresh fits after one complete warm fit, and five warmed predictions. Training steps below use post-first updates, with whole-fit-block bootstrap uncertainty in `main-analysis.json`. Cold fits include compilation; repeated complete fits retain recurrent first-update overhead.

| Case | Cold fit (s) | Repeated fit median (s) | Steady step median (s) | Predict median (s) | Process peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| resnet-direct-one | 11.573 | 1.367 | 0.0242 | 0.9289 | 998.5 |
| resnet-direct-multi | 18.014 | 1.437 | 0.0382 | 1.7013 | 1024.9 |
| resnet-mlpe-one | 13.570 | 1.449 | 0.0280 | 0.8751 | 1152.0 |
| resnet-mlpe-multi | 21.584 | 1.487 | 0.0452 | 1.7484 | 1249.2 |
| unet-direct-one | 11.518 | 1.770 | 0.0135 | 0.0308 | 1096.3 |
| unet-direct-multi | 18.094 | 1.759 | 0.0132 | 0.0615 | 1147.4 |
| unet-mlpe-one | 14.010 | 1.754 | 0.0112 | 0.0319 | 1265.2 |
| unet-mlpe-multi | 23.147 | 1.816 | 0.0113 | 0.0631 | 1255.1 |
| resnet-feature-heavy | 15.178 | 2.651 | 0.3616 | 1.3152 | 1073.4 |
| unet-feature-heavy | 12.058 | 1.668 | 0.0146 | 0.0322 | 1105.2 |
| resnet-solver-stress | 17.275 | 2.273 | 0.2039 | 1.0020 | 1251.3 |
| unet-mlpe-stress | 18.089 | 2.578 | 0.0161 | 0.1416 | 1336.6 |

The representative gates use the feature-heavy rows. Their initial uncertainty is:

- resnet-feature-heavy: complete-fit range 2.643–2.655 s, 95% bootstrap median interval 2.643–2.655 s; steady-step interval 0.3606–0.3617 s; recurrent first-update median 1.579 s.
- unet-feature-heavy: complete-fit range 1.639–1.752 s, 95% bootstrap median interval 1.639–1.752 s; steady-step interval 0.0077–0.0155 s; recurrent first-update median 1.566 s.

Only three fit blocks are available initially. These intervals are descriptive; apply the frozen ten-repeat escalation rule before interpreting a borderline PR ratio. A first-update compilation cause is inferred from timing/code, not established with compiler tracing. The raw report retains all first-update samples and all original hashes; the separate analysis file names its exact raw input, code and policy hashes.

RSS is the process high-water mark through timed workloads, including imports, compilation and allocator caches. The separately recorded post-validation RSS may be higher. CPU device allocator statistics are null; this evidence makes no claim about GPU allocation or the memory scaling of the new multispecies backward pass.

Exact command used from the isolated baseline worktree:

```sh
PYTHONPATH=src /home/boussang/ilg-toolkit/.venv/bin/python \
  benchmarks/spec52/main_driver.py --output benchmarks/spec52/results/main.json
```

The source guard verified the imported `src` tree against frozen main. A shared installed dependency environment was used, with the baseline worktree selected by `PYTHONPATH`; production sources were unchanged.

## Native comparison

The original observations above remain unchanged. The final native comparison
uses [main-feature-repeat10.json](../README.md#release-evidence) (archive member `results/main-feature-repeat10.json`) for the two
predeclared feature-heavy gates and retains this original baseline for the other
ten workloads. The repeat increases fit observations without changing the
production source, fixture, scientific work or thresholds.

See the [runtime report](../runtime-results-final-b2088e6.md), [final PR raw data](../README.md#release-evidence) (archive member `results/pr-final-b2088e6.json`),
and [machine-readable comparison](../README.md#release-evidence) (archive member `results/runtime-comparison-final-b2088e6.json`). All earlier
[runtime attempts](runtime-attempts/README.md) remain preserved, including failed
gates. The separate [large-registry report](../large-registry-memory-final-b2088e6.md) documents
the completed 14-case CPU memory and compilation matrix.
