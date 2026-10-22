# Spec #52 runtime and memory results

The measurements cover toolkit commit
`b2088e6c390e943ce90f2d14e678358c9f33afa0`, source tree
`9d8c31b784f4d716bc00898838220382bc2b5da7`, against single-species baseline
`8439f99ea37f04273a1868b8009e1076239ad316`. This branch retains that toolkit source.
[The README](README.md) gives commands to generate new measurements with the
same workloads and numerical settings. These tables summarize the recorded
measurements; elapsed times depend on hardware, dependencies and system load.

## Runtime

All 32 candidate workloads completed with ten synchronized public-fit repetitions
and species microbatch size 4. Features and neural parameters use float32;
resistance and maximum-likelihood population-effects (MLPE) calculations use
float64. Workers use central processing unit (CPU) cores 0–1, one numerical-library
thread and disabled persistent compilation cache. Workers ran on a shared machine.

All ten required comparisons passed. Ratios below compare candidate to baseline;
brackets show 95% bootstrap intervals from 10,000 separately sampled fit blocks.
Steady-update intervals retain updates 2 and 3 together within each fit.

| Feature-heavy case | Public fit ratio [95% interval] | Steady-update ratio [95% interval] |
| --- | ---: | ---: |
| ResNet | 0.5421115 [0.5357457, 0.5456540] | 1.0146009 [1.0074297, 1.0200617] |
| U-Net | 0.0429303 [0.0415549, 0.0434509] | 0.6221370 [0.4232513, 0.8254119] |

The upper interval must be at most 1.10. Each added-species cost below is divided
by the corresponding species' separate fit cost, with an upper limit of 0.50.

| Feature-heavy case | Added species | Increment ratio [95% interval] |
| --- | ---: | ---: |
| ResNet | 1→2 | 0.0230226 [0.0121142, 0.0331692] |
| ResNet | 2→3 | 0.0156971 [0.0077732, 0.0261128] |
| ResNet | 3→4 | 0.0203629 [0.0127401, 0.0351966] |
| U-Net | 1→2 | 0.4564803 [0.4454445, 0.4769167] |
| U-Net | 2→3 | 0.2145529 [0.1951427, 0.2294081] |
| U-Net | 3→4 | 0.2281815 [0.2127888, 0.2370833] |

The additional `resnet-direct-multi` steady-update comparison is inconclusive:
0.889809 [0.724501, 1.146589] crosses 1.10. It is outside the required comparisons.
Startup, warmed prediction and process resident set size (RSS) ratios stay below
the disclosure limits of 2.0, 1.5 and 1.5, respectively.

LogLinear has its own measurements. Memory is reported in mebibytes (MiB).

Note: LogLinear has no baseline counterpart.

| Workload | Fit median (s) | Steady-update median (s) | Compile + first fit (s) | Warm prediction median (s) | Peak RSS (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Feature-heavy, 1 species | 0.037458 | 0.003255 | 5.777078 | 0.016338 | 709.74 |
| Feature-heavy, 4 species | 0.124602 | 0.012127 | 6.968215 | 0.055007 | 717.90 |
| Solver-stress, 1 species | 0.701214 | 0.076597 | 11.191703 | 0.081067 | 913.69 |
| Solver-stress, 4 species | 1.870770 | 0.194101 | 13.417640 | 0.316247 | 929.84 |

Public-fit ratios include evaluation, calibration and the baseline's recurring
first-update compilation. They describe three-update fits. Compile-plus-first-fit
timing includes compilation and execution; warmed fits measure recurring work.

Note: these ratios do not estimate long-training or convergence speedups.

## Memory and compilation

All 14 cases passed: two neural families × species counts {4, 100, 1,000} ×
microbatch sizes {1, 4}, plus two unequal-sampling cases. Each fit takes two
updates with four selected species per update, and evaluates and calibrates every
species initially and finally. Changing selected species reuses compiled
operations; unequal sampling-unit counts require two shape specializations.
Process memory is reported in gibibytes (GiB).

| Family | Species | Microbatch | Species forward/backward compiled allocation (bytes) | Process peak RSS (GiB) |
| --- | ---: | ---: | ---: | ---: |
| ResNet | 4 | 1 | 251,235 | 0.832 |
| ResNet | 4 | 4 | 575,170 | 0.848 |
| ResNet | 100 | 1 | 526,185 | 0.861 |
| ResNet | 100 | 4 | 775,624 | 0.883 |
| ResNet | 1,000 | 1 | 3,344,969 | 1.013 |
| ResNet | 1,000 | 4 | 3,348,520 | 0.968 |
| U-Net | 4 | 1 | 52,477 | 0.848 |
| U-Net | 4 | 4 | 72,013 | 0.883 |
| U-Net | 100 | 1 | 74,375 | 0.871 |
| U-Net | 100 | 4 | 86,615 | 0.860 |
| U-Net | 1,000 | 1 | 279,591 | 0.991 |
| U-Net | 1,000 | 4 | 283,127 | 1.009 |

Raster, resistance and MLPE workspace sizes follow the species microbatch and
sampling-unit counts. Persistent projections, optimizer state, MLPE variance
parameters and calibration arrays grow with species count. Prediction checks
cover species selection, pair order, squared embedding distance and resistance
against a dense graph reference.

Compiled allocations measure individual operations. Pooled-backbone intermediate
values remain live during species calculations. Live-array snapshots count arrays
at observation boundaries. RSS includes compiler, Python and output storage.

Note: the CPU backend supplies no device-allocator peak; these observations do
not establish graphics processing unit (GPU) memory scaling. Live-array snapshots
omit internal temporary storage.
