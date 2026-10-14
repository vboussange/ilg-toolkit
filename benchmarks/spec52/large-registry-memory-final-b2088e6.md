# Final source large-registry memory evidence

The complete 14-case matrix was rerun on commit `b2088e6c390e943ce90f2d14e678358c9f33afa0`, production source tree `9d8c31b784f4d716bc00898838220382bc2b5da7`, after the private shared solver-context helper extraction. All cases passed the unchanged [analyzer](large_registry_analysis.py); [raw reports and analysis](results/large-registry/final-b2088e6/analysis.json) preserve source, driver, fixture, allocation and compilation identities.

The [original report](large-registry-memory.md) and its original 14 raw reports remain historical evidence. This rerun uses the same two-update public-fit protocol, registries 4/100/1,000, chunks 1/4, both neural families and two ragged cases. Initial/final evaluations cover every species, changing numeric training IDs reuse the bounded executables, and public prediction and calibration checks pass, and persistent-state bytes are accounted for. Disk artifact persistence is covered separately by the persistence tests; this probe does not save or reload artifacts.

| Family | Registry | Chunk | Tail forward/backward executable allocation (bytes) | Process peak RSS (GiB) |
| --- | ---: | ---: | ---: | ---: |
| resnet | 100 | 1 | 526,185 | 0.861 |
| resnet | 100 | 4 | 775,624 | 0.883 |
| resnet | 1000 | 1 | 3,344,969 | 1.013 |
| resnet | 1000 | 4 | 3,348,520 | 0.968 |
| resnet | 4 | 1 | 251,235 | 0.832 |
| resnet | 4 | 4 | 575,170 | 0.848 |
| unet | 100 | 1 | 74,375 | 0.871 |
| unet | 100 | 4 | 86,615 | 0.860 |
| unet | 1000 | 1 | 279,591 | 0.991 |
| unet | 1000 | 4 | 283,127 | 1.009 |
| unet | 4 | 1 | 52,477 | 0.848 |
| unet | 4 | 4 | 72,013 | 0.883 |

The coordinator independently rebuilt 392 source-file identities and 4,424 fixture-group identities, verified all 74 allocation-report/HLO file hashes, and audited registry-sized axes. Original and final numerical results, executable layouts and compiled allocation totals match exactly. The [source/fixture audit](results/final-review-refactor/coordinator65-review-fix-memory-audit.json) and [dump audit](results/final-review-refactor/coordinator65-review-fix-dumps-audit.json) retain the checks.

CPU compiled allocations are per executable, not a whole-fit device-memory peak. Encoder residuals remain live while bounded tails run; live-array snapshots omit executable transients. RSS includes compiler, host and output storage. These measurements do not establish a GPU peak or allocator scaling result.
