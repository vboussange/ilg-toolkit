# Large-registry memory and compilation evidence

**Complete: all 14 public-fit cases passed** on clean measured commit
`df7aea21ec55ce39bf34d78e53668ab8d4c86f44`, production source tree
`8d92e023e4d4b29beca50e0c7028572ef7dd78ba`.
The [raw matrix and machine-readable audit](results/large-registry/final-df7aea21/analysis.json)
cover both neural families at registries of 4, 100 and 1,000 species,
chunks of 1 and 4, plus one ragged case per family. Every case used a fresh
CPU process and XLA dump directory, affinity 2–3, one BLAS/OpenMP thread,
JAX x64, and disabled persistent compilation caching. This instrumented
matrix is separate from the [runtime protocol](README.md).

The fixture preserves the earlier [4/16-species probe](bounded-memory.md):
one 32×32×2 raster, eight sampling units, all 28 observed pairs per group,
the same MLPE settings and seeds, and four selected species per update.
Final schema v2 uses two updates and `evaluation_interval=2`, with complete
initial and final evaluations and completed calibrations for every species.
This memory-only cadence explicitly differs from the frozen three-update,
four-evaluation runtime workload. Both ragged cases use registry 4/chunk 1
and six units/15 pairs for the last species, with the other groups unchanged.

## Executed allocation accounting

These are CPU XLA per-executable allocation totals, including arguments,
outputs and temporary buffers. Encoder inference and backward totals are
constant across registry sizes; compiled forward preparation also passes
through the projection arrays, whose parameter storage grows with registry:

| Family | Registry | Inference bytes | Differentiated forward/residual bytes | Backward bytes |
| --- | ---: | ---: | ---: | ---: |
| ResNet | 4 | 6,061,428 | 9,049,996 | 9,437,624 |
| ResNet | 100 | 6,061,428 | 9,247,372 | 9,437,624 |
| ResNet | 1,000 | 6,061,428 | 11,097,772 | 9,437,624 |
| U-Net | 4 | 147,336 | 1,889,912 | 2,006,571 |
| U-Net | 100 | 147,336 | 1,901,432 | 2,006,571 |
| U-Net | 1,000 | 147,336 | 2,009,432 | 2,006,571 |

The forward evidence comes from the actual compiled encoder derivative
preparation executed during public fitting. Its residuals remain live while
the bounded tail forward/backward calls run; one encoder pullback follows
the reduced tail gradients. No per-species backbone gradient tree is retained.
Forward-total growth is exactly twice the projection-byte growth in the
persistent table below; it does not represent additional raster activations.

| Family | Registry | Chunk | Tail forward/backward bytes | Evaluation/posterior bytes |
| --- | ---: | ---: | ---: | ---: |
| ResNet | 4 | 1 | 251,235 | 176,141 |
| ResNet | 4 | 4 | 575,170 | 401,794 |
| ResNet | 100 | 1 | 526,185 | 276,365 |
| ResNet | 100 | 4 | 775,624 | 502,018 |
| ResNet | 1,000 | 1 | 3,344,969 | 1,215,965 |
| ResNet | 1,000 | 4 | 3,348,520 | 1,441,618 |
| U-Net | 4 | 1 | 52,477 | 25,180 |
| U-Net | 4 | 4 | 72,013 | 39,569 |
| U-Net | 100 | 1 | 74,375 | 32,476 |
| U-Net | 100 | 4 | 86,615 | 46,865 |
| U-Net | 1,000 | 1 | 279,591 | 100,876 |
| U-Net | 1,000 | 4 | 283,127 | 115,265 |

The embedded reports and entry layouts show that raster, terminal, adjoint
and MLPE workspace dimensions follow the bounded chunk and group shape.
At registries 100 and 1,000, the only registry-sized numerical shapes are
projection weights/biases, the two nuisance values per group, and their
corresponding gradients or finite-check masks. U-Net projections have
shapes `(S, 3, 4)`/`(S, 3)`; ResNet uses `(S, 256)`/`(S,)`; nuisance
arrays have shape `(S, 2)`. No registry-sized heavy surface, embedding,
resistance or MLPE intermediate appears. Legitimate parameter-sized
arguments, outputs and workspace explain why tail allocation totals can
grow even when the heavy scientific computation is bounded.

## Persistent storage and observed process memory

Logical persistent categories grow with registry size. These values are
identical between chunks for a given family/registry. The pooled backbone
is 2,963,712 bytes for ResNet and 45,552 bytes for this U-Net. The table
reports latest model/projection/nuisance sizes; selected model and nuisance
trees have the same logical sizes, and raw files record them separately.

| Family | Registry | Model bytes | Projection bytes | Adam bytes | Nuisance bytes | Calibration arrays bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet | 4 | 2,967,824 | 4,112 | 5,935,780 | 64 | 4,352 |
| ResNet | 100 | 3,066,512 | 102,800 | 6,136,228 | 1,600 | 108,800 |
| ResNet | 1,000 | 3,991,712 | 1,028,000 | 8,015,428 | 16,000 | 1,088,000 |
| U-Net | 4 | 45,792 | 240 | 91,716 | 64 | 4,352 |
| U-Net | 100 | 51,552 | 6,000 | 106,308 | 1,600 | 108,800 |
| U-Net | 1,000 | 105,552 | 60,000 | 243,108 | 16,000 | 1,088,000 |

Inputs, initial model retained by the driver, latest and selected state,
optimizer moments, nuisance rows, and completed calibration arrays are
separate logical categories in the raw records. They are not additive
allocations: buffers can be shared or aliased. Latest and selected step
numbers are recorded explicitly; public numerical checks use the selected
inference model.

| Family | Registry | Chunk | Peak process RSS (MiB) | Largest observed boundary span (MiB) |
| --- | ---: | ---: | ---: | ---: |
| ResNet | 4 | 1 | 893.7 | 22.673 |
| ResNet | 4 | 4 | 892.8 | 22.673 |
| ResNet | 100 | 1 | 906.2 | 23.700 |
| ResNet | 100 | 4 | 932.6 | 23.700 |
| ResNet | 1,000 | 1 | 1027.8 | 33.327 |
| ResNet | 1,000 | 4 | 1053.7 | 33.327 |
| U-Net | 4 | 1 | 870.6 | 1.998 |
| U-Net | 4 | 4 | 884.9 | 1.982 |
| U-Net | 100 | 1 | 872.1 | 2.315 |
| U-Net | 100 | 4 | 897.3 | 2.292 |
| U-Net | 1,000 | 1 | 1042.6 | 5.327 |
| U-Net | 1,000 | 4 | 1041.2 | 5.193 |

Boundary spans are synchronized live-array logical extents, deduplicated
only by identical start addresses. They omit internal executable transients;
distinct overlapping views can still double-count bytes. Process high-water
RSS includes host objects, compiler state, inputs, outputs and caches.
The CPU backend returns `null` for allocator memory statistics in every
snapshot. Neither a maximum nor a sum of executable allocation totals is
a measured complete-fit device peak, and no GPU peak is claimed.

## Compilation reuse, lifetime and scientific checks

Every case observed two inference encodes, two differentiated encoder
forwards, two encoder backwards and exactly two Adam updates. Each update
included exactly four distinct selected groups. Chunk 1 completed eight
training tails; chunk 4 completed two. Full evaluations completed `2S/C`
tails in the regular cases. Calls completed sequentially before the next
checked call, including each tail backward; only reduced gradients and
the common encoder residuals survive between training chunks.

The 100-species selections were `[20, 28, 72, 88]` then `[11, 31, 62, 97]`.
The 1,000-species selections were `[185, 406, 600, 742]` then
`[137, 410, 531, 732]`. Both families and chunks reused exactly one
gradient executable across these changing numeric IDs; complete evaluation
likewise reused one executable across all registry IDs. The ragged cases
produced exactly two specializations per tail role for the two legitimate
unit/pair/terminal shapes (eight-unit / six-unit totals):

| Family | Ragged gradient allocations (bytes) | Ragged evaluation allocations (bytes) |
| --- | ---: | ---: |
| ResNet | 251,235 / 235,539 | 176,141 / 157,421 |
| U-Net | 52,477 / 51,949 | 25,180 / 22,396 |

Every group received a complete calibration for the selected model. Public raw
distances and genetic predictions preserve requested species keys and pair
order; selecting a species alone gives identical values, and changing
unselected projection rows leaves requested outputs unchanged. U-Net passes
the independent squared-distance scaling reference. ResNet passes the
explicit NumPy dense graph Laplacian pseudoinverse reference at unchanged
`rtol=2e-5`, `atol=2e-6`; maximum observed absolute error is `6.5329062e-07`.
These post-fit checks do not change the recorded fit allocation stream.

## Repeat and audit

Commit source and driver first. Each case requires a fresh process and
empty XLA dump directory. Run memory probes separately from timing gates.
The full matrix is both families × registries `{4,100,1000}` × chunks
`{1,4}`, plus `--ragged` for each family at registry 4/chunk 1:

```sh
XLA_FLAGS='--xla_dump_to=/tmp/ilg-large/resnet-s1000-c1 --xla_dump_hlo_as_text' JAX_ENABLE_X64=true JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src taskset -c 2,3 python benchmarks/spec52/large_registry_probe.py --family resnet --species 1000 --chunk 1 --xla-directory /tmp/ilg-large/resnet-s1000-c1 --output /tmp/ilg-large/results/resnet-s1000-c1.json
python benchmarks/spec52/large_registry_analysis.py /tmp/ilg-large/results --output /tmp/ilg-large/results/analysis.json

# Recheck the archived final matrix without running JAX.
python benchmarks/spec52/large_registry_analysis.py benchmarks/spec52/results/large-registry/final-df7aea21 --output /tmp/ilg-large-final-analysis.json
```

The analyzer rejects missing cases, mixed production sources/drivers,
incomplete evaluations/calibrations, incorrect update counts, chunk-bound
violations, overlapping calls, missing allocation evidence, inconsistent
compiler totals and extra numeric-ID-specific compilation. Its output
records raw-file and analyzer hashes. An independent audit also rebuilt
every source/driver hash from Git, every seeded fixture/target hash, and
every embedded allocation-report hash for all 14 cases.
The coordinator separately verified all 74 original report/HLO file hashes
and inspected the full optimized HLO for registry sizes 100 and 1,000;
registry-sized axes occurred only in parameter/nuisance arrays and their
derivative or finite-check work, with no heavy all-species intermediate.

All seven preliminary schema-v1 one-update reports and their derived
[analysis](results/large-registry/preliminary-d7e410b/analysis.json) remain
byte-identical in their archive. They are historical diagnostics and are
not mixed into the final matrix. No final case failed or was replaced.
The complete matrix establishes the bounded execution and compilation
behavior for these CPU fixtures; unsupported allocator profiling remains
an explicit hardware limitation, not a measured GPU-memory result.
