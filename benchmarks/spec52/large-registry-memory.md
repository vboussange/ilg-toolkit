# Large-registry memory and compilation evidence

The reproducible probe extends the earlier [4/16-species evidence](bounded-memory.md)
to registries of 4, 100 and 1,000 species, with exactly four selected training
groups and chunks of 1 or 4. Both neural families retain the same 32×32×2 raster,
eight sampling units, all 28 observed pairs, MLPE policy, and seeds as that probe.
One public optimizer update is surrounded by complete initial and final evaluations:
all registered observed species receive completed calibrations. Two separate ragged
cases use four species and six units/15 pairs for the last species, keeping the
other three groups at eight units/28 pairs.

**Status: preliminary diagnostic evidence.** The complete 14-case matrix must be
run on the agreed final runtime candidate. Current measurements are retained in
`results/large-registry/preliminary-d7e410b/`; they do not replace the final matrix
or imply that the separate runtime gates pass. Measured source/driver commit:
`d7e410b055ec2312660485f75b5d82c146cfc73e`.

| Family | Registry | Chunk | Tail forward/backward bytes | Tail evaluation/posterior bytes |
| --- | ---: | ---: | ---: | ---: |
| U-Net | 4 | 1 | 52,477 | 25,180 |
| U-Net | 4 | 4 | 72,013 | 39,569 |
| U-Net | 100 | 1 | 74,375 | 32,476 |
| U-Net | 100 | 4 | 86,615 | 46,865 |
| ResNet | 1,000 | 1 | 3,344,969 | 1,215,965 |

These are actual executed CPU XLA compiled allocation totals, including arguments,
outputs and temporary buffers. They are not measured complete-fit device peaks.
Encoder residuals remain live while tail calls execute, so neither a maximum nor
a sum of executable totals is a complete-fit peak. Synchronized live-array snapshots
add logical byte spans at public-fit phase boundaries, deduplicated only by matching
start addresses; they do not see internal executable transients. CPU allocator
`memory_stats()` is unsupported (`null`). Process peak RSS includes host objects,
compiler caches, inputs and persistent output. No GPU measurement is claimed.

The full registry legitimately adds parameters and Adam moments, two nuisance
values per observed group, retained selected-state data, and complete MLPE posterior
output. The driver records those categories separately without treating their logical
sizes as an additive allocation total: initial/latest/selected trees can share buffers.
For example, ResNet with 1,000 species has 1,028,000 bytes of projection parameters,
16,000 bytes of latest nuisance values, and 1,088,000 bytes of completed calibration
arrays. Its shared backbone is 2,963,712 bytes; its optimizer is 8,015,428 bytes.
Full head/nuisance arrays also occur among differentiated executable arguments,
outputs and parameter-sized workspace. Therefore tail total bytes need not be
constant in registry size even when species-heavy work is bounded by the chunk.
The preserved report shapes and HLO entry layouts expose this distinction.

The numerical execution observer wraps only the existing synchronized checked-call
boundary. It does not replace numerical kernels, change gradients, alter selection,
or relax the solver. Every completed tail event records numeric species/group IDs,
payload shapes, chronological boundaries, and newly emitted executable reports.
The audit requires full initial/final group coverage, exactly four distinct selected
training groups, chunk bounds, sequential completion, the shared encoder forward and
backward, one update, and all completed calibrations. Compatible numeric IDs reuse
one executable per shape; the separate ragged cases must record two shape-driven
specializations per tail role. Different registries may specialize parameter shapes.

Selected public raw/genetic predictions verify exact requested species keys and
pair order, selected-alone equality, and isolation from modified unselected projection
rows. An independent squared-distance scaling reference checks U-Net; an explicit
NumPy dense graph Laplacian pseudoinverse checks ResNet. These checks occur after
the measured fit stream and do not alter the fit or compiler allocation reports.

## Repeat and audit

Commit the driver and production source first. The probe checks each recorded
file hash against that measured Git commit and refuses dirty tracked source. Use
one fresh process and empty dump directory per case; do not run these instrumented
probes concurrently with runtime gate measurements. The full matrix is both families
× registries `{4,100,1000}` × chunks `{1,4}`, plus `--ragged` for each family at
registry 4/chunk 1. For example:

```sh
XLA_FLAGS='--xla_dump_to=/tmp/ilg-large/resnet-s1000-c1 --xla_dump_hlo_as_text' \
JAX_ENABLE_X64=true JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src taskset -c 2,3 python \
  benchmarks/spec52/large_registry_probe.py \
  --family resnet --species 1000 --chunk 1 \
  --xla-directory /tmp/ilg-large/resnet-s1000-c1 \
  --output benchmarks/spec52/results/large-registry/resnet-s1000-c1.json

python benchmarks/spec52/large_registry_analysis.py \
  benchmarks/spec52/results/large-registry \
  --output benchmarks/spec52/results/large-registry/analysis.json
```

The analysis refuses a missing matrix case, incomplete coverage/calibrations,
extra numeric-ID-specific compilation, overlapping checked calls, a violated chunk
bound, or a mismatch between parsed XLA totals and recorded fields. The explicit
`--allow-partial` option is only for diagnostic subsets and labels the missing
cases. Analysis metadata hashes each raw artifact and its own script. Historical
#58 raw files and the preliminary #63 measurements remain unchanged.
