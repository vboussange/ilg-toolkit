# Bounded joint training: CPU execution and allocation evidence

Eight fresh-process public MLPE fits compare U-Net and ResNet registries of 4
and 16 species, at species chunk sizes 1 and 4. Each uses one 32×32×2 raster,
eight sampling units, all 28 pairs per species, one Adam update, and full initial
and final evaluations. Source commit: `7b60e4f67d81dbbbab7760149653838ec20aa667`.
The raw JSONs preserve the source-file hashes, deterministic fixture hash,
installed versions, affinity (CPUs 2–3), CPU device, persistent array sizes,
synchronized public-fit operation counts and actual XLA allocation reports.

Every case observed exactly **three logical regional encodings** (initial
inference, training, final inference), **one optimizer update**, and at most the
configured chunk size in a completed forward/backward species-tail call. A
chunk's backward call finishes before the next chunk starts. ResNet retains its
existing patch checkpoint policy; no additional outer checkpoint is installed.
The encoder parameter tree is absent from tail differentiation. Full evaluation
streams the same bounded tails, including complete MLPE posterior output.

The following are CPU compiler buffer allocations in bytes, including arguments,
outputs and temporary buffers within each executable. They are **not a measured
end-to-end or GPU memory peak**. In particular, regional encoder residuals remain
live while tails run: neither the maximum nor the sum of these per-executable
numbers is an observed total peak. Process RSS in the raw files additionally
includes imports, compilation caches and persistent inputs/outputs.

| Family | Registry | Chunk | Tail forward/backward | Tail evaluation/posterior |
| --- | ---: | ---: | ---: | ---: |
| U-Net | 4 | 1 | 52,477 | 25,180 |
| U-Net | 16 | 1 | 55,197 | 26,092 |
| U-Net | 4 | 4 | 72,013 | 39,569 |
| U-Net | 16 | 4 | 73,837 | 40,481 |
| ResNet | 4 | 1 | 251,235 | 176,141 |
| ResNet | 16 | 1 | 276,291 | 188,669 |
| ResNet | 4 | 4 | 575,170 | 401,794 |
| ResNet | 16 | 4 | 600,226 | 414,322 |

Across all registry/chunk cases, U-Net's encoder training forward (with residual
outputs) allocated 1,845,256 bytes and its backward executable 2,006,571 bytes.
ResNet's corresponding numbers were 6,077,820 and 9,437,624 bytes. Thus encoder
allocations stayed constant as species count increased. Tail allocation growth
with registry size includes the ordinary full head/nuisance gradient and parameter
arrays; the heavy surfaces, terminal solves and likelihood intermediates are
limited by the chunk. The embedded reports retain the actual tensor shapes.

Persistent output must grow: completed MLPE posterior arrays increased from
4,352 bytes for four groups to 17,408 bytes for sixteen. Model, optimizer and
nuisance arrays also grow with registry/group count and are reported separately.
No constant-total-memory claim, GPU allocator claim, or runtime acceptance-gate
claim is made here. Frozen-main timing comparisons remain a separate experiment.

## Reproduce

Run each `(family, species, chunk)` combination in a fresh process. The matrix is
`family={unet,resnet}`, `species={4,16}`, `chunk={1,4}`. For example, from the repo:

```sh
XLA_FLAGS='--xla_dump_to=/tmp/ilg-bounded/unet-s4-c1 --xla_dump_hlo_as_text' \
JAX_ENABLE_X64=true JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
PYTHONPATH=src taskset -c 2,3 python benchmarks/spec52/bounded_probe.py \
  --family unet --species 4 --chunk 1 \
  --xla-directory /tmp/ilg-bounded/unet-s4-c1 --output /tmp/unet-s4-c1.json
```

Use a fresh empty dump directory for every case. GPU/other hardware needs its
own allocator measurements and may have different compiler allocation behavior.

The original probe's numeric `total_allocated_bytes` field is null because its
parser expected `Total bytes used:` and this XLA emits `Total bytes:`. The full
raw text and measured hashes remain unchanged. The driver now accepts both
forms and rejects absent totals. The separate, hashed analysis parses the
preserved raw reports; it does not relabel or rerun numerical measurements:

```sh
python benchmarks/spec52/analyze_bounded_probe.py \
  benchmarks/spec52/results/bounded-memory \
  --output benchmarks/spec52/results/bounded-memory/analysis.json
```

`analysis.json` includes each raw file's SHA-256 and the analysis script's SHA-256.
The three encoder report roles follow actual compilation order in the fit:
initial inference, differentiated training forward, encoder pullback.
