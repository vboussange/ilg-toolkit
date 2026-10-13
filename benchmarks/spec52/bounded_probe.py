"""Record actual public-fit CPU compiler allocation reports and execution bounds.

Run each case in a fresh process with XLA_FLAGS set before importing JAX::

  XLA_FLAGS='--xla_dump_to=/tmp/ilg-bounded/unet-s4-c1 --xla_dump_hlo_as_text' \\
    JAX_ENABLE_X64=true JAX_PLATFORMS=cpu PYTHONPATH=src \\
    python benchmarks/spec52/bounded_probe.py --family unet --species 4 --chunk 1 \\
    --xla-directory /tmp/ilg-bounded/unet-s4-c1 --output /tmp/unet-s4-c1.json

This is memory/operation-count evidence, not a warmed performance comparison.
XLA reports are CPU compiled buffer allocations, not measured GPU allocator peaks.
Process peak RSS includes imports, compiler caches, inputs and persistent outputs.
"""

import argparse
import hashlib
import json
import os
import platform
import re
import resource
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("unet", "resnet"), required=True)
    parser.add_argument("--species", type=int, required=True)
    parser.add_argument("--chunk", type=int, required=True)
    parser.add_argument("--xla-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.species < 1 or args.chunk < 1:
        parser.error("species and chunk must be positive")
    if "--xla_dump_to=" not in os.environ.get("XLA_FLAGS", ""):
        parser.error("Set XLA_FLAGS to dump actual executed compilations before launching Python")

    import coordax as cx
    import equinox as eqx
    import jax
    import numpy as np

    from ilg_toolkit import (
        PairwiseObservations,
        RegionBatch,
        SamplingUnits,
        TargetSpec,
        TrainingConfig,
        fit,
        training,
    )
    from ilg_toolkit import _training_execution as execution
    from ilg_toolkit.models import ResNet9Conductance, UNetEmbeddingDistance

    species = tuple(f"species-{index}" for index in range(args.species))
    raster = np.random.default_rng(7301).normal(size=(32, 32, 2)).astype(np.float32)
    positions = np.array([[0, 0], [0, 31], [31, 0], [31, 31], [8, 8], [8, 20], [20, 8], [20, 20]])
    ids = tuple(f"unit-{index}" for index in range(len(positions)))
    prepared = RegionBatch(
        "probe",
        cx.field(raster, "row", "column", "feature"),
        {name: SamplingUnits(ids, positions) for name in species},
    )
    pairs = tuple((ids[i], ids[j]) for i in range(len(ids)) for j in range(i + 1, len(ids)))
    targets = np.random.default_rng(7302).uniform(0.1, 2.0, len(pairs))
    observed = {
        "probe": {
            name: PairwiseObservations.from_pairs(
                pairs, targets * (1 + index / 100), target=TargetSpec("probe divergence")
            )
            for index, name in enumerate(species)
        }
    }
    model = (
        ResNet9Conductance(
            2, species=species, patch_size=4, patch_batch_size=8, key=jax.random.key(7303)
        )
        if args.family == "resnet"
        else UNetEmbeddingDistance(
            2,
            species=species,
            patch_size=1,
            base_channels=4,
            embedding_dim=3,
            key=jax.random.key(7303),
        )
    )
    config = TrainingConfig(
        objective="mlpe",
        steps=1,
        species_chunk_size=args.chunk,
        mlpe_initial_variances=(0.2, 0.4),
        seed=7304,
    )
    events = []
    checked = training._checked_call

    def observe(function, *inputs, **options):
        result = checked(function, *inputs, **options)
        if function is execution.pooled_region:
            events.append({"kind": "encoder", "inference": options.get("inference", True)})
        elif function is execution.compiled_chunk_gradient:
            events.append(
                {"kind": "tail_forward_backward", "species": len(inputs[1].species_index)}
            )
        elif function is execution.compiled_chunk_evaluate:
            events.append({"kind": "tail_evaluation", "species": len(inputs[3].species_index)})
        elif function is training._compiled_update:
            events.append({"kind": "optimizer_update"})
        return result

    # Observe the existing synchronized host boundary only; do not replace any
    # numerical function, array, gradient, RNG key or timing synchronization.
    training._checked_call = observe
    started = time.perf_counter()
    try:
        state = fit([prepared], observed, distance_model=model, config=config)
    finally:
        training._checked_call = checked
    jax.block_until_ready(eqx.filter(state, eqx.is_array))
    elapsed = time.perf_counter() - started
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
        1 if platform.system() == "Darwin" else 1024
    )

    def array_bytes(tree):
        return sum(
            leaf.size * leaf.dtype.itemsize
            for leaf in jax.tree.leaves(eqx.filter(tree, eqx.is_array))
        )

    reports = []
    for path in sorted(args.xla_directory.glob("*memory-usage-report.txt")):
        if not any(name in path.name for name in ("encode", "chunk_gradient", "chunk_evaluate")):
            continue
        report = path.read_text()
        matched = re.search(r"Total bytes(?: used)?:\s*(\d+)", report)
        if matched is None:
            raise RuntimeError(f"Missing XLA allocation total in {path}")
        reports.append(
            {
                "file": path.name,
                "total_allocated_bytes": int(matched.group(1)),
                "report": report,
            }
        )
    if not reports:
        raise RuntimeError("No encoder/tail XLA memory allocation reports found")
    counts = {
        kind: sum(event["kind"] == kind for event in events)
        for kind in ("encoder", "tail_forward_backward", "tail_evaluation", "optimizer_update")
    }
    assert counts == {
        "encoder": 3,
        "tail_forward_backward": (args.species + args.chunk - 1) // args.chunk,
        "tail_evaluation": 2 * ((args.species + args.chunk - 1) // args.chunk),
        "optimizer_update": 1,
    }
    assert all(event.get("species", 0) <= args.chunk for event in events)
    source_root = Path(__file__).resolve().parents[2]
    files = [Path(__file__), *(source_root / "src/ilg_toolkit").glob("**/*.py")]
    result = {
        "schema": "spec52-bounded-cpu-evidence-v1",
        "family": args.family,
        "registry_size": args.species,
        "species_chunk_size": args.chunk,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
        ).strip(),
        "source_sha256": {
            str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        },
        "fixture_sha256": hashlib.sha256(
            raster.tobytes() + positions.tobytes() + targets.tobytes()
        ).hexdigest(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "jax": jax.__version__,
            "affinity": sorted(os.sched_getaffinity(0)),
            "devices": [str(device) for device in jax.devices()],
        },
        "public_fit_seconds_including_compilation": elapsed,
        "process_peak_rss_bytes_including_compiler": rss,
        "persistent_bytes": {
            "distance_model": array_bytes(state.distance_model),
            "optimizer": array_bytes(state.optimizer_state),
            "raw_nuisance": array_bytes(state.raw_variances),
            "selected_calibration_arrays": array_bytes(state.model.calibration),
        },
        "events": events,
        "counts": counts,
        "history_steps": [record.step for record in state.history],
        "final_loss": state.history[-1].training_loss,
        "cpu_compiled_allocation_reports": reports,
        "scope": (
            "Actual public-fit compiled forward/backward buffers; CPU compiler estimates, "
            "not GPU allocator peaks. Persistent calibration arrays grow with every complete "
            "group; RSS also includes imports and compiler caches."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "counts": counts,
                "reports": len(reports),
                "rss_bytes": rss,
            }
        )
    )


if __name__ == "__main__":
    main()
