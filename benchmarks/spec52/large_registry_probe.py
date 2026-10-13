"""Run one instrumented public MLPE fit with fixed four-species training selection.

Use fresh processes and empty XLA dump directories for each case. This extends
bounded_probe.py without editing its historical driver or raw evidence. CPU XLA
buffer allocations, live-array boundary snapshots and process RSS are different
quantities; none alone is an end-to-end device allocator peak.
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
from collections import Counter
from dataclasses import asdict
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("unet", "resnet"), required=True)
    parser.add_argument("--species", type=int, choices=(4, 100, 1000), required=True)
    parser.add_argument("--chunk", type=int, choices=(1, 4), required=True)
    parser.add_argument("--ragged", action="store_true")
    parser.add_argument("--xla-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.ragged and (args.species != 4 or args.chunk != 1):
        parser.error("Ragged diagnostic uses registry 4 and chunk 1")
    if f"--xla_dump_to={args.xla_directory}" not in os.environ.get("XLA_FLAGS", ""):
        parser.error("Set the matching XLA dump directory before importing JAX")
    if args.xla_directory.exists() and any(args.xla_directory.iterdir()):
        parser.error("Each process requires a fresh empty XLA dump directory")
    root = Path(__file__).resolve().parents[2]

    def git(*arguments):
        return subprocess.check_output(["git", *arguments], cwd=root, text=True).strip()

    if git("status", "--porcelain", "--untracked-files=no"):
        parser.error("Commit tracked source changes before recording measurements")
    source_commit = git("rev-parse", "HEAD")
    files = [Path(__file__), *(root / "src/ilg_toolkit").glob("**/*.py")]
    source_hashes = {str(path.relative_to(root)): digest(path) for path in files}
    for path, expected in source_hashes.items():
        committed = subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=root)
        if hashlib.sha256(committed).hexdigest() != expected:
            parser.error(f"Uncommitted measured source: {path}")

    import coordax as cx
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np

    from ilg_toolkit import (
        PairwiseObservations,
        RegionBatch,
        SamplingUnits,
        TargetSpec,
        TrainingConfig,
        UnpooledMLPECalibrationModule,
        fit,
        training,
    )
    from ilg_toolkit import _training_execution as execution
    from ilg_toolkit.models import ResNet9Conductance, UNetEmbeddingDistance

    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError("The unchanged MLPE fixture requires explicit JAX x64")
    jax.config.update("jax_enable_compilation_cache", False)
    species = tuple(f"species-{index}" for index in range(args.species))
    raster = np.random.default_rng(7301).normal(size=(32, 32, 2)).astype(np.float32)
    positions = np.array([[0, 0], [0, 31], [31, 0], [31, 31], [8, 8], [8, 20], [20, 8], [20, 20]])
    ids = tuple(f"unit-{index}" for index in range(len(positions)))
    all_pairs = tuple((ids[i], ids[j]) for i in range(len(ids)) for j in range(i + 1, len(ids)))
    targets = np.random.default_rng(7302).uniform(0.1, 2.0, len(all_pairs))
    units, observations, fixture_groups = {}, {}, {}
    for index, name in enumerate(species):
        count = 6 if args.ragged and index == args.species - 1 else 8
        labels = ids[:count]
        pairs = tuple(pair for pair in all_pairs if set(pair).issubset(labels))
        values = np.asarray([targets[all_pairs.index(pair)] for pair in pairs]) * (1 + index / 100)
        units[name] = SamplingUnits(labels, positions[:count])
        observations[name] = PairwiseObservations.from_pairs(
            pairs, values, target=TargetSpec("probe divergence")
        )
        fixture_groups[name] = {
            "units": count,
            "pairs": len(pairs),
            "targets_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        }
    prepared = RegionBatch("probe", cx.field(raster, "row", "column", "feature"), units)
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
        steps=2,
        species_batch_size=4,
        species_chunk_size=args.chunk,
        evaluation_interval=2,
        mlpe_initial_variances=(0.2, 0.4),
        seed=7304,
    )

    def array_bytes(tree):
        return sum(
            leaf.size * leaf.dtype.itemsize
            for leaf in jax.tree.leaves(eqx.filter(tree, eqx.is_array))
        )

    def live_snapshot(label):
        arrays = jax.live_arrays()
        spans = {}
        logical = 0
        for array in arrays:
            size = array.size * array.dtype.itemsize
            logical += size
            address = array.unsafe_buffer_pointer()
            # Same-address views can have different extents. This is a logical
            # span count, not an allocator capacity or internal kernel peak.
            spans[address] = max(spans.get(address, 0), size)
        return {
            "phase": label,
            "live_array_objects": len(arrays),
            "logical_array_bytes": logical,
            "same_address_deduplicated_span_bytes": sum(spans.values()),
            "distinct_start_addresses": len(spans),
            "device_memory_stats": [device.memory_stats() for device in jax.devices()],
        }

    report_roles = {}
    shape_catalogue = {}
    events, snapshots = [], [live_snapshot("before_fit")]
    checked = training._checked_call
    seen_reports = set(args.xla_directory.glob("*memory-usage-report.txt"))

    def observe(function, *inputs, **options):
        if function in (execution.pooled_region, execution.compiled_pooled_region):
            inference = options.get("inference", True)
            kind = "encoder_inference" if inference else "encoder_training_forward"
            payload = None
        elif function is execution.compiled_chunk_gradient:
            kind, payload = "tail_forward_backward", inputs[1]
        elif function is execution.compiled_chunk_evaluate:
            kind, payload = "tail_evaluation", inputs[3]
        elif function is training._compiled_update:
            kind, payload = "optimizer_update", None
        else:
            kind, payload = "encoder_backward", None
        event = {"kind": kind, "step": options["step"], "started_ns": time.monotonic_ns()}
        if payload is not None:
            event["species_indices"] = np.asarray(payload.species_index).tolist()
            event["group_indices"] = np.asarray(payload.group_index).tolist()
            shapes = [(list(leaf.shape), str(leaf.dtype)) for leaf in jax.tree.leaves(payload)]
            signature = hashlib.sha256(json.dumps(shapes).encode()).hexdigest()
            event["payload_signature"] = signature
            shape_catalogue[signature] = shapes
        result = checked(function, *inputs, **options)
        event["finished_ns"] = time.monotonic_ns()
        current = set(args.xla_directory.glob("*memory-usage-report.txt"))
        new = sorted(current - seen_reports)
        seen_reports.update(current)
        relevant = [
            path
            for path in new
            if any(
                token in path.name
                for token in ("encode", "pooled_region", "chunk_gradient", "chunk_evaluate")
            )
        ]
        event["new_memory_reports"] = [path.name for path in relevant]
        for path in relevant:
            report_roles[path] = kind
        events.append(event)
        if payload is None or (
            kind == "tail_forward_backward" or max(event["species_indices"]) == args.species - 1
        ):
            snapshots.append(live_snapshot(f"after_{kind}_step{options['step']}"))
        return result

    training._checked_call = observe
    started = time.perf_counter()
    try:
        state = fit([prepared], {"probe": observations}, distance_model=model, config=config)
    finally:
        training._checked_call = checked
    jax.block_until_ready(eqx.filter(state, eqx.is_array))
    elapsed = time.perf_counter() - started
    snapshots.append(live_snapshot("returned_complete_state"))
    fit_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
        1 if platform.system() == "Darwin" else 1024
    )
    if not isinstance(state.model.calibration, UnpooledMLPECalibrationModule):
        raise AssertionError("Expected complete native group MLPE calibrations")
    distance_model = state.distance_model
    assert isinstance(distance_model, (UNetEmbeddingDistance, ResNet9Conductance))
    calibrated = state.model.calibration.groups["probe"]
    if set(calibrated) != set(species):
        raise AssertionError("Full initial/final evaluation omitted registered observed groups")
    for name, head in calibrated.items():
        if head.calibration_pairs != observations[name].observed_pairs:
            raise AssertionError("Completed calibration omitted pairs")
    if not all(np.isfinite(record.training_loss) for record in state.history):
        raise AssertionError("Nonfinite complete evaluation loss")

    # Public requested-pair outputs and cross-species isolation, outside the fit
    # allocation/compilation stream. No full all-species prediction is requested.
    requested = (species[0], species[-1])
    selections = {
        name: ((units[name].ids[3], units[name].ids[0]), (units[name].ids[1], units[name].ids[2]))
        for name in requested
    }
    raw = state.model.predict_raw_distances(prepared, species=requested, pairs=selections)
    predicted = state.model.predict(prepared, species=requested, pairs=selections)
    unselected = [index for index, name in enumerate(species) if name not in requested]
    altered = eqx.tree_at(
        lambda value: value.projection_weight,
        state.distance_model,
        distance_model.projection_weight.at[jnp.asarray(unselected)].add(1.0),
    )
    changed = altered.predict_raw_distances(prepared, species=requested, pairs=selections)
    numerical = {}
    for name in requested:
        assert raw[name].pairs == selections[name] == predicted[name].pairs
        assert raw[name].species_name == name == predicted[name].species_name
        np.testing.assert_array_equal(raw[name].values, changed[name].values)
        single = state.model.predict_raw_distances(
            prepared, species=(name,), pairs={name: selections[name]}
        )
        np.testing.assert_array_equal(raw[name].values, single[name].values)
        numerical[name] = {
            "pairs": selections[name],
            "raw_distances": np.asarray(raw[name].values).tolist(),
            "genetic_predictions": np.asarray(predicted[name].values).tolist(),
        }
    assert tuple(raw) == requested == tuple(predicted)
    if args.family == "unet":
        scaled = eqx.tree_at(
            lambda value: value.projection_weight,
            state.distance_model,
            distance_model.projection_weight.at[0].multiply(1.1),
        ).predict_raw_distances(
            prepared, species=(species[0],), pairs={species[0]: selections[species[0]]}
        )
        np.testing.assert_allclose(
            scaled[species[0]].values, raw[species[0]].values * 1.21, rtol=2e-5
        )
        numerical["independent_reference"] = "squared distance scales by 1.1**2 for scaled row"
    else:
        surfaces = state.model.conductance_surface(prepared, species=requested)
        errors = []
        for name, surface in surfaces.items():
            grid = np.asarray(surface, dtype=np.float64)
            height, width = grid.shape
            laplacian = np.zeros((height * width, height * width))
            for row in range(height):
                for column in range(width):
                    for dr, dc in ((1, 0), (0, 1)):
                        if row + dr < height and column + dc < width:
                            left, right = row * width + column, (row + dr) * width + column + dc
                            edge = (grid[row, column] + grid[row + dr, column + dc]) / 2
                            laplacian[left, left] += edge
                            laplacian[right, right] += edge
                            laplacian[left, right] -= edge
                            laplacian[right, left] -= edge
            inverse = np.linalg.pinv(laplacian, hermitian=True)
            node = {
                label: int(row // 4 * width + column // 4)
                for label, (row, column) in zip(ids, positions, strict=True)
            }
            expected = np.asarray(
                [
                    inverse[node[a], node[a]]
                    + inverse[node[b], node[b]]
                    - 2 * inverse[node[a], node[b]]
                    for a, b in selections[name]
                ]
            )
            np.testing.assert_allclose(raw[name].values, expected, rtol=2e-5, atol=2e-6)
            errors.append(float(np.max(np.abs(np.asarray(raw[name].values) - expected))))
        numerical["independent_reference"] = "NumPy dense graph Laplacian pseudoinverse"
        numerical["maximum_absolute_errors"] = errors

    reports = []
    for path, role in sorted(report_roles.items()):
        report = path.read_text()
        matched = re.search(r"Total bytes(?: used)?:\s*(\d+)", report)
        if matched is None:
            raise RuntimeError(f"Missing XLA allocation total: {path}")
        hlo = path.with_name(path.name.replace("-memory-usage-report", ""))
        reports.append(
            {
                "file": path.name,
                "role": role,
                "total_allocated_bytes": int(matched[1]),
                "entry_layout": hlo.read_text().splitlines()[0],
                "report": report,
                "report_sha256": digest(path),
                "hlo_sha256": digest(hlo),
            }
        )
    result = {
        "schema": "spec52-large-registry-v2",
        "family": args.family,
        "registry_size": args.species,
        "species_batch_size": 4,
        "species_chunk_size": args.chunk,
        "ragged": args.ragged,
        "source_commit": source_commit,
        "source_tree": git("rev-parse", "HEAD:src"),
        "source_sha256": source_hashes,
        "fixture_sha256": hashlib.sha256(
            raster.tobytes() + positions.tobytes() + targets.tobytes()
        ).hexdigest(),
        "fixture_groups": fixture_groups,
        "training_config": asdict(config),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "jax": jax.__version__,
            "equinox": eqx.__version__,
            "numpy": np.__version__,
            "affinity": sorted(os.sched_getaffinity(0)),
            "devices": [str(device) for device in jax.devices()],
            "xla_flags": os.environ["XLA_FLAGS"],
            "persistent_compilation_cache": False,
        },
        "public_fit_seconds_including_compilation_and_instrumentation": elapsed,
        "process_peak_rss_bytes_including_compiler": fit_rss,
        "persistent_bytes": {
            "initial_distance_model_retained_by_driver": array_bytes(model),
            "latest_distance_model": array_bytes(state.distance_model),
            "latest_backbone": array_bytes(distance_model.backbone),
            "latest_species_projection": array_bytes(
                (distance_model.projection_weight, distance_model.projection_bias)
            ),
            "optimizer": array_bytes(state.optimizer_state),
            "latest_nuisance": array_bytes(state.raw_variances),
            "selected_nuisance": array_bytes(state.selected_raw_variances),
            "selected_distance_model": array_bytes(state.model.distance_model),
            "selected_calibration_arrays": array_bytes(state.model.calibration),
            "prepared_region_arrays": array_bytes(
                (prepared.feature_array, *(unit.grid_positions for unit in units.values()))
            ),
            "observation_arrays": array_bytes(
                tuple(observed.values for observed in observations.values())
            ),
            "note": (
                "Logical category sizes; shared/aliased initial/latest/selected buffers "
                "are not additive allocations."
            ),
        },
        "payload_shapes_by_signature": shape_catalogue,
        "events": events,
        "counts": dict(Counter(event["kind"] for event in events)),
        "live_array_boundary_snapshots": snapshots,
        "history_steps": [record.step for record in state.history],
        "final_loss": state.history[-1].training_loss,
        "calibrated_groups": len(calibrated),
        "selected_prediction_checks": {
            "passed": True,
            "selected_species": requested,
            "numerical": numerical,
        },
        "cpu_compiled_allocation_reports": reports,
        "scope": (
            "CPU compiled buffer estimates and synchronized boundary live-array spans, "
            "not an end-to-end GPU/device peak; RSS includes host/compiler/output storage."
        ),
    }
    from large_registry_analysis import analyze_case

    analyze_case(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "fit_seconds": elapsed,
                "counts": result["counts"],
                "rss_bytes": fit_rss,
            }
        )
    )


if __name__ == "__main__":
    main()
