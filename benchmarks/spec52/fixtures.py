"""Revision-independent synthetic inputs for the frozen spec 52 protocol."""

import hashlib
import json
from pathlib import Path

import numpy as np

PROTOCOL_PATH = Path(__file__).with_name("protocol.json")


def protocol():
    return json.loads(PROTOCOL_PATH.read_text())


def fixture(case, settings):
    """Return NumPy arrays and labels only; no library compatibility layer."""
    output = []
    grid_side = case["side"] // case["patch_size"]
    nodes = np.linspace(0, grid_side**2 - 1, case["units"], dtype=np.int64)
    positions = np.column_stack(np.unravel_index(nodes, (grid_side, grid_side)))
    positions = positions * case["patch_size"]
    left, right = np.triu_indices(case["units"], 1)
    for index in range(case["regions"]):
        rng = np.random.default_rng(settings["data_seed"] + index)
        features = rng.normal(size=(case["side"], case["side"], case["channels"])).astype(
            np.float32
        )
        separation = np.linalg.norm(positions[left] - positions[right], axis=1)
        targets = 0.2 + separation / case["side"] + rng.uniform(0.0, 0.03, size=len(left))
        ids = tuple(f"region-{index}-unit-{unit:03d}" for unit in range(case["units"]))
        pairs = tuple((ids[a], ids[b]) for a, b in zip(left, right, strict=True))
        digest = hashlib.sha256()
        digest.update(json.dumps({"ids": ids, "pairs": pairs}, sort_keys=True).encode())
        for array in (features, positions, targets):
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        output.append(
            dict(
                name=f"region-{index}",
                features=features,
                ids=ids,
                positions=positions,
                pairs=pairs,
                targets=targets,
                sha256=digest.hexdigest(),
            )
        )
    return output


def summary(samples):
    """Raw observations plus descriptive uncertainty, never a CI assertion."""
    values = np.asarray(samples, dtype=np.float64)
    rng = np.random.default_rng(52053)
    medians = np.median(rng.choice(values, size=(10000, len(values)), replace=True), axis=1)
    return dict(
        samples_seconds=values.tolist(),
        median_seconds=float(np.median(values)),
        minimum_seconds=float(values.min()),
        maximum_seconds=float(values.max()),
        sample_std_seconds=float(np.std(values, ddof=1)) if len(values) > 1 else None,
        median_bootstrap_95_percent_seconds=np.quantile(medians, [0.025, 0.975]).tolist(),
        bootstrap_resamples=10000,
        bootstrap_seed=52053,
    )
