"""Frozen added-species data for a future native-API performance comparison."""

import hashlib
import json

import numpy as np
from fixtures import fixture


def species_fixture(case, settings, species_index):
    """Return exactly matched joint/independent inputs without library API code."""
    if species_index not in range(4):
        raise ValueError("The frozen comparison declares species indices 0 through 3")
    data = fixture(case, settings)
    if species_index == 0:
        return data
    grid_side = case["side"] // case["patch_size"]
    left, right = np.triu_indices(case["units"], 1)
    for region_index, item in enumerate(data):
        rng = np.random.default_rng(settings["data_seed"] + 104729 * species_index + region_index)
        nodes = np.sort(rng.choice(grid_side**2, size=case["units"], replace=False))
        positions = np.column_stack(np.unravel_index(nodes, (grid_side, grid_side)))
        item["positions"] = positions * case["patch_size"]
        separation = np.linalg.norm(item["positions"][left] - item["positions"][right], axis=1)
        item["targets"] = 0.2 + separation / case["side"] + rng.uniform(0.0, 0.03, size=len(left))
        digest = hashlib.sha256()
        digest.update(
            json.dumps({"ids": item["ids"], "pairs": item["pairs"]}, sort_keys=True).encode()
        )
        for array in (item["features"], item["positions"], item["targets"]):
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        item["sha256"] = digest.hexdigest()
    return data
