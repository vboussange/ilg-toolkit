"""Public-fit differential reference for one sampled native group; no private calls."""

import json
import sys
from pathlib import Path

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
    fit,
)
from ilg_toolkit.models import UNetEmbeddingDistance

species = ("fox", "lynx", "owl", "deer")
ids = ("a", "b", "c", "d")
pairs = tuple((a, b) for i, a in enumerate(ids) for b in ids[i + 1 :])
region = RegionBatch(
    "forest",
    cx.field(
        jax.random.normal(jax.random.key(601), (4, 4, 2), dtype=jnp.float32),
        "row",
        "column",
        "feature",
    ),
    {s: SamplingUnits(ids, [[0, 0], [0, 3], [3, 0], [3, 3]]) for s in species},
)
observed = {
    "forest": {
        s: PairwiseObservations.from_pairs(
            pairs,
            np.array([0.3, 0.8, 0.9, 0.2, 0.6, 0.4]) * (i + 1),
            target=TargetSpec("divergence"),
        )
        for i, s in enumerate(species)
    }
}
model = UNetEmbeddingDistance(
    2,
    species=species,
    patch_size=1,
    base_channels=2,
    embedding_dim=2,
    dropout=0.2,
    key=jax.random.key(602),
)
arrays = {}
reports = {}
for objective in ("direct_log1p", "mlpe"):
    config = TrainingConfig(
        objective=objective,
        steps=3,
        species_batch_size=1,
        evaluation_interval=2,
        seed=603,
        mlpe_initial_variances=(0.2, 0.4),
    )
    snapshots = []
    state = fit([region], observed, distance_model=model, config=config, on_step=snapshots.append)
    resumed = fit([region], observed, state=snapshots[0], config=config)
    assert eqx.tree_equal(state, resumed), "public continuation changed"
    for label, value in [
        ("parameters", state.distance_model),
        ("adam", state.optimizer_state),
        ("nuisance", state.raw_variances),
        ("selected", state.best_model),
        ("rng", state.rng_key),
    ]:
        for i, leaf in enumerate(jax.tree.leaves(eqx.filter(value, eqx.is_array))):
            arrays[f"{objective}-{label}-{i:03d}"] = np.asarray(leaf)
    for label, predictions in [
        ("raw", state.model.predict_raw_distances(region, species=species)),
        ("genetic", state.model.predict(region, species=species)),
    ]:
        for name, result in predictions.items():
            arrays[f"{objective}-{label}-{name}"] = np.asarray(result.values)
    reports[objective] = {
        "history": [r.training_loss for r in state.history],
        "steps": [r.step for r in state.history],
        "selected_step": state.selected_step,
        "resumed_exact": True,
    }
Path(sys.argv[1] + ".json").write_text(json.dumps(reports, indent=2) + "\n")
np.savez(sys.argv[1] + ".npz", **arrays)
print(json.dumps(reports))
