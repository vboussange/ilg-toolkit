"""Pool a neural backbone across species and regions, with exact group MLPE."""

from itertools import combinations

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
    UnpooledMLPECalibrationModule,
    fit,
)
from ilg_toolkit.models import UNetEmbeddingDistance

jax.config.update("jax_enable_x64", True)
rng = np.random.default_rng(31)
species = ("fox", "lynx")  # Fixed ordered registry: every species has training evidence.
target = TargetSpec("synthetic divergence", units="index")
regions = []
observations = {}
validation = {}
for name, width, present in (("headwaters", 4, species), ("lowlands", 6, ("fox",))):
    features = rng.normal(size=(4, width, 2)).astype(np.float32)
    units = {}
    observed = {}
    for animal in present:
        positions = (
            np.array([[0, 0], [0, width - 1], [1, 1], [2, 2], [3, 0], [3, width - 1]])
            if animal == "fox"
            else np.array([[0, 1], [1, 0], [1, 3], [2, 1], [3, 2]])
        )
        # A biological unit belongs to one region within its species. Text may
        # repeat across species; species identity keeps the observations separate.
        labels = tuple(f"{name}-{i}" for i in range(len(positions)))
        units[animal] = SamplingUnits(labels, positions)
        pairs = tuple(combinations(labels, 2))
        left, right = np.triu_indices(len(labels), 1)
        covariates = features[positions[:, 0], positions[:, 1]]
        weights = np.array([0.4, 1.2]) if animal == "fox" else np.array([1.1, 0.3])
        distances = np.sum(((covariates[left] - covariates[right]) * weights) ** 2, axis=1)
        effects = rng.normal(scale=0.06, size=len(labels))
        measured = (
            2 + 0.12 * distances + effects[left] + effects[right] + rng.normal(0, 0.03, len(pairs))
        )
        if name == "headwaters" and animal == "fox":
            # This small validation subset does not cover lynx or lowlands/fox.
            validation[name] = {
                animal: PairwiseObservations.from_pairs(
                    pairs[-2:],
                    measured[-2:],
                    sampling_unit_ids=labels,
                    target=target,
                )
            }
            pairs, measured = pairs[:-2], measured[:-2]
        observed[animal] = PairwiseObservations.from_pairs(
            pairs,
            measured,
            sampling_unit_ids=labels,
            target=target,
        )
    regions.append(
        RegionBatch(
            name,
            cx.field(features, "row", "column", "feature"),
            units,
            feature_names=("elevation", "canopy"),
        )
    )
    observations[name] = observed

initial = UNetEmbeddingDistance(
    2,
    species=species,
    patch_size=1,
    base_channels=2,
    embedding_dim=2,
    dropout=0.1,
    key=jax.random.key(9),
)
result = fit(
    regions,
    observations,
    validation=([regions[0]], validation),
    distance_model=initial,
    config=TrainingConfig(
        objective="mlpe",
        steps=3,
        learning_rate=0.001,
        seed=17,
        region_batch_size=1,  # Sample one regional raster per update.
        species_batch_size=1,  # Sample one observed species inside that region.
        species_chunk_size=1,  # Bound simultaneous species work, not the likelihood.
        evaluation_interval=2,  # Full evaluations at 0, 2 and the final step 3.
    ),
)
trained = result.latest_distance_model
assert isinstance(trained, UNetEmbeddingDistance)
assert not eqx.tree_equal(initial.backbone, trained.backbone)
# Both species must receive an update under this short seeded schedule;
# changing only the shared backbone would not demonstrate both learned responses.
for index, animal in enumerate(species):
    assert not np.array_equal(initial.projection_weight[index], trained.projection_weight[index]), (
        f"The demonstration schedule did not train {animal}"
    )
assert [record.step for record in result.history] == [0, 2, 3]
print("Registry:", result.model.species)
print("Optimizer step / selected evaluated step:", result.step, result.selected_step)
for record in result.history:
    print("Evaluation", record.step, "species:", record.training_by_species)
    print("Groups:", record.training_by_group)
    print(
        "Missing validation species/groups:",
        record.validation_missing_species,
        record.validation_missing_groups,
    )

# Every selected group uses all its declared training pairs for normalization,
# signed GLS profiling and MLPE. There is no random-pair likelihood minibatch.
assert isinstance(result.model.calibration, UnpooledMLPECalibrationModule)
for region in regions:
    selected = tuple(region.sampling_units)
    requests = {animal: observations[region.name][animal].observed_pairs[:2] for animal in selected}
    raw = result.model.predict_raw_distances(region, species=selected, pairs=requests)
    genetic = result.model.predict(region, species=selected, pairs=requests)
    for animal in selected:
        head = result.model.calibration.for_group(region.name, animal)
        assert head.region_name == region.name and head.species_name == animal
        assert np.isfinite(raw[animal].values).all() and np.isfinite(genetic[animal].values).all()
        print(
            region.name,
            animal,
            "ecological:",
            raw[animal].values,
            "genetic:",
            genetic[animal].values,
        )
# The last sampled update need not touch every group. The selected snapshot still
# has all three exact group calibrations, fitted with its matching nuisance state.
assert sum(len(groups) for groups in result.model.calibration.groups.values()) == 3
