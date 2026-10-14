# Distance models, inputs and training

A distance model turns prepared landscape features and sampling locations into
**ecological distances**, without reading genetic targets. `LandscapeGeneticsModel`
composes it with a genetic calibration. Both one-species and multispecies fits use
this same boundary; there is no separate multispecies wrapper.

## Choose an architecture

Import these classes from `ilg_toolkit.models`. The complete hierarchy is:

| Abstract family | Concrete model | Ecological distance and learned parameters |
| --- | --- | --- |
| `AbstractDistanceModel` → `ConductanceModel` | `ResNet9Conductance` | Effective resistance; neural backbone pooled across species/regions, species-specific positive conductance projections. |
| `AbstractDistanceModel` → `ConductanceModel` | `LogLinearConductance` | Effective resistance; independent coefficient vector and intercept for each species, pooled across its regions. |
| `AbstractDistanceModel` → `EmbeddingDistanceModel` | `UNetEmbeddingDistance` | Squared Euclidean embedding distance; pooled neural backbone and species-specific embedding projections. |

`IBDModel`, imported from `ilg_toolkit`, is a separate geographic-distance baseline.
It accepts distances with explicit physical units and retains its own equal-region
regression policy; it is not a landscape distance architecture.

```python
import jax
from ilg_toolkit import ResistanceSolverConfig
from ilg_toolkit.models import LogLinearConductance, ResNet9Conductance, UNetEmbeddingDistance

# Enable double precision before constructing resistance or MLPE inputs.
jax.config.update("jax_enable_x64", True)
species = ("fox", "lynx")
solver = ResistanceSolverConfig(rtol=1e-6, atol=1e-6, max_steps=1000, use_amg=False)
resistance = ResNet9Conductance(
    3, species=species, patch_size=4, min_conductance=1e-6,
    patch_batch_size=16, solver_config=solver, key=jax.random.key(1),
)
embedding = UNetEmbeddingDistance(
    3, species=species, patch_size=1, base_channels=8, embedding_dim=4,
    dropout=0.1, key=jax.random.key(2),
)
loglinear = LogLinearConductance(
    3, species=species, patch_size=32, min_conductance=0.01,
    max_conductance=1000.0, solver_config=solver,
)
```

Constructors require positive channel/patch settings and a nonempty ordered
registry of unique species names. ResNet patches are at least 4 cells per side;
its default is 4. U-Net defaults are patch size 32, base channels 32, embedding
dimension 16 and dropout 0.1. Raster dimensions must support the chosen patch
layout. A registry fixes parameter-row meanings: it cannot be reordered or
extended after construction. Every registered species needs training observations
in at least one region; registration alone does not train a species.

Conductance models use a four-neighbor graph with positive arithmetic-mean
endpoint conductances as edge weights. Species share regional geometry, adjacency
and compatible solver preparation, but each species has its own weights and
resistance solve. No species-specific topology, mask or disconnected barrier is
introduced. Solver failures preserve numerical context; the toolkit does not
silently relax tolerances. Optional AMG requires the `amg` extra.

## Prepare one raster per region

`RegionBatch(name, features, sampling_units, feature_names=...)` holds a Coordax
field with named `row`, `column`, `feature` dimensions and a mapping such as
`{"fox": SamplingUnits(ids, grid_positions), "lynx": SamplingUnits(...)}`. Grid
positions are integer raster `(row, column)` cells. Sampling units default to
populations; explicitly declare individual kinds when appropriate. Population
MLPE accepts population dissimilarities, not an individual-relatedness likelihood.

Supply `fit` a nonempty region sequence or name-matching region mapping and
observations nested as `observations[region_name][species_name]`. Observations,
partitions, support, folds and access records use these same identity axes. A
one-species workflow retains the nesting. Missing observations or missing species
coverage stay absent; do not fill an artificial Cartesian product with zero targets.

Every group must have the same ordered prepared feature meanings and preprocessing,
and one compatible `TargetSpec` name, kind, units and transform. Preparation owns
reprojection, alignment, scaling and categorical encoding. Named Coordax feature
coordinates or `feature_names` record channel meanings. Validation does not infer
whether two unnamed covariates mean the same thing.

The identity of a biological unit is `(species, label)`. It belongs to one region
within that species; reusing text across different species is valid. New-region
populations need their own identities. Spatial proximity neither merges populations
nor excuses duplicate assignments. `ObservationPartition(region, species, pairs,
role=...)` records exact target use; distinct unordered pairs belong to their group.

## Query ecological or genetic distances

```python
requested = {"fox": [("south", "east")], "lynx": [("ridge", "marsh")]}
raw = model.predict_raw_distances(region, species=("fox", "lynx"), pairs=requested)
predicted = model.predict(region, species=("fox", "lynx"), pairs=requested)
fox_ecological = raw["fox"].values
fox_genetic = predicted["fox"].values  # Target's original units.
matrix = predicted["fox"].as_matrix()
```

The distance model itself has `predict_raw_distances` too. “Raw” means before
genetic calibration. Outputs remain species-keyed, even for one selected species.
A sole registered species can be inferred; a multispecies model always requires
explicit selection. Pair mappings exactly match selected species. Pair order,
orientation, species, region and sampling-unit order are retained. Unknown labels,
self-pairs and duplicate unordered pairs are rejected.

Ordinary fitting and requested-pair prediction compute pair vectors directly.
`ecological_distances(region, species="fox")["fox"]` is an explicit complete-matrix
diagnostic. `Prediction.as_matrix()` only represents its stored pairs: other
entries remain missing, with structural zero dissimilarity diagonals. It does not
infer unrequested values or a joint uncertainty model. Conductance models expose
`conductance_surface(region, species="fox")["fox"]`; embedding models do not.

The calibration hierarchy is independent of architecture.
`AbstractCalibrationModule` has `DirectCalibrationModule`, which applies the
inverse declared target transform, and `UnpooledMLPECalibrationModule`, whose
`groups[region][species]`/`for_group(region, species)` selects one `MLPECalibration`.
Group intercepts, signed slopes, variances and population effects are unpooled.
Raw prediction is available in compatible new regions; MLPE genetic prediction
requires a fitted relationship for the particular group. See
[statistics](statistics.md), [conditioning](mlpe-conditioning.md) and
[region transfer](../examples/region_transfer.py).

## Sampling, execution and selection

`TrainingConfig.steps` counts Adam updates. The full objective gives each species
equal influence and each of its observed regions equal influence, after each
complete group loss is divided by its pair count. Region-first sampling includes
weights correcting both inclusion probabilities and species coverage.

| Setting | Role |
| --- | --- |
| `region_batch_size` | Sample this many regional rasters per update; `None` means all. |
| `species_batch_size` | Sample at most this many observed species within each selected region; `None` means all. |
| `species_chunk_size` | Bound simultaneous selected species work, including differentiated tails/solves. |
| `evaluation_interval` | Fully evaluate at this interval, initially and at the final budget. |
| ResNet `patch_batch_size` | Separately bound encoder patch activations with recomputation. |

For each selected region, pooled neural features are logically computed once and
used by the selected species. Backward checkpointing may deliberately recompute
features. Reduced gradients accumulate before one update; chunking does not cause
one optimizer update per chunk or change which observations enter a selected
MLPE group. Each group retains complete-pair normalization, exact endpoint MLPE
algebra, signed GLS profiling and learned nuisance variances. No random-pair
likelihood approximation is implied by these controls.

Compatible shapes use bounded numerical batching. Names remain host metadata;
compatible dynamic species IDs reuse numerical compilation. Ragged region, unit
and pair shapes can legitimately specialize compilation; there is no global raster
padding to the largest region. Temporary heavy work is bounded, while parameter,
Adam, nuisance and returned calibration/posterior storage still grow with registry
and group count. Requested resistance terminals and their adjoints can also be
large; pair selection is not a negligible-memory guarantee. See
[requested-pair arithmetic](performance/requested-pairs.md) and
[MLPE costs](performance/mlpe.md) for the per-group limits.

Full evaluation reports `training_by_species`, `training_by_group`,
`validation_by_species` and `validation_by_group`. Missing validation groups and
species are explicit. Available validation coverage determines its species-balanced
score; the score does not certify absent groups. Validation uses each group's
training-fitted distance moments, coefficients and variances without reprofiling
on validation targets, and selects only fully evaluated snapshots. With no
validation, the completed final model is selected. Every required calibration is
installed from that selected snapshot and its matching nuisance state, regardless
of which groups were sampled in the last update.

`TrainingState.step` is progress; `.history` is sparse evaluated `StepRecord`s.
`.latest_distance_model` and `.raw_variances` belong to the current update;
`.model`, `.selected_step` and `.selected_raw_variances` describe the selected
snapshot. `on_step` runs after each completed update, including between evaluations,
without forcing evaluation. A zero-step fit has an initial evaluation and no
callback. See [persistence](persistence.md) for fixed-budget interruption and
exact continuation under unchanged execution conditions.

### Choose geographic domains explicitly

Users choose subregions and execution budgets for their raster dimensions and
available memory. Regions may overlap geographically, including when different
species' sampling clusters need different extents. Assign each species' biological
units to one region so overlapping inputs do not duplicate independent group
evidence. There is no automatic crop selector, GPU tuner or prescribed schedule.

A region is a statistical and computational domain. Cropping changes the graph's
available resistance paths and the context of spatial networks. Subregions are
not transparent tiles of a full-raster solution, and adding a halo does not
generally establish equivalence, especially with U-Net spatial normalization.
Independent within-species fold draws also do not establish geographic separation
of a pooled backbone. Use caller-coordinated spatial partitions where needed;
[evaluation](evaluation.md) distinguishes those claims from a common outer test.

## Interpret generic log-linear conductance

Prepared features may be continuous or reference-coded categorical covariates of
any supplied channel count. With an intercept, omit a redundant full set of category
indicators and avoid other collinearities. The toolkit does not silently discard a
category; feature scaling and reference choices determine coefficient interpretation.

For species $s$, first average each covariate within a nonoverlapping patch. Then
apply its coefficient vector and intercept:

$$\eta_s=b_s+\bar x^\top\beta_s,\qquad
c_s=\operatorname{clip}\bigl(\exp[\min(\eta_s,\log(b-a))]+a,\ a,\ b\bigr).$$

Here $a$ and $b$ are positive finite conductance bounds with $a<b$ (separate from
the intercept $b_s$). Defaults are $a=0.01$, $b=1000$, patch size 32. Zero-initialized
coefficients and intercepts give conductance 1.01; construction needs no random
key. Predictor clipping precedes exponentiation to avoid positive overflow.
Averaging pixel-level conductances after the link is a different model.

`coefficients[species_index, feature_index]` follows registry and prepared-feature
order; `intercepts[species_index]` follows the same species association. Away from
the bounds, a unit feature change multiplies conductance *above the floor* by the
exponentiated coefficient, holding other patch means fixed. At saturation that
interpretation and local sensitivity cease to apply. The intercept controls the
baseline for the supplied feature origin, including categorical references.

There is no learned cross-species backbone, shrinkage or shared-plus-deviation
parameterization. Reusing fixed patch means and graph geometry shares computation,
not learned coefficients. Multiplying all conductances by a common constant scales
resistances inversely on a fixed graph. MLPE's distance standardization can remove
such global scale, so an intercept need not be identifiable from standardized MLPE;
the additive floor and cap further limit exact scaling arguments. Do not transfer
an intercept interpretation from a paper-specific preprocessing/likelihood without
checking this objective. See [training statistics](statistics.md).

## Extend the native hierarchy

Custom Equinox classes subclass `AbstractDistanceModel` directly, or one of its
families, and declare a fixed static `species: tuple[str, ...]`. The inherited
public boundary validates region/species/label selections and creates keyed
`RawDistances` metadata. The private numeric primitive
`_pair_distances(features, selection, species_index, *, context=None,
inference=True, key=None, patch_batch_size=None)` returns a pair vector.
`selection.nodes` contains deduplicated numeric endpoints; `selection.left` and
`.right` index them, and `species_index` is a scalar JAX array. Numerical work must
index trainable species rows dynamically, respect stochastic keys, and remain
compatible with differentiation/JIT. It must not close over selected species names
or construct a full pair matrix merely to select requested pairs.

A custom `ConductanceModel` can implement `_conductance(features, species_index,
*, patch_batch_size=None)` and declare `patch_size` and `solver_config`, inheriting
native resistance calculation. Custom embedding or other models can implement the
pair primitive directly. Built-in private pooled execution optimizations are
optional; they are not an additional public interface a custom caller must expose.
Callers never create a latent object, call `encode`, or manage a feature cache.
Custom models use the same fit/prediction and supported explicit-template
[artifact contract](persistence.md), without a legacy adapter or registration.
