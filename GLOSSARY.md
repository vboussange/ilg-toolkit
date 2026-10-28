# ILG Toolkit

The glossary names each concept of ILG Toolkit with one term, used identically
in code, docstrings, examples, notebooks and issues.

## Paper terms

The toolkit follows the vocabulary of the NeurIPS Tackling Climate Change with
Machine Learning (TCCML) workshop paper that introduces the pipeline. Public
identifiers keep their names; this table maps each paper term to the glossary
term and the public identifier that implements it.

| Paper term | Glossary term | Public identifier |
| --- | --- | --- |
| Patch encoder | Patch encoder | `LogLinearConductance`, `ResNet9Conductance` |
| Conductance grid graph | Conductance grid graph | `ConductanceModel.conductance_surface` |
| Vertex | Vertex | One cell of `ConductanceModel.conductance_surface(...)` |
| Conductance $C$ | Conductance surface | `ConductanceModel.conductance_surface` |
| Effective-resistance layer | Effective-resistance layer | `ResistanceSolverConfig`, `ConductanceModel.ecological_distances` |
| Local resistance $Q$ | Resistance surface | `1 / ConductanceModel.conductance_surface(...)` |
| Complete pooling | Complete pooling | `train` with a region collection |
| No pooling | No pooling | One `train` call per region |
| Population-holdout evaluation | Population-holdout evaluation | `PairwiseObservations.select_pairs` |
| Isolation-by-distance (IBD) model | Isolation-by-distance (IBD) model | `IBDModel` |
| U-Net embedding-distance model | Embedding | `UNetEmbeddingDistance` |
| Initialization at the best uniform conductance | Patch encoder | `initialize_uniform_conductance` |
| ILG pipeline | Distance model | `AbstractDistanceModel` |
| Population | Sampling unit | `RegionBatch.sampling_unit_ids` |
| Species | Species | `AbstractDistanceModel.species`, `RegionBatch.species`, `PairwiseObservations.species` |
| Genetic dissimilarity | Pairwise genetic observation | `PairwiseObservations` |
| Covariates | Landscape features | `RegionBatch.features` |
| Geographic region | Region | `RegionBatch` |
| Fit | Training | `train` |
| Objective | Loss | `Log1pLoss`, `MLPELoss` |

## Language

**Inverse landscape genetics (ILG)**:
Inferring spatial ecological networks from geolocated genetic data by learning
a conductance grid graph whose edge weights encode local landscape permeability
to movement, so that the effective resistances it induces between sampled
populations predict their genetic dissimilarity.
ILG Toolkit also supports squared embedding distances as ecological distances,
and individuals as sampling units.

**Region**:
A landscape and its associated sampling units and genetic observations.
_Paper_: geographic region.
_Avoid_: Dataset, when referring to one region within a multi-region study.

**Region batch**:
The prepared landscape features, sampling-unit identifiers, and grid positions
for one region, independent of genetic observations.
A grid position is the raster cell, as a row and column index, that contains a
sampling unit. `RegionBatch.from_points` snaps projected sampling-unit coordinates
to grid positions; synthetic studies give grid positions directly.
An optional species column labels each sampling unit; when omitted, all sampling
units belong to one species.
_Avoid_: Prepared region, when naming this input boundary.

**Individual-based region batch**:
A region batch whose sampling units are individuals
(`RegionBatch(..., individual_based=True)`).
MLPE losses and MLPE calibration heads reject individual-based region batches;
the log1p loss accepts them.

**Landscape features**:
Named raster layers describing a region's landscape, with row, column, and feature dimensions.
A labeled feature axis names the layers (`RegionBatch.feature_names`); row and column
coordinates, when present, locate the raster cells in a projected coordinate reference system.
Their meanings and order are shared by every region a distance model is trained on.
_Paper_: covariates.

**Sampling unit**:
An identified population or individual with a landscape location, linked to other sampling units by pairwise genetic observations.
_Paper_: population.
_Avoid_: Population, when the sampling unit represents an individual.

**Species**:
The ordered, unique species names a distance model carries, one for each
species-specific head, given once with `species=` when the distance model is
built. A single-species distance model may leave its species unnamed
(`species=None`). Training, calibration, prediction and saved models read the
names from the distance model. Species names select a species' parameters,
genetic predictions and conductance surface.
_Avoid_: Species order; species count; registry.

**Pairwise genetic observation**:
A measured genetic relationship between two sampling units, whose meaning and scale are defined by the chosen target. Genetic dissimilarity and individual relatedness are different kinds of pairwise genetic observation.
_Paper_: genetic dissimilarity.
_Avoid_: Genetic distance, when the observation is a similarity or relatedness measure.

**Region–species group**:
One species' sampling units and pairwise genetic observations within one region.
Losses are evaluated, and MLPE calibrations fitted, for each region–species group.

**Distance model**:
A model deriving ecological distances between sampling units from landscape
features and sampling-unit locations. Patch encoders, the IBD model and the
U-Net embedding-distance model are distance models. Effective resistance through
a conductance grid graph and squared embedding distance are supported forms of
ecological distance.
_Paper_: ILG pipeline, for a patch encoder composed with the effective-resistance layer.
_Avoid_: Genetic predictor; relationship, which names the genetic relationship
fitted by calibration.

**Patch encoder**:
The distance model that maps landscape features to the vertex conductances of a
conductance grid graph, one patch of landscape features per vertex, and so
provides a feature-to-conductance map. The patch size decouples the resolution
of landscape features from that of the graph. Patch encoders are trained; the
conductance and resistance surfaces they produce are learned.
The log-linear patch encoder (`LogLinearConductance`) has low capacity; the
ResNet9 patch encoder (`ResNet9Conductance`) is a deep patch encoder.
_Avoid_: Landscape relationship; common relationship; neural network or network,
for a patch encoder.

**Backbone**:
The species-shared part of a distance model, mapping landscape features to the
backbone output that every species-specific head reads
(`AbstractDistanceModel.backbone`).
_Avoid_: Patch features; shared patch features, for the backbone output.

**Species-specific head**:
The layer mapping backbone output to one species' conductance or embedding.
_Avoid_: Head, without a qualifier.

**Isolation-by-distance (IBD) model**:
The distance model whose ecological distance is a non-negative affine function
of the straight-line distance between sampling units' vertices. One IBD model is
trained jointly across regions.
_Avoid_: IBR, when referring to this geographic-distance baseline; common
relationship; geographic relationship.

**Conductance surface**:
A spatial grid of positive values describing how readily a landscape permits movement for one species, with one conductance $C$ per vertex of the conductance grid graph.

**Resistance surface**:
Local resistance $Q = 1/C$ at each vertex, the inverse of the conductance surface.
A trained patch encoder yields a learned resistance surface, the main result of
an ILG study.

**Conductance grid graph**:
A four-neighbor grid graph with one vertex per patch of landscape features,
weighted by the conductance surface; the edge between neighboring vertices $k$
and $l$ has conductance $(C_k + C_l)/2$.
_Avoid_: Conductance graph; resistance graph.

**Vertex**:
One point of a conductance grid graph, standing for one patch of landscape
features and carrying one conductance; sampling units in the same patch share a
vertex. Identifiers such as `RegionBatch.pixel_nodes` keep their names.
_Avoid_: Node, in prose.

**Effective-resistance layer**:
The differentiable computation of effective resistance between sampling-unit
vertices of a conductance grid graph, with gradients with respect to its
conductances.
_Avoid_: Distance solver; resistance calculation.

**Embedding**:
A vector representation of a sampling unit's landscape features whose squared Euclidean distance from another such vector is their ecological distance.
_Avoid_: Squared separation, when referring to squared embedding distance.

**Ecological distance**:
A distance model's pairwise separation between two sampling units based on their landscape, before any genetic calibration. Effective resistance through a conductance grid graph is the main example.
_Avoid_: Raw distance; landscape score; genetic prediction, before calibration to the target scale.

**Complete pooling**:
Training one distance model jointly from every region, with equal region influence.
With several species, the backbone is shared by all regions and species, each
species keeps its own species-specific head, species receive equal influence,
and regions receive equal influence within each species.
_Avoid_: Pooled backbone; distance model pooled across regions.

**No pooling**:
Training one distance model per region, using only that region's genetic observations.

**Training configuration**:
The chosen loss, joint optimization budget, and batching settings for training a distance model and its statistical parameters.
_Avoid_: Calibration settings, when referring to the training budget.

**Training**:
Joint estimation of distance-model parameters and the statistical parameters required by a loss, using supplied genetic observations.
_Paper_: fit.
_Avoid_: Fitting, when naming this optimization workflow.

**Checkpoint**:
A saved training state, holding the latest parameters, statistical parameter state, optimizer and random state, and selected model, for continuing training.
_Paper_: none.

**Loss**:
A statistical criterion relating a region–species group's ecological distances to its genetic observations, together with the criterion's numerical settings.
_Paper_: objective.
_Avoid_: Objective name, when referring to the criterion and its settings.

**Statistical parameter state**:
The current statistical parameters estimated jointly with distance-model parameters during training, as required by the chosen loss.
For maximum-likelihood population effects (MLPE) these are variance parameters; the direct log1p criterion has none.
_Avoid_: MLPE calibration, when referring only to the parameters being optimized.

**Landscape genetics model**:
A distance model composed with a calibration head. The distance model maps
landscape features to ecological distances; the calibration head maps
ecological distances to genetic predictions on the original scale.
_Avoid_: Distance model, when referring to the complete genetic prediction model.

**Genetic calibration**:
The relationship between ecological distances and genetic observations on the model scale.
Direct calibration applies the inverse target transform in every region; MLPE calibration is group-specific.

**Calibration head**:
The part of a landscape genetics model that maps ecological distances to genetic
predictions on the original scale and owns the target contract. A direct
calibration head applies the inverse target transform in any region; an MLPE
calibration head holds one MLPE calibration per region–species group.
_Avoid_: Head, without a qualifier; calibration module; recalibrated distance
model, when the distance model stays frozen.

**MLPE calibration**:
One region–species group's fitted relationship between ecological distances and
genetic targets on the model scale, under maximum-likelihood population effects
(MLPE): distance moments, coefficients, both variance components and the
population-effect posterior.
_Avoid_: Regional MLPE calibration; fitted calibration; complete model, when
referring only to this fitted relationship.

**Group-specific calibration**:
Calibration coefficients and variance parameters estimated separately for each
region–species group, retaining population dependence within each MLPE calibration.
_Avoid_: Unpooled genetic calibration; regional calibration, when describing this policy.

**Ensemble**:
An ordered collection of compatible fitted landscape genetics models whose genetic predictions are averaged on the original target scale.
_Avoid_: Fold ensemble, when no particular training split is implied.

**Ensemble member**:
One fitted model at a given position in an ensemble.
_Avoid_: Fold, when referring to the fitted model rather than the population partition.

**Member spread**:
Descriptive variability among ensemble members' predictions or conductance surfaces.
_Avoid_: Predictive uncertainty, when referring only to member variability.

**Population pair**:
An unordered pair of distinct populations within a landscape, associated with an observed or predicted genetic dissimilarity.
_Avoid_: Sample, when referring to a pair rather than an individual population or genetic specimen.

**Selected model**:
A fitted landscape genetics model retained for prediction from the evaluated
training steps, chosen by the lowest validation loss when validation
observations are supplied. Without validation observations, the final evaluated
model is selected.
_Avoid_: Best model; snapshot, when referring to the selected model.

**Latest model**:
The fitted landscape genetics model corresponding to the current training step's distance and statistical parameters.
_Avoid_: Selected model, when referring to the state used for continued training.

**Population-holdout evaluation**:
Assessment of predictive accuracy on pairwise genetic observations involving sampling units withheld from training and selection.
Each region's populations are split into folds; for each fold, a model trained without the fold's populations is scored on the within-region pairs that involve them.

**Explanatory evaluation**:
Assessment of the association between frozen ecological distances and genetic
observations through fresh calibration on the assessed populations.
Its score measures association; population-holdout evaluation measures
predictive accuracy.

**Observation selection**:
A subset of measured sampling-unit pairs that retains their targets, species, and sampling-unit identities.
_Avoid_: Observation partition, when referring to the selected measurements themselves.

**Query pair**:
An unordered pair of distinct sampling units for which a prediction is requested.
Its measured target, when available for evaluation, is used afterward to score the prediction.

**Support observations**:
Pairwise genetic observations used to condition population effects for predictions on separate query pairs.
The distance model, distance moments, fitted coefficients, and variance parameters stay fixed.

**Marginal prediction**:
A genetic prediction using the fitted relationship with population effects averaged over their zero-mean distribution.

**Saved-effect prediction**:
A genetic prediction using the population-effect posterior retained by an MLPE calibration.
_Avoid_: Support conditioning, when no additional observations are supplied.

**Support conditioning**:
An update of the saved population-effect posterior using additional observations disjoint from calibration and query measurements.
_Avoid_: Recalibration, which estimates a new genetic relationship.

**Population effect**:
A population's additive random contribution to model-scale genetic targets in MLPE calibration, shared by every pair involving that population.

**Original scale**:
The measurement scale of pairwise genetic observations, in the target's units, on which genetic predictions are returned.

**Model scale**:
The scale of genetic targets after the chosen target transform, on which genetic calibration is fitted.
It coincides with the original scale when the transform is the identity.

**Genetic prediction**:
An estimated genetic relationship for an identified pair of sampling units in a
region, expressed in the target's original units.
A prediction matrix arranges values for the requested pairs.
_Avoid_: Calibrated distance.
