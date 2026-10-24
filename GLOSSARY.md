# ILG Toolkit

Inverse landscape genetics: learning relationships between landscapes and pairwise genetic observations among sampled populations or individuals.

## Language

**Region batch**:
The prepared landscape features, sampling-unit identifiers, and grid positions for one region, prepared before genetic observations are attached.
An optional species column labels each sampling unit; when omitted, all sampling units belong to one species.
_Avoid_: Prepared region, when naming this input boundary.

**Training configuration**:
The optimization budget and numerical settings supplied as `TrainingConfig`.
_Avoid_: FitConfig, when naming this public configuration.

**Landscape genetics model**:
A distance model composed with genetic calibration, retaining feature meanings,
target semantics and the observations used for training and selection.
Its `predict` returns genetic predictions on the original scale.
_Avoid_: Encoder, when referring to the complete genetic prediction model.

**Distance model**:
A relationship deriving ecological distances from prepared features and
sampling-unit locations without genetic observations. Effective resistance and
squared embedding distance are supported forms of ecological distance; its `predict`
returns ecological distances.
_Avoid_: Genetic predictor, when the relationship has no genetic calibration.

**Conductance surface**:
A spatial grid of positive values describing how readily a landscape permits movement for one species, from which a conductance graph is formed.

**Embedding**:
A vector representation of a sampling unit's landscape features whose squared Euclidean distance from another such vector is their ecological distance.
_Avoid_: Squared separation, when referring to squared embedding distance.

**Pooled backbone**:
Landscape feature learning with one set of parameters estimated jointly across
regions and species.
_Avoid_: Shared encoder, when describing this pooling policy.

**Distance model pooled across regions**:
A species' landscape-to-distance relationship whose parameters are estimated
jointly from its observations across regions.
_Avoid_: Shared distance model, without specifying the pooling axis.

**Calibration module**:
The authoritative target contract and relationship mapping ecological distances to
genetic predictions. Direct calibration applies the inverse target
transform. Maximum-likelihood population effects (MLPE) calibration selects a
fitted relationship for the query region and species.

**MLPE calibration**:
One region–species group's fitted relationship between ecological distances
and genetic targets, including population dependence, uncertainty and
observation-use provenance.
_Avoid_: Regional MLPE calibration; complete model, when referring only to this
fitted relationship.

**Unpooled genetic calibration**:
Calibration coefficients and variance parameters estimated separately for each
region–species group, retaining population dependence within each MLPE calibration.
_Avoid_: Regional calibration, when describing the across-group pooling policy.

**Resistance calculation**:
Effective resistance between sampling-unit nodes in a conductance graph, computed
by `effective_resistance`. `ResistanceSolverConfig` declares numerical settings;
`ResistanceSolverContext` holds a reusable solver for one graph shape.
_Avoid_: Distance solver, when referring specifically to resistance computation.

**Isolation by distance (IBD)**:
A reference relationship between geographic separation and genetic dissimilarity, independent of landscape resistance.
_Avoid_: IBR, when referring to the geographic-distance baseline.

**Fold ensemble**:
A collection of fitted models trained using different sampling-unit partitions of the same dataset.
_Avoid_: Fold-averaged model, when referring to the collection rather than an aggregate prediction.

**Ensemble member**:
One fitted model belonging to a fold ensemble.
_Avoid_: Fold, when referring to the fitted model rather than the population partition.

**Population pair**:
An unordered pair of distinct populations within a landscape, associated with an observed or predicted genetic dissimilarity.
_Avoid_: Sample, when referring to a pair rather than an individual population or genetic specimen.

**Selected model**:
A fitted landscape genetics model retained for prediction from the evaluated training steps, chosen by the lowest validation loss when validation observations are supplied.
_Avoid_: Best model; snapshot, when referring to the selected model.

**Out-of-fold prediction**:
A genetic prediction for a query pair averaged over ensemble members eligible under the sampling-unit holdout and observation-use rules.
The default rules require both sampling units to remain unseen by each eligible ensemble member.
_Avoid_: Deployment prediction, which uses all ensemble members.

**Unseen sampling unit**:
A sampling unit whose genetic observations have not been used by a particular ensemble member for training, model selection, calibration, or support conditioning.
_Avoid_: Unseen location, when referring to genetic-observation use.

**Out-of-fold coverage**:
The count and fraction of requested query pairs with a successful genetic prediction from at least one eligible ensemble member.

**Sampling unit**:
An identified population or individual with a landscape location, linked to other sampling units by pairwise genetic observations.
_Avoid_: Population, when the sampling unit represents an individual.

**Pairwise genetic observation**:
A measured genetic relationship between two sampling units, whose meaning and scale are defined by the chosen target. Genetic dissimilarity and individual relatedness are different kinds of pairwise genetic observation.
_Avoid_: Genetic distance, when the observation is a similarity or relatedness measure.

**Region**:
A landscape and its associated sampling units and genetic observations, with its own relationship between ecological distances and observed genetic targets.
_Avoid_: Dataset, when referring to one region within a multi-region study.

**Data role**:
The permitted use of pairwise genetic observations for training, validation-based model selection, genetic calibration, support conditioning, or query evaluation.

**Query pair**:
An unordered pair of distinct sampling units for which a prediction is requested.
Like a query in meta-learning, it identifies what to predict; its measured target, when available for evaluation, is used afterward to score the prediction.

**Support observations**:
Pairwise genetic observations used to condition population effects for predictions on separate query pairs.
Like the support set in meta-learning, they supply measurements for adaptation before predicting queries; here the distance model and fitted calibration coefficients and variance parameters stay fixed.

**Population effect**:
A population's additive random contribution to model-scale genetic targets in MLPE calibration, shared by every pair involving that population.

**Original scale**:
The measurement scale of pairwise genetic observations, in the target's units, on which genetic predictions are returned.

**Model scale**:
The scale of genetic targets after the chosen target transform, on which genetic calibration is fitted.
It coincides with the original scale when the transform is the identity.

**Ecological distance**:
A distance model's pairwise separation between two sampling units based on their landscape, before any genetic calibration. Effective resistance through a conductance graph is the main example.
_Avoid_: Raw distance; landscape score; genetic prediction, before calibration to the target scale.

**Genetic prediction**:
An estimated genetic relationship for an identified pair of sampling units in a
region, expressed in the target's original units.
A prediction matrix arranges values for the requested pairs.
_Avoid_: Calibrated distance.

**Genetic calibration**:
The relationship between ecological distances and genetic observations on the target scale. This relationship may differ between regions.

**Species order**:
An ordered sequence of species names retained by a landscape genetics model to associate species with their parameters and predictions.
_Avoid_: Registry, when referring to species order.
