# ILG Toolkit

Inverse landscape genetics: learning relationships between landscapes and pairwise genetic observations among sampled populations or individuals.

## Language

**Region batch**:
The prepared landscape features and sampling-unit locations for one region, independent of genetic observations.
_Avoid_: Prepared region, when naming this input boundary.

**Training configuration**:
The explicit optimization budget and numerical settings supplied as `TrainingConfig`.
_Avoid_: FitConfig, when naming this public configuration.

**Landscape genetics model**:
A distance model composed with genetic calibration, retaining feature meanings,
target semantics and the observations used for training and selection.
_Avoid_: Encoder, when referring to the complete genetic prediction model.

**Distance model**:
A relationship deriving ecological distances from prepared features and
sampling-unit locations without genetic observations. Effective resistance and
squared embedding distance are supported forms of ecological distance.
_Avoid_: Genetic predictor, when the relationship has no genetic calibration.

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
genetic predictions. Direct calibration applies the declared inverse target
transform. MLPE calibration selects a fitted relationship for the query region
and species.

**MLPE calibration**:
One region-and-species group's fitted relationship between ecological distances
and genetic targets, including population dependence, uncertainty and
observation-use provenance.
_Avoid_: Regional MLPE calibration; complete model, when referring only to this
fitted relationship.

**Unpooled genetic calibration**:
Separate calibration coefficients and variance parameters for each region-and-species
group, without parameter sharing or hierarchical shrinkage across those groups.
Population dependence within each MLPE calibration is retained.
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

**Sampling unit**:
An identified population or individual associated with a location in a landscape and serving as an endpoint of genetic observations.
_Avoid_: Population, when an endpoint represents an individual.

**Pairwise genetic observation**:
A measured genetic relationship between two sampling units, whose meaning and scale are defined by the chosen target. Genetic dissimilarity and individual relatedness are different kinds of pairwise genetic observation.
_Avoid_: Genetic distance, when the observation is a similarity or relatedness measure.

**Region**:
A landscape and its associated sampling units and genetic observations, with its own relationship between ecological distances and observed genetic targets.
_Avoid_: Dataset, when referring to one region within a multi-region study.

**Ecological distance**:
A distance model's pairwise separation between two sampling units based on their landscape, before any genetic calibration. Effective resistance through a conductance graph is the main example.
_Avoid_: Landscape score; genetic prediction, before calibration to the target scale.

**Genetic prediction**:
An estimated genetic relationship for an identified pair of sampling units in a
region, expressed in the declared target's original units. Its matrix representation
does not imply predictions for unrequested pairs or a joint uncertainty model.

**Genetic calibration**:
The relationship between ecological distances and genetic observations on the declared target scale. This relationship may differ between regions.
