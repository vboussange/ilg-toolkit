# ILG Toolkit

Inverse landscape genetics: learning relationships between landscapes and pairwise genetic observations among sampled populations or individuals.

## Language

**Region batch**:
The prepared landscape features and species-specific sampling-unit locations for one region, independent of genetic observations.
_Avoid_: Prepared region, when naming this input boundary.

**Training configuration**:
The explicit optimization budget, group-sampling policy and numerical settings for a fit.
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

**Calibration module**:
The authoritative target contract and relationship mapping ecological distances to
genetic predictions. Direct calibration applies the declared inverse target
transform. MLPE calibration selects a fitted relationship for the query region and species.

**MLPE calibration**:
One region–species group's fitted relationship between ecological distances and genetic targets, including population dependence, uncertainty and observation-use provenance.
_Avoid_: Complete model, when referring only to one group relationship.

**Resistance calculation**:
Effective resistance between sampling-unit nodes in a conductance graph under declared numerical settings.
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
An unordered pair of distinct populations of one species within a region, associated with an observed or predicted genetic dissimilarity.
_Avoid_: Sample, when referring to a pair rather than an individual population or genetic specimen.

**Sampling unit**:
An identified population or individual of one species associated with a landscape location and serving as an observation endpoint. Its biological identity combines species and label, and it belongs to one region within that species.
_Avoid_: Population, when an endpoint represents an individual.

**Pairwise genetic observation**:
A measured genetic relationship between two sampling units, whose meaning and scale are defined by the chosen target. Genetic dissimilarity and individual relatedness are different kinds of pairwise genetic observation.
_Avoid_: Genetic distance, when the observation is a similarity or relatedness measure.

**Region**:
A geographic domain with prepared landscape features, sampling units and a computational boundary. Each observed species in the region has its own genetic calibration.
_Avoid_: Dataset, when referring to one region within a multi-region study.

**Ecological distance**:
A distance model's pairwise separation between two sampling units based on their landscape, before any genetic calibration. Effective resistance through a conductance graph is the main example.
_Avoid_: Landscape score; genetic prediction, before calibration to the target scale.

**Genetic prediction**:
An estimated genetic relationship for an identified pair of sampling units in a
region and species, expressed in the declared target's original units. Its matrix representation
does not imply predictions for unrequested pairs or a joint uncertainty model.

**Genetic calibration**:
The relationship between ecological distances and genetic observations on the declared target scale. This relationship may differ between region–species groups.

**Species registry**:
The fixed ordered set of named species represented by a distance model. Species identity binds ecological responses and predictions to their biological meaning.

**Region–species group**:
One species' observations and sampling units in one region. Missing geographic coverage does not create a group or imply zero observations.

**Pooled backbone**:
A neural landscape-feature relationship learned jointly across the model's species and regions.
_Avoid_: Frozen backbone, when pooled parameters remain trainable; partial pooling, for ordinary shared deterministic parameters.

**Species-specific projection**:
A species' learned relationship from pooled neural features to its conductance or embedding, pooled across that species' regions.

**Unpooled genetic calibration**:
Independent fitted genetic relationships for the observed region–species groups, with no cross-group shrinkage or population-effect covariance. Population random effects within one group do not establish pooling between groups.

**Selected model**:
The fully evaluated model chosen for prediction, with genetic calibrations belonging to that same snapshot. With validation it may precede the current optimizer progress.
