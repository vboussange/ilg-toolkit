# ILG Toolkit

Inverse landscape genetics: learning relationships between landscapes and pairwise genetic observations among sampled populations or individuals.

## Language

**Region batch**:
The prepared landscape features and sampling-unit locations for one region, independent of genetic observations.
_Avoid_: Prepared region, when naming this input boundary.

**Training configuration**:
The explicit optimization budget and numerical settings supplied as `TrainingConfig`.
_Avoid_: FitConfig, when naming this public configuration.

**Calibrated model**:
A fitted landscape encoder together with the target-scale relationship and regional calibration needed for genetic prediction. Direct regression uses its declared target transform; MLPE uses explicit regional genetic calibration.
The public API names this boundary `CalibratedModel`.
_Avoid_: Predictor, when naming this fitted model boundary.

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
A landscape and its associated sampling units and genetic observations, with its own relationship between landscape-derived scores and observed genetic targets.
_Avoid_: Dataset, when referring to one region within a multi-region study.

**Landscape score**:
A model-derived measure of separation between two sampling units based on their landscape, before any genetic calibration.
_Avoid_: Genetic prediction, when the score has not been related to the genetic target scale.

**Genetic calibration**:
The relationship between landscape scores and genetic observations on the declared target scale. This relationship may differ between regions.
