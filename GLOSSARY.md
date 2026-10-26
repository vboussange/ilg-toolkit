# ILG Toolkit

Inverse landscape genetics concerns estimating relationships between landscapes and pairwise genetic observations among sampled populations or individuals.

## Language

**Region batch**:
The prepared landscape features, sampling-unit identifiers, and grid positions for one region, independent of genetic observations.
An optional species column labels each sampling unit; when omitted, all sampling units belong to one species.
_Avoid_: Prepared region, when naming this input boundary.

**Training configuration**:
The chosen loss, joint optimization budget, and execution settings for training a distance model and its statistical parameters.
_Avoid_: Calibration settings, when referring to the training budget.

**Training**:
Joint estimation of distance-model parameters and the statistical parameters required by a loss, using supplied genetic observations.
_Avoid_: Fitting, when naming this optimization workflow.

**Loss**:
A statistical criterion relating a region–species group's ecological distances to its genetic observations, together with the criterion's numerical settings.
_Avoid_: Objective name, when referring to the criterion and its settings.

**Statistical parameter state**:
The current statistical parameters estimated jointly with distance-model parameters during training, as required by the chosen loss.
For maximum-likelihood population effects (MLPE) these are variance parameters; the direct log1p criterion has none.
_Avoid_: Fitted calibration, when referring only to the parameters being optimized.

**Landscape genetics model**:
A distance model composed with genetic calibration, retaining feature meanings,
target semantics, and species order.
Its predictions describe genetic relationships on the original scale.
_Avoid_: Encoder, when referring to the complete genetic prediction model.

**Distance model**:
A relationship deriving ecological distances from prepared features and
sampling-unit locations without genetic observations. Effective resistance and
squared embedding distance are supported forms of ecological distance.
_Avoid_: Genetic predictor, when the relationship has no genetic calibration.

**Conductance surface**:
A spatial grid of positive values describing how readily a landscape permits movement for one species, from which a conductance graph is formed.

**Embedding**:
A vector representation of a sampling unit's landscape features whose squared Euclidean distance from another such vector is their ecological distance.
_Avoid_: Squared separation, when referring to squared embedding distance.

**Pooled backbone**:
Landscape feature representation with one set of parameters estimated jointly across
regions and species.
_Avoid_: Shared encoder, when describing this pooling policy.

**Distance model pooled across regions**:
A species' landscape-to-distance relationship whose parameters are estimated
jointly from its observations across regions.
_Avoid_: Shared distance model, without specifying the pooling axis.

**Calibration module**:
A fitted genetic relationship and its target contract, mapping ecological distances to genetic predictions.
Direct calibration applies the inverse target transform; MLPE calibration retains a fitted relationship for each supplied region–species group.

**Fitted calibration**:
A genetic relationship estimated for supplied observations and a particular distance model.
For MLPE it includes distance moments, coefficients, variance estimates, and a population-effect posterior.
_Avoid_: Recalibrated distance model, when its distance parameters have stayed fixed.

**MLPE calibration**:
One region–species group's fitted relationship between ecological distances
and genetic targets, including population dependence and population-effect uncertainty.
_Avoid_: Regional MLPE calibration; complete model, when referring only to this
fitted relationship.

**Unpooled genetic calibration**:
Calibration coefficients and variance parameters estimated separately for each
region–species group, retaining population dependence within each MLPE calibration.
_Avoid_: Regional calibration, when describing the across-group pooling policy.

**Resistance calculation**:
Effective resistance between sampling-unit nodes in a conductance graph.
_Avoid_: Distance solver, when referring specifically to resistance computation.

**Isolation by distance (IBD)**:
A reference relationship between geographic separation and genetic dissimilarity, independent of landscape resistance.
_Avoid_: IBR, when referring to the geographic-distance baseline.

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
A fitted landscape genetics model retained for prediction from the evaluated training steps, chosen by the lowest validation loss when validation observations are supplied.
Without built-in validation, the final evaluated model is selected; a study's own candidate-selection procedure may retain a different model.
_Avoid_: Best model; snapshot, when referring to the selected model.

**Latest model**:
The fitted landscape genetics model corresponding to the current training step's distance and statistical parameters.
_Avoid_: Selected model, when referring to the state used for continued training.

**Explanatory evaluation**:
Assessment of the association between frozen ecological distances and genetic observations through fresh calibration on the assessed populations.
When those observations fit and score calibration, the resulting score measures association rather than untouched predictive accuracy of the complete model.

**Sampling unit**:
An identified population or individual with a landscape location, linked to other sampling units by pairwise genetic observations.
_Avoid_: Population, when the sampling unit represents an individual.

**Pairwise genetic observation**:
A measured genetic relationship between two sampling units, whose meaning and scale are defined by the chosen target. Genetic dissimilarity and individual relatedness are different kinds of pairwise genetic observation.
_Avoid_: Genetic distance, when the observation is a similarity or relatedness measure.

**Region**:
A landscape and its associated sampling units and genetic observations, with its own relationship between ecological distances and observed genetic targets.
_Avoid_: Dataset, when referring to one region within a multi-region study.

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
A genetic prediction using the population-effect posterior retained by fitted calibration.
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
