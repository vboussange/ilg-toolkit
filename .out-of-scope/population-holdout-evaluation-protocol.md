# Population-holdout evaluation protocol

ilg-toolkit provides what population-holdout evaluation needs, and no protocol for it.
`PairwiseObservations.select_pairs` withholds a fold's sampling units, `train` fits a
landscape genetics model on the rest, and the model predicts the held-out pairs. How a
study forms its folds, which predictions it scores, which metrics and null models it
compares, and how it reports uncertainty stay with the downstream pipeline.

## Why this is out of scope

ilg-toolkit is a minimal engine for training distance models on pairwise genetic
observations. An evaluation protocol is a study's scientific design, and it changes with
the study:

- Under MLPE, a held-out sampling unit shares one population effect across all its
  pairs. Whether a pipeline scores marginal predictions, conditions on the calibrated
  effects of the other units, or reports a contrast error is a choice about what the
  study claims, not a numerical method.
- The folds depend on the sampling design: one population at a time, or clusters of
  nearby populations held out together.
- Null models, held-out metrics and uncertainty reports, such as a jackknife over folds
  or a Laplace approximation, decide what a study can report, and each study fixes them
  before its runs.

A protocol inside the engine would make one study's choices the default for every user.
The onebionet-t data assimilation pipeline holds its own protocol, in its evaluation
module and its workflow. The reference pipeline notebook calls the engine's pieces to
show the API.

## Prior requests

- #139: "Reference pipeline: leave-one-population-out evaluation of a penalized learned
  resistance surface"
