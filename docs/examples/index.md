# Executable tutorials

Work through these notebooks in increasing complexity. Each creates its own small,
deterministic synthetic data and executes independently on CPU with float64.
Downloaded sources contain no saved outputs; rendered pages show the outputs from
the documentation build. Install the `docs` extra to run all notebook dependencies.

1. [Quickstart](quickstart.md): an accessible first analysis, fitted predictions,
   save/load, and continuation from a single-run checkpoint.
2. [Isolation by distance](ibd.md): geographic units, fitting a reference relationship,
   diagnostic plots, and the limits of the log1p comparison.
3. [Maximum-likelihood population effects (MLPE) workflow](mlpe_workflow.md): ResNet9 conductance across regions,
   population splits, fresh calibration and geographic comparison, prediction
   modes, and transfer to a separate region.
4. [Ensemble](ensemble.md): ordinary training loops, seeds, checkpoints and
   candidate selection, ordered members, predictions, and conductance summaries.
5. [Multispecies](multispecies.md): experimental labeled inputs, pooled feature
   learning, species-specific parameters, calibration, and species queries.

The examples teach workflow and interpretation. Scientific correctness is checked
separately by the package tests. See [development](../development.md) for the
execution and strict-build commands.

!!! note
    The small optimization budgets make execution practical; they do not establish
    ecological validity, statistical power, or a recommended training budget for a real study.
