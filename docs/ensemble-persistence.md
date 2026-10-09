# Portable ensembles and durable ensemble runs

`save_ensemble(path, ensemble)` writes a portable inference artifact containing
every requested member's identity, population fold, outcome, and completed
deployment predictor. `load_ensemble(path)` restores the full composition without
the training observations, a model template, or the research checkout. Regional
calibration, posterior effects, target-access provenance, target transforms,
feature contracts, and model constructor settings travel with each predictor.
The saved authority is `member.predictor`, including any explicit recalibration.
An inference reload has `member.fit_result is None`.

```python
from ilg_toolkit import load_ensemble, save_ensemble

save_ensemble("ensemble.ilg", ensemble)
restored = load_ensemble("ensemble.ilg")
prediction = restored.predict(region)
```

Member predictions and aggregate predictions retain the original target scale.
The member spread remains descriptive population SD, not a confidence interval.
Numeric arrays round-trip exactly; the tested deterministic CPU runtime also
reproduces predictions exactly. Inference on another compatible numerical
runtime inherits the single-predictor numerical behavior. Failed and pending
members remain explicit; prediction refuses to silently remove them.

`fit_ensemble_run` manages a sequential training run in a directory. It uses the
same independent member fitter, single-fit checkpoints and predictor archives
as the in-memory workflows.

```python
import jax
from ilg_toolkit import FitConfig, fit_ensemble_run, save_ensemble

with jax.enable_x64():
    ensemble = fit_ensemble_run(
        "training-run", region, observations,
        n_folds=3, holdout_size=2, fold_seed=41,
        initialization_seeds=(11, 29),
        config=FitConfig(objective="mlpe", epochs=30),
    )
    # After an interruption, omitted options use the saved composition/config.
    ensemble = fit_ensemble_run(
        "training-run", region, observations, resume=True,
    )
    save_ensemble("ensemble.ilg", ensemble)
```

Use one writer per run directory. A fresh call refuses an existing manifest.
Resume first validates every saved member and the complete requested
composition, then skips compatible members already at the requested budget.
Unfinished members continue their own latest encoder, regional nuisance
parameters, optimizer, random key, history, and separately selected best
predictor. A larger explicit total epoch budget continues previously completed
members from their full checkpoints. Reducing the saved requested budget or
changing any other fit/solver configuration is rejected. The initialization
factory runs only for pending members; it cannot replace a saved encoder.
New members must match the saved model family, constructor settings, leaf
shapes/dtypes and dynamic architecture scalars, including dropout settings.

Training and validation measurements, partitions, region identities, landscape
features, sampling-unit locations/kinds and target/feature contracts must agree.
The original fold definitions and initialization identities must agree too.
Unaccessed query measurements do not determine a training checkpoint's identity.
The workflow retains each fold's declared query/validation roles; evaluation
must still establish eligibility from actual target access.

Each complete epoch is saved in a new immutable member generation before an
atomic replacement of `run.ilg` publishes its reference. The reference includes
an integrity digest and artifact kind. Completed publication also references
the selected inference predictor, which is checked against the checkpoint's
best predictor. A failed publication preserves the preceding manifest and
referenced generations. Unreferenced generations may remain after interruption;
they are never guessed or automatically substituted. Paths are generated from
member identity hashes, restricted to the run's `members` directory, and checked
against traversal and symlink escapes.

`on_epoch(identity, state)` runs after durable epoch publication.
`on_member(member)` runs after durable outcome publication, including skipped
members. Callback exceptions stop the run and propagate. Initialization or fit
failures are saved with the member identity and diagnostics; later requested
members still run. Resume retains those failures explicitly instead of retrying
them with different parameters. Start a separate run to retry a changed request.
Missing, corrupt, partial or incompatible referenced artifacts raise
`ArtifactError` with the affected member identity before training resumes.

Run manifests and portable ensembles are distinct artifact kinds. The portable
artifact is self-contained for inference; the run directory must retain all
referenced generations for continuation. Both use the common strict JSON and
numeric-array archive format without pickle or imported user classes. Supported
model families and exact-resume runtime/precision requirements are those of
[predictor persistence](../README.md) and
[training checkpoints](checkpoints.md).
