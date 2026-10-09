# Training checkpoints

`save_checkpoint(path, state)` saves a complete continuation state in one atomic
archive. `load_checkpoint(path)` restores it without a model template or research
checkout. These functions preserve the latest encoder, MLPE nuisance parameters,
Adam moments and count, random key, completed history, fixed learning-rate and
stopping policies, and the separately selected best encoder and regional heads.

```python
from dataclasses import replace
from ilg_toolkit import fit, load_checkpoint, save_checkpoint

result = fit(
    region, observations, model=model, config=config,
    on_epoch=lambda state: save_checkpoint("training.ilg", state),
)

state = load_checkpoint("training.ilg")
continued = fit(
    region, observations, state=state,
    config=replace(state.config, epochs=100),
)
```

Supply the same training/validation inputs and partitions when resuming. Only
the total epoch budget may increase or stay equal; reducing a saved requested
budget is rejected, including for an intermediate callback checkpoint. Changed
landscapes, selected measurements, target metadata, feature meaning/order,
sampling-unit identities/locations/kinds, partitions, model overrides, and
relevant optimization/solver configuration are rejected. Reordering equivalent
region mappings preserves identity. The default budget is the saved configuration.

The callback runs after initialization (epoch zero) and after every complete
update, evaluation and predictor selection. It receives an internally consistent
`TrainingState`; exceptions propagate and stop fitting. A zero-budget fresh fit
therefore invokes it once. A continuation requesting no further updates invokes
it zero times and retains the selected predictor. Saving only at the end is also
supported: `save_checkpoint(path, result.state)`.

MLPE fitting and checkpoint loading require the same explicit JAX float64
setting used when saving, for example within `with jax.enable_x64():`. Exact
continuation requires matching recorded Python and numerical library versions,
backend, device kind/count and precision settings, under the same deterministic
execution conditions. The loader refuses an incompatible runtime instead of
silently changing the trajectory. Solver contexts are rebuilt from configuration.

The checkpoint and inference-artifact loaders accept distinct artifact kinds.
A checkpoint includes the latest optimization state even when validation selected
an earlier predictor; inference export alone cannot resume training. Current
checkpoints support the shipped stateless U-Net and ResNet encoders. Unknown
custom architectures and mutable model state require explicit future codecs.

The versioned archive contains strict JSON metadata and numeric array payloads,
without pickle or imported user classes. Malformed, incomplete or incompatible
content raises `ArtifactError`. Saving writes a complete temporary archive beside
the destination and atomically replaces it after flushing. Serialization or
replacement failure preserves the previous complete generation, including its
latest and best state together.
