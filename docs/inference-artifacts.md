# Saving a fitted model

```python
from ilg_toolkit import load_model, save_model

save_model("fitted.ilg", result.model)
model = load_model("fitted.ilg")
prediction = model.predict(prepared_query_region)
```

Loading requires no training data, query genetic observations or caller model
instance. The prepared query region still supplies landscape features and labelled
locations with the saved feature meanings/order. Its preparation contract is HWC
features and native row/column grid locations; feature normalization remains the
caller's preparation step. Predictions retain declared target units and transforms.

The artifact preserves the shipped encoder architecture, parameters and dynamic
scalar settings, solver options, regional MLPE calibrations and population-effect
posteriors, and encoder training/validation and calibration target-access identities.
Reloaded MLPE models support the same explicit known-effect and supplied-support
operations described in [MLPE conditioning](mlpe-conditioning.md). Calibration is
never recomputed during loading. Joint Adam heads retain their fixed-budget
`converged=False` diagnostic rather than becoming standalone converged fits.

An inference artifact can retain calibration for only selected regions,
including a newly calibrated transfer region. Label-free landscape scoring still
works in other compatible regions; target prediction there requires an explicit
regional head. Training checkpoints require calibration for every training region.

The current codecs support the exact shipped `UNetEmbeddingDistance` and
`ResNet9Conductance` classes. Both use stateless GroupNorm; that absence of mutable
model state is recorded explicitly. Unsupported custom architectures, static
architecture modifications and unimplemented mutable state fail with `ArtifactError`.
No arbitrary class imports or pickle fallback are used.

Format version 1 is one ZIP archive with a standard finite-JSON `manifest.json`
and numeric NPY entries under `arrays/`. The manifest identifies the artifact kind,
schema, runtime versions, architecture codec versions, complete model metadata
and every array's shape, dtype and SHA256 digest. Numeric loading disables pickle.
Unknown schemas, model codecs, missing/extra entries, duplicate identities and
invalid metadata fail with `ArtifactError`. This format is an inference artifact;
optimizer/RNG/progress state belongs to the distinct training checkpoint format.

Save writes a complete same-directory temporary archive, flushes it, then atomically
replaces the destination. A failed replacement leaves the preceding file intact.
Parent directories must already exist. Integrity checks detect malformed payloads;
they do not authenticate who produced an artifact.

Array dtypes are preserved. Loading 64-bit JAX model leaves requires
`JAX_ENABLE_X64=true` (or an active `jax.enable_x64()` context); the loader refuses
silent precision truncation. NumPy float64 MLPE posterior arrays retain precision
independently of this setting. Conductance scoring also retains its existing explicit
JAX x64 requirement. Runtime versions are recorded; inference reconstruction checks
the codec, leaf paths/count/shapes and dtype categories rather than promising identical
outputs across devices or future library versions. Under matching deterministic CPU
conditions, round-trip tests retain float32 model outputs exactly and float64 graph
outputs within 1e-12; the installed-wheel subprocess comparison uses rtol=1e-6,
atol=1e-7.
