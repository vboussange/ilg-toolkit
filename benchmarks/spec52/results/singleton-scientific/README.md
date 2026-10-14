# Scientific diagnostics for the scoped performance fixes

`singleton_reference.py` runs public fits with four species, one sampled group,
three updates, sparse full evaluation, dropout, both direct and complete MLPE
objectives, and exact interrupted/resumed state equality. The final public
before/after NPZ files include all parameters, Adam moments, nuisance state,
selected model/calibration, RNG, raw distances and calibrated predictions.

- Before: clean `06812b760` worktree; its production source tree
  `a3c07b8e868bc7a2cba735d06f2e20053a038aa1` is identical to the initial measured
  #63 commit `3d3a4c1`.
- After: clean `ef041936328ab3bd7b6955812d3334fecc02e3e4`, production source tree
  `fe2dbac8be657e46187e75aa04a5b7a80c68f6c9`.
- `singleton-public-comparison.json`: all 28 raw distance, calibrated prediction
  and selected MLPE posterior comparisons pass unchanged `rtol=2e-5, atol=1e-7`.
  Maximum raw error is 8.35e-7; calibrated prediction error is 1.19e-6; posterior
  error is 2.82e-7. Both implementations resume exactly within their own path.

The independent review predates these final public comparisons. It records the
initial diagnostic arrays (`before`, scalar `after`, `fused-pooled`) and correctly
requires observable checks rather than assuming every parameter drift is harmless.
The largest float32 bias drift is an Adam-amplified near-zero gradient along a
GroupNorm translation-invariant direction; some non-bias drift also exists.
Do not interpret the accepted fusion as producing bit-identical old parameters.
The `fused-barriers` arrays preserve an unsuccessful rounding-boundary experiment.
No optimizer epsilon or scientific tolerance was changed.

Reproduce the public fit arrays with each source revision on `PYTHONPATH`, CPU JAX
with x64 enabled, and `python benchmarks/spec52/singleton_reference.py OUTPUT_PREFIX`.
The later `tests/test_pooled_training_reference.py` independently checks full
likelihood directional gradients, every Adam moment/update identity, exact resume
and chunk equivalence against public NumPy scientific references. It covers one
and two species, direct and MLPE objectives.

A broader multi-species chunk fusion was subsequently rejected because an existing
float32 dropout chunk-equivalence test failed unchanged tolerances. The retained
`one-chunk-fusion-rejected.patch` reproduces that diagnostic when applied to
`ef041936`; run `pytest tests/test_bounded_training.py -k unet_reuses`.
The accepted `af4cf33` change instead caches encoder-VJP preparation while keeping
tail and encoder-backward numerical boundaries separate. That failing test and the
full bounded/reference suite pass unchanged. No accepted runtime uses the rejected
broader fusion. `sha256.json` retains the original diagnostic artifact bytes.
