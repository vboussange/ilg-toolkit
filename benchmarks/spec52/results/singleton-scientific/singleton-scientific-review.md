# Independent read-only singleton-gradient review

Reviewed the provisional scalar fallback and current `execution.single_group_gradient` / `_region_gradient` code in `/tmp/ilg-spec52/issue-63`, plus the public-fit script `singleton_regression.py` and saved before/after/fused-pooled arrays. No source edits, JAX execution, or heavy tests were performed. NumPy comparisons are recorded in `singleton-independent-numpy-review.json`.

## Scientific contract

I found no changed loss, coverage weight, key policy, nuisance row, or species-memory bound in the current singleton wrapper. It obtains the same pooled encoding and encoder VJP with `inference=False` and the unchanged region key, wraps the same numeric payload into a one-element chunk, calls the same reduced chunk objective/gradient, and applies that single feature cotangent through the encoder pullback. The payload retains its previously calculated inclusion-corrected weight; no realized-batch normalization or extra PRNG split appears. `group_index` still selects the same row of the full nuisance array. Direct optimization returns a model-shaped gradient; MLPE returns `(model_gradient, raw_gradient)`, preserving the Adam tree.

The earlier scalar fallback is algebraically equivalent as well: direct `weight * mean(log1p error squared)`; MLPE `weight * profiled_nll / pair_count` on complete same-group distances/targets with unchanged standardization, signed GLS, raw variance row, jitter and floor. Its arithmetic association can differ from the reduced-vmap expression, and fusing either formulation removes compiler boundaries. Algebraic equivalence does not imply bitwise optimizer trajectories in float32.

The new branch is limited to exactly one selected group in a region and `jit=True`. Thus it cannot create several simultaneous species surfaces/solves/posteriors; it holds one regional encoder residual set and one complete group tail, with ordinary full-registry parameter/nuisance gradient storage. It retains the model's existing encoder/patch checkpoint function. Four-selected-group memory probes continue to use the unchanged bounded chunk route. Full evaluation/calibration and the global single Adam update remain outside this dispatch change.

## Identifiable floating-point source

The public differential fixture is a tiny U-Net with base channels 2, then 4 and 8, and GroupNorm with one channel per group. Each convolution/patch-linear bias immediately followed by that normalization is mathematically a translation-invariant direction: subtracting the same channel's spatial mean cancels the bias. The final species embedding bias also cancels in pair differences. In float32, derivative reductions can leave gradients near zero with either sign; compiler fusion/reassociation can change those residuals.

The arrays strongly support Adam amplification of this effect. The largest direct-model parameter difference, **0.00210306**, is `patch_embedding.bias` (parameter leaf 1). At its largest-drift coordinate, the saved first moment is `-1.52737e-10` before versus `7.30343e-9` after, and second moment `4.57717e-19` versus `2.88915e-18`. `decoder1.conv2.bias` differs by **0.00175508** with first moments `9.98378e-9` versus `-5.96046e-9`. Those tiny signed gradients are large relative to Adam's small denominator/epsilon and can create learning-rate-scale parameter displacements even along an exact mathematical invariance.

Across all model gradient moments after three updates:

| Objective | Maximum parameter difference | Maximum first-moment difference | Maximum second-moment difference | Latest nuisance difference |
| --- | ---: | ---: | ---: | ---: |
| direct_log1p | 0.00210306 | 5.64731e-7 | 5.58794e-9 | not applicable |
| MLPE | 0.00157092 | 1.29361e-6 | 5.58794e-9 | 2.22045e-16 |

RNG arrays match exactly. The largest recorded training-history difference is about `1.13e-7`; both paths report exact within-path interruption/resume. The current fused-pooled artifact has exactly the scalar variant's model parameter arrays for both objectives. Their remaining difference is only in MLPE nuisance optimizer moments, at most `3.47e-18`. Thus simply expressing the former cut/VJP inside an outer JIT does not preserve the old rounding boundary, even though it preserves the mathematical chain rule.

Do not claim every changed parameter is an exact null direction. Outside the identified bias leaves, `merge2.weight` changes by up to **1.29704e-4** (direct) and **5.53243e-5** (MLPE). At those coordinates final moment differences are small (`~3e-11` direct; `~1e-11` MLPE), compatible with accumulated rounding/near-null early updates, but the saved final arrays alone do not prove every intermediate cause. All remaining non-null leaves differ by at most roughly `1.09e-6` direct / `8.35e-7` MLPE.

## Assessment and follow-up

This looks like compiler-boundary rounding amplified by Adam, not an identifiable change in the scientific objective or species weighting. The current wrapper is a clearer auditable formulation because it directly reuses the established reduced group loss and encoder VJP. Retain the independent moment/objective scientific tests, exact resume checks within the new path, and selected numeric-ID reuse tests. Add/inspect public raw-distance and calibrated-prediction comparisons before calling the non-null drift immaterial; a small aggregate loss difference alone is not a complete prediction comparison. Compare numerical results with justified float32 tolerances, rather than changing optimizer epsilon, zeroing gradients, or removing biases to force old parameter equality.

Artifact SHA256 at review:

- before NPZ: `ee26805284e545ccc532891b6e7331449ed4df9b305305eed0e152f7be3b5317`
- scalar-after NPZ: `5a7e66cf4476b11be7bbf8450a94e2dc0016bc3ffd2fdf97ce68426f320c14c8`
- fused-pooled NPZ: `da3031e5b03656264f5dd20b567550fe2274f6541f870b779004f69eb0f59f9b`
