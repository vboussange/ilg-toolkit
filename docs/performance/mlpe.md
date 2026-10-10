# Exact MLPE endpoint-system arithmetic

Standalone calibration and joint training share JAX full-ML likelihood, signed
GLS and population-effect posterior kernels. For $m$ observed pairs and $n$
sampling units, the incidence matrix $Z$ has two ones per row. With population
variance $u$ and declared residual variance $d=e+j$,

$$V=dI+uZZ^\top,\quad G=Z^\top Z,\quad K=I+(u/d)G.$$

Endpoint scatters build $G$ without allocating $Z$ or a pair-sized covariance.
The determinant identity gives $\log\det V=m\log d+\log\det K$. For any columns
$B$, compute $A=(u/d)K^{-1}Z^\top B$ and $R=B-ZA$ with solves and endpoint gathers.
Then

$$B^\top V^{-1}B=R^\top R/d+A^\top A/u.$$

This positive representation avoids cancellation between large nearly equal
Gram matrices. It gives the two-column signed GLS system and residual quadratic.
For fitted residuals, $A$ is the effect posterior mean, $uK^{-1}$ its covariance,
and $\operatorname{chol}(K)/\sqrt u$ its precision factor. All operations use
linear solves. Ordinary storage is $O(n^2+m)$ and factorization costs $O(n^3)$,
compared with $O(m^2)$ storage and $O(m^3)$ dense Cholesky. For complete pairs,
$m=n(n-1)/2$; the encoder and graph solver may still dominate total training time.

## Numerical limits

Input dtypes control the pure kernels' precision. Standalone calibration and
prediction scope float64 locally, preserving the caller's global JAX setting;
joint MLPE training requires explicitly enabled float64. Neither importing the
package nor preparing unrelated inputs changes global precision.

The conservative supported bound is

$$\epsilon_{\rm machine}\left(1+2\max_i\deg(i)\frac{u}{e+j}\right)\le0.01.$$

A Gershgorin bound controls roundoff in $K$, including incomplete or bipartite
endpoint graphs. Unsupported ratios, failed factors, invalid scores and singular
or unidentified profiles return nonfinite kernel diagnostics, which public fit
boundaries reject. Failure never changes a variance or increases jitter. A frozen
likelihood can score one query pair; variance fitting needs overlapping endpoints
and at least three observed pairs with a nonsingular score design.

A boolean `pair_mask` excludes false rows from moments, determinant counts and
quadratics in eager and JIT execution, even with nonfinite values or placeholder
endpoints. Independent tests retain dense Gaussian, frozen R/lme4, finite-
difference and high-precision gradient references; dense arithmetic is confined
to test/reference machinery. [Statistical guidance](../statistics.md) gives the
observation model and distinguishes converged standalone calibration from
fixed-budget joint fitting.

## Optional measurement

```bash
PYTHONPATH=src python benchmarks/benchmark_mlpe.py \
  --sizes 12 24 48 --repeats 5 --output benchmarks/results/mlpe.json
```

The small experiment invokes the maintained public JAX likelihood on CPU with
float64 inputs. Each size runs in a fresh process. Compilation and one warm call
precede synchronized timed calls; first-call time is reported separately. Whole-
process peak RSS includes imports, compilation, native workspace and allocator
behavior; it does not measure individual buffers or accelerator peak memory.
Results belong in the ignored `benchmarks/results/` directory. There are no
historical timings or wall-clock correctness thresholds for the revised backend.
