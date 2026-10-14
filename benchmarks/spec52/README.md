# Matched runtime protocol for native multispecies workflows

This protocol is frozen before implementing issue #52. Main is
`8439f99ea37f04273a1868b8009e1076239ad316`. `protocol.json` fixes all workloads,
numerical settings, repetitions, and acceptance thresholds before any PR timing.
This is a hardware-specific, fixed-budget comparison, outside correctness CI.
It makes no time-to-convergence claim and imposes no universal GPU limit.

## Workload equivalence

The eight factorial cases cross ResNet/U-Net, direct/MLPE, and one/two regions.
Two feature-heavy cases increase raster channels and native patch pixels; the
resistance stress case has 256 graph nodes and 1,128 pairs; the MLPE stress case
has 128 sampling units and 8,128 pairs. These are modest reproducible CPU workloads,
not a claim to represent every production raster or GPU. Every pair among each
region's distinct sampled patch cells is observed and predicted. Features, target
values, positions, labels and hashes are fixed by `fixtures.py`, independently of
either library API. Region identities have distinct sampling-unit labels.

The matched PR run must use one species, all regions on every optimizer step,
all species selected, and the same patch/species execution settings where they
apply. Each group contributes its complete pair set to all three Adam updates.
Use full evaluation at initialization and after every update (four evaluations),
and perform final calibration for every group. Main refreshes complete MLPE
calibrations at each evaluation; count this work in end-to-end fit time. Record
extra final calibration work in the PR rather than subtracting its cost. Do not
compare a sparse-evaluation or species-subsampled PR run against main as if it did
the same work. A future multispecies scaling report must separately record each
species' update exposure, selected groups, region/species chunk bounds, and all
calibration work. It cannot claim paired speedup over a one-species baseline.

Neural initial parameter values can change with the native architecture, so this
protocol requires identical seeds, architecture capacity/settings, data, solver
accuracy and scientific eligibility, not identical fitted parameter arrays or
convergence rates. A changed parameter count must be reported. U-Net dropout is
0.1 and therefore the backward pass includes stochastic training behavior.

## Timing and memory

Each case runs in a fresh child process with persistent JAX compilation caching
disabled, two allowed CPU cores, one requested BLAS/OpenMP thread, float64 enabled,
and CPU selected before importing JAX. The environment report includes actual
software versions, device, CPU affinity, source-tree identity, thread/allocator
variables and host memory. Revisions run in separate worktrees; the baseline
worker refuses any production source tree other than frozen main. A shared
read-only dependency environment is acceptable only with the exact recorded
package versions and a revision-specific `PYTHONPATH`. No production adapter or
artifact reader is introduced for this benchmark.

Input preparation and initial model construction are reported separately. The
first synchronized public fit includes tracing, compilation, initialization,
three updates, four evaluations, final calibrations and host overhead; it is
explicitly **compile plus first fit**, not compiler time alone. Synchronized
callback intervals additionally expose initialization and the first update plus
evaluation. A revision-specific timer around the baseline checked-call boundary
measures each training step from its first regional gradient through the
synchronized Adam update, including host accumulation and excluding evaluation
and calibration. This benchmark-only instrumentation does not change production
functions, results, solver settings or the public API. It is deliberately tied
to the frozen revision; the PR driver must measure the same step scope. After one complete untimed warm fit, three complete fits from the
same initial model provide warmed samples. Prediction requests the same full
pair set and synchronizes results. Its first call and five warmed calls are
reported separately. No lazy-device dispatch is timed without synchronization.

Report raw samples, median, range, sample standard deviation, and seeded 95%
bootstrap intervals for warmed medians. One cold fit per case is a descriptive
observation with no uncertainty estimate; rerun fresh processes if a cold-time
regression is near its threshold. Whole-process high-water RSS includes imports,
compiler state, repeated fits, and allocator caches; it is not live tensor size,
backward-only memory, or GPU memory. Device allocator statistics, where exposed,
are reported separately and are null when unsupported. Run revisions sequentially
on the same host/affinity under comparable load; do not compare this CPU report
against GPU results or independent hardware.

## Acceptance frozen before PR measurement

Every case must finish; histories, raw distances, predictions and learned arrays
must be finite. Raw distances must be nonnegative and symmetric with zero
diagonal within the JSON tolerances, and MLPE inputs must be nonconstant.
Conductance predictions must agree with an independent dense graph pseudoinverse
using the actual learned conductances. MLPE calibration uses all observed pairs,
positive finite variances and finite fitted coefficients. These checks are
performed outside measured prediction intervals. The final integration suite
adds independent likelihood/gradient references and new-API behavior tests.

The 1.10/0.50 requirements come from implementation issue #63. For the two
predeclared feature-heavy representative cases, complete warmed fit and warmed
training-step time (excluding evaluation/calibration) must be at most 1.10 times
main. All factorial and stress cases are also reported; they expose solver and
calibration costs rather than being silently dropped. Measure joint 1-, 2-, 3-, and
4-species fits with equal per-species update exposure. On those two feature-heavy
cases, each added-species marginal cost must be at most 0.50 of that species'
independent PR fit; report both increments and total joint time against the sum
of independent fits. Report every adjacent increment (1→2, 2→3, 3→4)
against that added species' independent fit, with its uncertainty. The future PR driver must provide the matched independent
species denominator and cannot substitute old-main timing.

Report startup ratios above 2.0 and prediction/RSS ratios above 1.5 as material
regressions; these disclosures do not replace the required 1.10/0.50 gates.
If bootstrap uncertainty crosses a timing threshold, collect more samples before
declaring a pass or failure. Bootstrap intervals with three fit samples are
descriptive and cannot establish reliable tail behavior. No speedup claim is
allowed solely from overlapping intervals or unpaired host runs. Use a seeded
10,000-resample ratio bootstrap with independent resampling of main/PR fit blocks,
increasing to at least ten fit repetitions when a required threshold is ambiguous; unresolved uncertainty
is an inconclusive gate, not a pass. This repeatability policy and the designated
representative cases must not be changed after observing PR performance.

## Reproduction

Use the main source tree with the baseline revision driver. The exact baseline
command and machine report are saved with `results/main.json`; the runner writes
completed cases incrementally so an interrupted run does not discard evidence.

```sh
PYTHONPATH=/path/to/frozen-main/src /path/to/dependencies/bin/python \
  benchmarks/spec52/main_driver.py --output benchmarks/spec52/results/main.json
```

Use `--case NAME` for an isolated case or `--list` to inspect the frozen matrix.
A PR-specific driver must translate the fixture into its native public API and
retain this protocol unchanged. It belongs beside this driver, not in `src/`.
The baseline driver is intentionally only runnable against the frozen main
source tree; it is not a compatibility test for the new API.

## Analysis policy frozen before PR measurement

`analysis_policy.json` resolves two measurement details discovered by reviewing
the main observations. It changes neither the workload nor the numeric gates.
The original `results/main.json`, protocol bytes, per-fit arrays, and measured
source hashes remain intact. All 12 measured workers used benchmark commit
`5963194`; its driver/fixture files remain retrievable from Git even though the
current driver has subsequently been formatted, type-narrowed, and given a
repetition override. `results/main-analysis.json` records its own analysis code,
policy and raw-file hashes.

A fresh public fit on main recreates the optimizer. Its first update incurs
recurrent overhead consistent with static-optimizer compilation, even after a
previous fit. The full fit measurement retains that real cost. For the warmed
training-step gate, use `steady_training_step` from the analysis report: updates
2 and 3 within every repeated fit. Report first updates separately as
`recurrent_first_update`. The historical raw report's `warm_training_step`
summary includes **all** updates and must not be used as the steady-state gate.
Every observed update remains available in `warm_training_steps_by_fit_seconds`.
Bootstrap whole fit blocks, retaining their post-first steps together; main and
PR runs have no paired repetition IDs, so their blocks are resampled independently.
The small initial repeat count remains descriptive; an uncertain gate requires
at least ten repetitions and is never an automatic pass.

`multispecies_fixtures.py` freezes the added-species fixture generation: species
zero exactly reproduces the main input, all species share each region's features,
and later species have independently seeded locations/targets with the same
unit and complete-pair counts. Use the same species fixture in its joint and
independent fit. Initialize each independent fit with the joint initialization's
exact pooled backbone and that species' output projection; verify the arrays
match before timing. A common random seed alone does not establish this when
output-array shape changes.

```sh
/path/to/dependencies/bin/python benchmarks/spec52/analyze.py \
  benchmarks/spec52/results/main.json \
  --output benchmarks/spec52/results/main-analysis.json

# Increase observations without editing frozen workloads or thresholds.
PYTHONPATH=/path/to/frozen-main/src /path/to/dependencies/bin/python \
  benchmarks/spec52/main_driver.py --case resnet-feature-heavy --fit-repeats 10 \
  --output benchmarks/spec52/results/main-resnet-feature-heavy-10.json
```

The override is recorded in the new output's effective protocol and command;
its `protocol_sha256` still identifies the immutable protocol file. Repetition
changes require fresh baseline and PR samples under comparable host conditions.
Type-check this revision-specific driver against frozen main. It intentionally
uses API names that the native multispecies revision removes, so checking it
against new production types would test the wrong revision.

## Native revision measurements and hard acceptance

`pr_driver.py` translates the frozen arrays directly to the native public API.
It requires a clean committed source and driver, records the exact revision,
and refuses to overwrite an existing result file. `--smoke` permits development
checks but marks their results ineligible for acceptance. Every worker observes
all updates, group contributions and complete evaluations; it checks requested
pairs, finite results, final calibration, and independent dense resistance
outside timed intervals. The checked-call timer starts at the first regional
training encoding and ends at synchronized Adam completion, including all
regional tails, encoder pullbacks and accumulation.
For one selected group, the same pooled encoder, reduced tail derivative and
encoder pullback execute inside one compiled call; its timer covers that entire
operation through Adam. This avoids dispatch overhead without changing the
group likelihood, sampling weight, dropout key or update count.
For multiple selected groups, encoder derivative preparation is itself cached
inside a compiled call. The tail and encoder backward retain their separate
numerical boundaries; a broader fusion experiment was rejected because it
violated the existing float32 chunk-equivalence tolerance.

Joint fits and independent species fits use exactly copied and hashed backbone
and projection-row initializations from the same four-species model. The initial
execution choice is species chunk one. Scientifically equivalent bounded chunk
optimizations may be measured in separate complete batches; preserve prior
attempts and disclose the chosen configuration, without changing frozen work or
thresholds. LogLinear uses its default bounded link and explicit frozen patch and
solver settings; its separate report has no main comparison.

```sh
PYTHONPATH=src python benchmarks/spec52/pr_driver.py --suite all --fit-repeats 3 \
  --output /tmp/pr-spec52-attempt-01.json
python benchmarks/spec52/compare.py --main benchmarks/spec52/results/main.json \
  --pr /tmp/pr-spec52-attempt-01.json --output /tmp/pr-spec52-comparison-01.json
```

The comparison exits nonzero for a failed, inconclusive or incomplete gate. It
requires all twelve single-species cases, both joint 1/2/3/4 and independent
0/1/2/3 matrices, and the separate LogLinear cases. It checks workload manifests,
initialization, source consistency and full scientific accounting before drawing
independent fit-block bootstrap samples. Inconclusive comparisons require fresh
matched batches with at least ten repetitions; retain all earlier raw evidence.
Use repeat output filenames, never rewrite the original frozen main report.
For a fresh baseline repeat, pass the original report first and the new report
second with another `--main`. A new batch of at least ten fits can supersede an
initial smaller batch only with identical source, fixture, software and protocol
identities; corresponding PR fits must also have at least ten repetitions. Two
competing full batches are rejected. The report retains both input hashes and
names every repeated baseline case.

## Completed native evidence

The [final runtime report](runtime-results-final-b2088e6.md) records all 32 workloads, the
passing required gates, diagnostic uncertainty, separate LogLinear results,
startup/prediction/RSS disclosures, and preserved failed attempts. Its
[machine-readable comparison](results/runtime-comparison-final-b2088e6.json) is reproducible
from the archived raw files without rerunning the measurements.

The [large-registry report](large-registry-memory-final-b2088e6.md) records the complete
14-case memory/compilation matrix through 1,000 species on the same production
source. It separates bounded scientific workspace from parameter/output growth
and explicitly describes the CPU allocator-profiling limitation.
