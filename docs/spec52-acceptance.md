# Native multispecies acceptance trace

This records the complete #52 scope: all 62 user stories, all 11 implementation
sections, all 17 testing decisions and the added runtime gate. Tickets #53–#64 were
independently integrated/verified before this final acceptance slice. The final
source, examples, experimental consumer and release fixture use the native API;
no compatibility reader, alias, wrapper or adapter was restored.

Production source tree is `8d92e023e4d4b29beca50e0c7028572ef7dd78ba`, identical
to measured candidate `df7aea21ec55ce39bf34d78e53668ab8d4c86f44` and verified
integration `7871653f5e3234cce40e7b7c985b31b846ddc8f2`. #65 changes only the
checkout-only Wishart diagnostic/tests/docs, installed-release fixture and
test-process fixture lifetime, plus this trace. It adds no Wishart training or package exports.

## Validation bundles

All test paths below are repository-relative. Node names are existing public
behavioral boundaries; the complete suite runs their parameterizations together.

| Bundle | Final evidence |
| --- | --- |
| N | tests/test_species_workflow.py and tests/test_observations.py: native one/many species, independent balanced updates, irregular coverage, biological identity, common feature/target contracts, rejected flat forms. |
| P | tests/test_prediction.py and sparse requested-pair scientific references in test_conductance.py/test_mlpe_conditioning.py. |
| C | tests/test_conductance.py: dense graph/gradient references, species surfaces, selected terminals, patch controls, solver failures and native persistence. |
| M | tests/test_species_mlpe.py, test_mlpe_training.py, test_mlpe.py, test_mlpe_acceleration.py and independent tests/mlpe_reference.py: exact full-group Gaussian/profiled ML, frozen validation and gradients. |
| B | tests/test_bounded_training.py, test_multiregion.py and test_pooled_training_reference.py: corrected stochastic weights, chunk/dropout equivalence, complete heads, sparse evaluation/resume, numeric species-ID compilation reuse and independent complete-group Adam references. |
| L | tests/test_loglinear.py: generic prepared covariates, means before stable bounded link, distinct species coefficients/intercepts, gradients/extremes and direct/MLPE workflows. |
| K | tests/test_recalibration.py, test_species_conditioning.py and test_mlpe_conditioning.py: frozen whole distance model, group replacement/retention, roles, species-isolated effects and independent Gaussian support references. |
| A | tests/test_persistence.py, test_distance_models.py and retained recalibration/conductance artifact consumers: exact static/ordered registry/dtype contract, native custom models, corruption/atomic/cross-process behavior. |
| S | tests/test_checkpoint.py and test_native_state.py: all families/objectives, original-budget between-evaluation disk resume, latest/selected parameters, Adam/nuisance/RNG/input fingerprints and sparse history. |
| F | tests/test_ensemble.py and test_species_ensemble.py: deterministic independent group holdouts, explicit coordinated spatial folds, selected keyed original-scale deployment, failures and member continuation. |
| O | tests/test_evaluation.py, test_species_evaluation.py and test_artifact_ensemble.py: same-endpoint nominal/actual nonaccess, own-query exclusion, group/species coverage, missing/failed contributions, conditional provenance and retained artifact/member continuation. |
| D | README.md, GLOSSARY.md, docs/models.md/statistics.md/persistence.md/mlpe-conditioning.md/evaluation.md; tests/test_readme.py and test_package_boundary.py; exactly eight examples. |
| R | benchmarks/spec52/runtime-results.md and results/runtime-comparison.json: all 32 workloads, both required parity pairs and all six adjacent marginal gates, ten fits, seeded intervals, startup/prediction/RSS disclosures and separate LogLinear. |
| V | benchmarks/spec52/large-registry-memory.md and results/large-registry/final-df7aea21/analysis.json: all 14 public-fit cases, two updates, complete evaluations/heads through 1,000 species, bounded differentiated work and honest CPU allocation/live/RSS accounting. |
| X | tests/test_ibd.py and test_experimental_wishart.py; unchanged IBD policy and independent Wishart density/holdout/identifiability/no-go references. Full local checks, examples/research, installed distributions and actual CI are recorded below. |

## All 62 user stories

| Story | Required outcome | Ticket trace | Validation trace |
| --- | --- | --- | --- |
| 1 | Joint multispecies neural learning | #55/#56/#58 | N, C, B; R measured savings. |
| 2 | Same families/workflow for one species | #55/#56/#57/#59/#64 | N, C, M, L; D examples. |
| 3 | Native abstract classes, no parallel wrappers | #55/#56/#59/#65 | N, C, L; export/source/obsolete-API audit. |
| 4 | Species-specific surface or embedding projection | #55/#56 | N learned predictions; C selected softplus outputs and ignored unselected rows. |
| 5 | Distinct ecological distances per species | #54/#55/#56 | N, P, C independent outputs. |
| 6 | Separate region–species genetic calibration | #57/#60 | M group oracle, K identity/retention. |
| 7 | Explicit architecture construction | #55/#59/#64 | N fresh-fit constructor guard, L; D catalogue/snippets. |
| 8 | Internal representation, no caller latent lifecycle | #54/#55/#58/#64 | Native custom N/P path, B; D guidance and API audit. |
| 9 | Fixed named ordered species registry | #55/#61 | N registry guards; A same-shape reassociation rejection; S contract. |
| 10 | One prepared raster with species sampling units | #55 | N and observations fixtures; B pooled regional computation. |
| 11 | Species-specific locations/counts/pairs | #55/#57/#58 | N irregular fixture, M complete irregular groups, B ragged calibration. |
| 12 | Irregular geographic coverage, no fabrication | #55/#57 | N incomplete coverage/absence guards; M weighted group oracle. |
| 13 | Common feature and target contract | #55/#60/#61 | observations feature/TargetSpec checks, K compatibility, A/S persistence. |
| 14 | Species identity in partitions/provenance | #55/#57/#60/#62 | N nested partitions, K; F/O member/access scope. |
| 15 | User supplies manageable subregions | #55/#58/#64 | N ragged/overlap, B region scheduling, V evidence; D user-control caveats. |
| 16 | Geographic overlap permitted | #55/#64 | N declared assignment/overlap test; D distinction from duplicated biology. |
| 17 | Cropping changes scientific domain | #54/#64 | P uses full regional graph; requested-pairs guide; D crop/U-Net-context caveats. |
| 18 | One region per biological unit within species | #55/#58/#60/#61 | N duplicate assignment/overlap; B full unobserved catalogue; K retained/supplied identity; S disk contract. |
| 19 | Region-first minibatching | #58 | B unbiased sampling and public update behavior; V large-registry evidence. |
| 20 | Optional within-region species subsampling | #58 | B weighted selection/dynamic ID/complete group calculation. |
| 21 | Separate species computation chunk control | #58 | B chunk equivalence; V differentiated bounds. |
| 22 | Accumulate before one optimizer update | #58 | B every gradient and update/chunk equivalence incl dropout and nuisance; Adam counts S. |
| 23 | Equal species influence despite unequal coverage/pairs | #55/#57/#58 | N independent balanced update, M exact MLPE oracle, B enumerated sampling expectation. |
| 24 | Exact complete-group MLPE unchanged | #57/#58 | M dense/frozen-R/endpoint/finite-difference references; B complete ragged finalization. |
| 25 | Requested-pair computation, no routine full matrices | #54/#55/#56 | P overflow/sparse-gradient tests, C sparse terminal reference. |
| 26 | Compatible species bounded vectorized computation | #58/#63 | B chunks; V allocation/shape evidence and R runtime. |
| 27 | Reuse regional graph geometry/preparation | #54/#56/#58 | C distinct weights/independent graph outputs, reload rebuilt context; V/R accounting. |
| 28 | Reuse compilation across compatible species | #55/#58/#63 | B dynamic species test; V numerical-ID versus shape specializations. |
| 29 | Steps, region/species batches/chunks, evaluation control | #55/#58/#64 | B invalid controls/sparse history; S continuation policy; D control table. |
| 30 | Comparable full evaluation at intervals | #58 | B sparse records/selection; M frozen validation; S exact step history. |
| 31 | Selected model has all required calibrated groups | #57/#58 | M selected posterior oracle; B complete ragged heads and earlier selected snapshot. |
| 32 | Raw ecological versus genetic predict distinction | #55/#60/#64 | observations target transform, P/N predictions, K new-region behavior; D explanation. |
| 33 | Explicit multispecies output selection | #55/#60/#62 | N ambiguity/invalid/missing-region selection; K; F/O. |
| 34 | Infer sole registered species | #55/#60/#62 | N one-species workflow, K; F deployment convenience. |
| 35 | Outputs species-keyed even for one | #55/#60/#62/#64 | N/K native output assertions; F/O and D. |
| 36 | Inspect species conductance independently | #56/#59 | C selected native surfaces; L inspectable coefficients/surface. |
| 37 | Frozen distance model selected-group recalibration | #60 | K full-tree freeze, group replacement and retained unrelated groups. |
| 38 | Marginal/known/support modes preserve species provenance | #57/#60/#61/#62 | M standalone, K Gaussian/species isolation, A round-trip; O. |
| 39 | Species-balanced validation and per-group reports | #55/#57/#58/#62 | N/M/B aggregate and group reports; O separate OOF metrics. |
| 40 | Missing validation species/groups explicit | #55/#58/#61 | N nested coverage, B sparse selected record, S saved missing coverage; D example. |
| 41 | Automatic holdouts within each region–species group | #62 | F/O implemented deterministic independent group holdouts. |
| 42 | Caller-coordinated spatial folds | #62/#64 | F/O implemented explicit joint spatial fixture and leakage checks; D explanation. |
| 43 | Native fold ensembles and OOF | #62 | F/O implemented, including species-keyed summaries and retained artifact consumers. |
| 44 | Query targets excluded from fit/calibration/selection | #55/#57/#60/#62 | N partition/validation guards, M frozen target checks, K roles; O all-mode eligibility. |
| 45 | Artifact exact registry/architecture association | #61 | A same-shaped reorder/name/static/captured-state/dtype/cross-process checks. |
| 46 | Restore optimizer/sampling/random trajectory | #58/#61 | B sampled in-memory resume; S exact disk continuation of every family/objective. |
| 47 | Optimizer progress independent of sparse history | #55/#58/#61 | B between-evaluation callback, S explicit step/selected_step/progress validation. |
| 48 | Custom native extension, no legacy adapter | #54/#55/#58/#61/#64 | N custom root/family paths; A/S generic custom persistence; D guidance. |
| 49 | Generic low-capacity log-linear conductance | #59 | L direct and MLPE native public references; A/S family round-trips. |
| 50 | Arbitrary supplied covariates | #59/#64 | L generic channel/layout tests; D no study-specific schema. |
| 51 | Explicit scaling/categorical reference preparation | #59/#64 | L supplied feature order/reference fixture; class docstring and D explanatory review. |
| 52 | Patch averaging precedes linear predictor/link | #59 | L nonlinear-order and within-patch permutation reference. |
| 53 | Positive bounded exponential and trainable species intercept | #59 | L defaults/limits/extremes/independent gradient/narrow interval. |
| 54 | No learned cross-species log-linear coefficient pooling | #59/#64 | L independent-species update; D interpretation; R separate diagnostic timing. |
| 55 | README hierarchy catalogue of every choice | #64 | D implemented, with IBD separate and composition/calibration distinction. |
| 56 | Seven migrated examples remain single-species | #64 | D implemented, exact listed roles and successful public execution. |
| 57 | One neural multispecies learning/memory example | #64 | D implemented, irregular coverage/exact heads/explicit species/group reports. |
| 58 | No dedicated log-linear example | #64/#65 | D implemented exact eight-file catalogue/count audit. |
| 59 | Pooling vocabulary consistently states axis | #55/#57/#59/#60/#64 | N/M/L/K actual separation; D glossary/docs review; obsolete audit. |
| 60 | Remove old APIs, aliases, shims and artifact migrations | #54–#65 | Focused audit below, N rejected flat forms, A rejected schema; native F/O/D/X migrations. |
| 61 | Public behavioral tests plus independent science references | #54–#65 | N/P/C/M/B/L/K/A/S; F/O/D/X final suite; reference integrity review. |
| 62 | Measured memory/batching includes backward/solver | #54/#58/#63 | P/C solver adjoint accounting, B numerical correctness; V/R final measured evidence. |

## All 11 implementation sections

| Section | Full decision checklist and evidence |
| --- | --- |
| 1. Native hierarchy/model choice/pooling | #55/#56/#59 N/C/L verify one native root with conductance/embedding abstract families, explicit constructors, fixed ordered registry and species projections; #57 M unpooled calibration hierarchy. D lists exact concrete classes, separate IBD and composed LandscapeGeneticsModel, distinguish deterministic pooled weights from hierarchical partial pooling, retain geographic “region,” and cover all applicable architecture settings. Audit prohibits wrapper families/backbone copies/old names. |
| 2. Prepared data/identity/uniform form | #55 N/observations: SamplingUnits IDs/positions/kinds, one labelled feature raster, same channels/meanings/order/preprocessing, nested region/species mappings even singly, pair/matrix constructors with absent NaNs, authoritative TargetSpec, incomplete coverage with every registered species trained, partitions/roles/unknown identity errors. #58 B/#61 S full declared units; #60 K one biological species/unit region across retained/new evidence; cross-species label reuse valid. M preserves population-only MLPE and dissimilarity/identifiability/precision guards. |
| 3. Geography/graph reuse/subregions | #54/#56 P/C keep full regional four-neighbor positive graph, node-to-patch rule and arithmetic-mean endpoint edges, host geometry/optional preparation reuse but fresh weighted solves, explicit solver policy/failures. N permits geographic overlap with disjoint within-species units. D explains crop paths/U-Net context/halo non-equivalence; no masks/disconnected topology/automatic cropping. V records solver/adjoint memory. |
| 4. Public distance/prediction | #54/#55/#60 P/N/K verify native RegionBatch/species/keyed pairs, private numerical custom seam, no public latent lifecycle, pair vectors without full matrix intermediate, selected species/endpoints, explicit ecological_distances matrices and conductance maps, original-target predict without query values, sole-species inference but keyed output, requested order/orientation and all invalid pair/selection guards, raw unseen-region versus required MLPE head. |
| 5. Exact statistical objective/species balance | #55/#57/#58 N/M/B independently verify `(1/S) sum_s (1/coverage_s) sum_r mean_group_loss`, nonempty eligible species, direct and MLPE meanings, exact full-group standardization/signed GLS/profiled Gaussian ML/nuisance variances, no pair minibatch approximation, inclusion probability correction/no sampled re-normalization, endpoint algebra O(U²+P) storage/O(U³) factorization, independent region/species nuisance/population effects. |
| 6. Training/scheduling/memory | #58 B verifies validated steps/region batch/species batch/species chunk/evaluation interval; uniform region-first then eligible-species selection; one logical regional pooled realization and one accumulated Adam update; valid backward recomputation with preserved dropout; ResNet patch control separate; bounded compatible-shape work without raster padding/full-model gradient per species; no eager all-regions/species intermediates; numerical IDs and random stream stable. R/V measure backward/solver/full-evaluation behavior and legitimate shape specialization. D describes user choices and disclaims automatic hardware tuning. |
| 7. Evaluation/selection/completed calibration | #55/#57/#58 N/M/B validate initial/interval/final complete evaluation, same species balance on available validation with explicit missing coverage, no validation gradient/refit/random perturbation, earlier best snapshot with validation/final policy without validation, actual sparse step indices, all required heads matching selected distance/nuisance state regardless of last sampled group, bounded finalization. S preserves these distinctions on disk. |
| 8. Calibration/conditioning/provenance/ensembles | #57/#60 M/K complete MLPE group identity/target/posterior/access; frozen whole model, replace only supplied groups, strict recorded-training/default versus declared development roles; exact marginal/known/support conditioning and cross-species independence. F/O #62 integrated extend fold/member/support/query/access/coverage/failures, independent group holdouts/caller spatial splits, nominal and actual access, original-scale averaging, member continuation, and distinct surface/raw/genetic aggregates. |
| 9. Persistence/continuation | #61 A/S exact ordered registry/static/captured settings/array shapes/dtypes/target/features/group identity, explicit step/selected_step/on_step between evaluations, latest distance/nuisance/Adam/RNG/contracts versus selected snapshot/posterior, complete original input fingerprints/order, exact direct versus joint-MLPE optimizer layouts, sparse records and complete coverage, cross-process custom/built-ins, corruption/atomic/x64 errors, new schemas with old readers rejected. B ensures current sampling behavior; O/D artifact consumers. |
| 10. Generic log-linear conductance | #59 L validates generic channel/species construction, caller scaling/reference coding, fixed nonoverlapping feature patch means before eta, independent species coefficient rows/intercepts pooled across regions, no learned backbone/hierarchical shrinkage, exact `clip(exp(min(eta, log(b-a)))+a,a,b)` stable link, positive finite bounds a=.01/b=1000 and patch size 32 defaults, zero initialization 1.01/no key, raster/graph/x64 policy, feature association, narrow/extreme cases. D explains intercept scale limitation/floor-cap effect, and ResNet link unchanged; R separately reports log-linear without neural amortization claims. |
| 11. Examples/README/terminology/migration | #64 D implemented exact seven single-species files (rename shared_regions→pooled_regions) plus one neural multispecies example, no ninth log-linear script; hierarchy/IBD/composition/calibration catalogue; construction/nested inputs/selection/keyed raw vs genetic/custom extension; pooling axes and controls; crop/identity/selection/reproducible artifacts; log-linear formula/coding/intercept; glossary/exports/links/callback/state/docs. #65 migrates remaining consumers directly, preserving independent scientific behavior without a compatibility branch. |

## All 17 testing decisions

| Decision | Final verification |
| --- | --- |
| 1. Public workflow boundary | Full N/P/C/M/K/A/S suite exercises fit→raw/genetic/recalibrate/reload/resume; F/O/D. No new public encoding test interface. |
| 2. Behavioral tests, not incidental structure | Changed tests preserve outputs, gradients, identity, selected state and externally visible execution evidence. Dynamic-ID compilation checks observe reuse without tying the contract to one helper/vmap layout. |
| 3. Independent numerical references | Preserve C dense Laplacian, M dense Gaussian/frozen-R, independent finite differences and N/B equal-species/coverage expectations; expected values cannot be computed by the production method under test. |
| 4. Uniform single/multiple/irregular species | N/M/B/L keyed public data/model path covers differing regions/units/pairs/shapes and explicit selection/rejection. D confirms readable one-species usage. |
| 5. Neural pooling/group separation | N/C show backbone learning and distinct species outputs; M/K show independent calibration/population identities. V/R complete measured execution, not a “fixed means frozen” claim. |
| 6. Objective/sampling correctness | N/M/B independently enumerate unequal coverage/pair weights and sampled expectation, exact group standardization/profiling/variance gradients. |
| 7. Chunk equivalence | B compares losses/gradients/updates/all group heads with identical selected groups and dropout realization; no extra Adam update per chunk. C preserves patch chunk numerical behavior. |
| 8. Requested pairs | P/C compare independent graph/embedding references, repeated/ground/coincident endpoints, requested order/orientation, matrices with missing off-diagonals, and selected-species isolation. |
| 9. Geometry/overlap | N declared biological assignment versus geographic overlap; C distinct edge weights on shared geometry. D does not assert cropped/full-domain equivalence. |
| 10. Log-linear | All L numerical/default/coding/mean-before-link/extreme/gradient tests plus A/S round-trip; D formula/interpretation. |
| 11. Calibration/access | K exact freeze/retention/missing groups/roles/species-known-support cases, M frozen validation, O leakage after recalibration and support. |
| 12. Evaluation/selection | N/M/B/S species balance and visible missing groups, fully evaluated selected model, complete matching heads, trajectory/order/evaluation independence; O scoring semantics. |
| 13. Folds/ensembles | F/O implemented automatic independent holdouts, coordinated spatial partitions, original-scale aggregation, actual-access OOF, missing contributors/failures and exact member continuation. |
| 14. Artifacts/exact continuation | A/S all native families and direct/MLPE, exact ordered registry/static/dtype, between-evaluation dropout/sampling/Adam/selected-head resume, corruption/atomic/cross-process custom; no legacy-format acceptance. O/D actual downstream consumers. |
| 15. Memory/compilation | V/R implemented fixed selected/chunk bounds toward 100/1,000, backward resistance and full evaluation/finalization; separate persistent parameter/optimizer/calibration growth; dynamic IDs vs shape specialization; no universal hardware budget. |
| 16. Manageable ordinary CI | Small synthetic fixtures, no private ecological datasets or hardware timing thresholds in pytest. Performance acceptance remains required but separately scheduled. Check overall CI 30-minute budget honestly; do not skip tests to hide runtime. |
| 17. Docs/package integration | D/X implemented exact README code, relative links/images/catalogue/examples, public exports, isolated installed wheel, all examples/research smoke/source+wheel build and complete required static/test checks. |

## Runtime and candidate identity

The official runtime and memory analyzers were rerun against committed raw files
on this final production tree. Their complete outputs equal the archived analyses
exactly (CLI raw-input path spelling may differ). Runtime status is pass and the
memory matrix is complete; no `--allow-partial` was used. The coordinator also
independently audited every interval/input/initialization/work/source manifest and
ran a fresh matching U-Net100/chunk4 public-fit probe on integration. No new
performance timing was mixed with ordinary test load.

Frozen target main is `8439f99ea37f04273a1868b8009e1076239ad316`; the
coordinator freshly verified it before final acceptance. Both native measured
drivers, main driver, fixtures and frozen protocol/policy remain byte-identical
to the measured candidate. Earlier unfavorable attempts and rejected broad
fusion are preserved. Two non-gated steady diagnostics remain inconclusive;
full-fit savings include removal of recurring first-update overhead and do not
claim general long-training or convergence speedup. CPU compiled allocations,
logical live arrays and RSS are distinct; unavailable full-fit GPU allocator
peaks remain unclaimed. See the committed R/V reports for exact values and limits.

## Representative final public test nodes

The trace bundles include these exact existing node IDs (parameterized variants
are collected by the full suite):

- `tests/test_species_workflow.py::test_irregular_species_coverage_matches_independent_balanced_objective_and_update`
- `tests/test_species_mlpe.py::test_irregular_mlpe_species_objective_gradients_and_final_calibrations_match_dense_oracle`
- `tests/test_conductance.py::test_resnet_species_graphs_and_joint_gradients_match_independent_reference`
- `tests/test_bounded_training.py::test_region_first_sampling_has_unbiased_species_coverage_weights`
- `tests/test_bounded_training.py::test_unet_reuses_region_encoding_and_chunk_updates_match_with_dropout`
- `tests/test_pooled_training_reference.py::test_bounded_neural_adam_matches_public_distance_likelihood_direction`
- `tests/test_loglinear.py::test_loglinear_mlpe_gradients_and_group_predictions_match_independent_gaussian_reference`
- `tests/test_checkpoint.py::test_between_evaluations_disk_resume_matches_uninterrupted`
- `tests/test_species_ensemble.py::test_nested_holdouts_are_deterministic_independent_and_have_no_validation`
- `tests/test_species_evaluation.py::test_disjoint_development_and_common_outer_test_keep_measurements_out_of_prediction`
- `tests/test_species_evaluation.py::test_group_and_species_metrics_score_unique_queries_once_and_keep_missing_coverage`
- `tests/test_artifact_ensemble.py::test_individual_ensemble_member_disk_resume_uses_a_custom_encoder`
- `tests/test_package_boundary.py::test_built_package_runs_public_workflow_outside_source_tree`
- `tests/test_experimental_wishart.py::test_actual_graph_diagnostic_connects_covariance_and_retains_no_go_gate`

## Clean API and scope audit

Current production/callers/docs have no obsolete public epoch/callback,
calibration, flat-data, raw-distance or shared-regions API. The frozen main_driver.py and archived raw/rejected
patches intentionally preserve historical names and are revision-specific
evidence; they are not supported current APIs. Frozen main driver is separately
typed against its frozen source. Native public exports include SamplingUnits,
RawDistances, StepRecord, MLPECalibration and UnpooledMLPECalibrationModule;
models exports include both abstract families and all three concrete choices.

Artifact reading accepts only the new schema 2/kinds and exact explicit-template
contract. No migration or solver-policy overwrite is present. Numerical pair
primitives dynamically select registered species rows; public extension has no
latent/encode lifecycle. Legitimate sampling_unit_ids on observations/results
remain identity metadata. The one-species IBD API remains its separate baseline.

All scope exclusions remain: fixed registry/no zero-shot species, no hierarchical
calibration/log-linear shrinkage/cross-species covariance, no graph masks or
species-specific feature schemas, no approximate pair-minibatch MLPE or promoted
Wishart/individual-relatedness likelihood, no automatic crop/GPU tuner, no
crop/halo equivalence, no private paper data/study-specific covariates, no ResNet
link or IBD redesign, and no universal hardware or cross-device bitwise promise.

The experimental Wishart diagnostic now requires explicit native species,
records species_name and uses model-owned solver policy through the explicit
complete ecological-distance diagnostic. Independent density/scoring references
and the no_go gate are preserved. Installed-release fixture copies benchmarks
and GLOSSARY alongside tests/docs/examples/experiments; the installed import-origin
assertion, Python 3.12 and AMG checks remain intact.

## Test compilation-cache lifetime

Two local whole-suite processes (source Python 3.13 and installed Python 3.12)
terminated with native segmentation faults in JAX `backend_compile_and_load`
during different LogLinear tests. The exact node and complete LogLinear module
passed in fresh processes. Early, middle and late preceding-file subsets also
passed (57, 74 and 105 cases). Clean GitHub CI completed all scientific tests;
its two separate-process helper-import failures were fixed by explicitly adding
only the test helper directory in those children, preserving toolkit resolution.

An external controlled comparison clears only `jax.clear_caches()` between test
modules and logs process resources/configuration without changing numerical
inputs, precision, assertions or within-module compilation reuse. Both source and
installed Python 3.12+AMG controlled runs passed the former LogLinear failure
points. The minimal `tests/conftest.py` fixture follows that cache-lifetime policy;
it adds no production cache clearing, test exclusion, skip or tolerance change.
The complete suite is the regression seam. Full controlled/canonical outcomes
remain required and are recorded in the coordinator's final evidence.

The upstream compiler cause remains unresolved. This is a test-process
cache-lifetime workaround supported by the comparison, not a claimed numerical
kernel fix or proof of system OOM. Original crash logs, resource snapshots and
prefix outcomes are retained with the final acceptance evidence.

## Final execution record

Stable implementation commit: `457449482f3bd29d976d4747258fc2f75defb01f`.
Coordinator-verified integration: `f768e857eba6491aeba2efbefce6ae013bf8305c`.
Those initial implementation trees are identical. Later test-harness changes
supply the explicit subprocess helper path and scope compilation caches to test
modules; final test/fixture commit identities are recorded with closing evidence.
Production numerical source and measured drivers remain unchanged. The original user checkout's glossary edit and deleted ADR
remain untouched and match the coordinator's original patch checksum.

Completed local checks use Python 3.13.13, JAX/JAXlib 0.10.2, Equinox 0.13.8,
NumPy 2.5.4, SciPy 1.18.1, Optax 0.2.8, Coordax 0.2.9 and JAXscape 0.0.10:

- Full Ruff lint and formatting pass (85 Python files); shell syntax and changed
  whitespace pass. All 42 relative Markdown links resolve.
- Full Pyright passes with zero errors/warnings/information under the project's
  Python 3.12 target and only its frozen main-driver exclusion. The frozen driver
  separately passes against the frozen main source.
- Wishart migration red result was two failures/11 passes, then 13 passes in
  11.01 seconds after direct native migration. Research smoke returns finite
  reference/held-out likelihoods and `no_go`. Coordinator independently obtained
  13 passes and the same smoke behavior on integration.
- Reproducing the old release source fixture fails benchmark-test collection
  because benchmarks are absent. Copying the required benchmarks yields 39
  passes in 1.18 seconds across runtime/memory-policy and README checks. This
  fixture check complements the complete installed-distribution gates below.
- Runtime and memory analyzers exactly reproduce their archived results. Both
  production source and measured driver/fixture/policy identities are verified;
  the coordinator independently reproduces all bootstrap gates as well.

The complete local suite collects 438 cases. Its command is:

```sh
PYTHONPATH=src:tests JAX_ENABLE_X64=true JAX_PLATFORMS=cpu \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 taskset -c 2,3 python -m pytest -ra
```

All eight examples run in fresh subprocesses from temporary working directories,
with absolute source PYTHONPATH and the same CPU/x64/thread settings. The checkout
research smoke uses `python -m experiments.wishart`. These complete runs are
monitored to completion; their final counts and any optional dependency skips are
recorded with the final acceptance evidence rather than inferred from partial
progress.

The coordinator independently builds fresh source and wheel distributions from
the stable implementation, then runs the actual
`.github/scripts/test-release.sh` with both artifacts, isolated Python 3.12 and
the declared AMG extra. Package/test/release-fixture identity is verified after
integration. Actual current-head GitHub CI must also pass full types, tests,
examples/research and build. At trace publication, full local execution,
installed-distribution suites and current-head CI remain pending; no result is
claimed for a gate still running.

Final commands, counts, artifact identities, current-head CI and independent
review outcomes are recorded in the coordinator's closing evidence on
[issue #65](https://github.com/vboussange/ilg-toolkit/issues/65) and
[PR #66](https://github.com/vboussange/ilg-toolkit/pull/66). Closure is conditional
on those actual outcomes. Parent #52 remains open pending the authorized PR merge;
this trace does not authorize merge or release.
