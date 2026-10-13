# Preserved runtime attempts

These raw files are immutable observations from clean committed candidates. Each
contains its exact source, driver, inputs, initialization, scientific work, software
and affinity metadata; `sha256.json` records file hashes. None is silently combined
with the final comparison batch.

| Attempt | Candidate / chunk | Observed outcome |
| --- | --- | --- |
| 01 feature-heavy single | 3d3a4c1 / 1 | U-Net steady step failed: 1.2731× main, CI [1.1957, 2.3972]. Other parity gates passed. |
| 02 feature-heavy single | ef041936 / 1 | Both families passed whole-fit and steady-step parity using fresh ten-fit main and PR batches. |
| 02 joint / independent | ef041936 / 1 | U-Net 1→2 failed: 0.7913× added independent fit, CI [0.7699, 0.8039]. Other five adjacent increments passed. |
| 03 U-Net 1→2 diagnostic | ef041936 / 4 | Existing chunk4 alone failed: 0.7717, CI [0.7608, 0.7936]. |
| 04 U-Net 1→2 diagnostic | af4cf33 / 4 | Cached encoder-VJP preparation passed: 0.4481, CI [0.3937, 0.4666]. Full final matrix is separately required. |

The baseline repeat is `../main-feature-repeat10.json`, preserving the original
`../main.json`. It uses the unchanged frozen main production source and input
protocol; only fit repetitions increase from three to ten. The final comparison
uses this fresh matched batch for the two designated parity workloads, retaining
the original baseline for the other ten cases. Comparison analysis rejects
competing full batches or an incomplete repeat batch.

The execution chunk was not an immutable workload setting in the frozen protocol.
After the documented chunk1 failure, the candidate explicitly adopts bounded
chunk4 for every final single/joint/independent and LogLinear timing. Per-species
pairs, updates, likelihoods, solver accuracy, evaluations and final calibration
are unchanged. This is a measured implementation optimization, not a relaxed
acceptance threshold or selection of favorable repetitions.

Singleton fusion removes repeated checked dispatches. Caching the multigroup
encoder VJP removes repeated host derivative preparation while preserving the
separate tail/backward arithmetic boundaries. The rejected broader fusion and
its scientific failure are documented in `../singleton-scientific/`.
