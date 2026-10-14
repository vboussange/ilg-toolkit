# Requested ecological-distance computation

Fitting, prediction, frozen recalibration and support-conditioned prediction
calculate the requested pair distances directly. The explicit
`model.ecological_distances(region, species="fox")["fox"]` diagnostic returns a
complete matrix for the selected species; the public output remains species-keyed.
Pair order and orientation are preserved, and distinct labels at the same graph
vertex have zero resistance. Selecting pairs does not crop the regional graph or
remove any declared training observations from their complete-group likelihood.

Operation preparation maps requested endpoints to distance nodes and deduplicates
them on the host before JIT or differentiation. Embedding models gather those
nodes and calculate squared differences for each requested pair. Conductance
models reuse the regional four-neighbor graph geometry and solver policy; each
learned surface supplies fresh arithmetic-mean edge weights and fresh solves.
An optional all-ones AMG hierarchy is reused as a preconditioner, never as a
solution for a different conductance surface.

For resistance, let the grounded graph have `N - 1` unknown vertices and let `U`
be the number of distinct requested terminals. The independent column solves
produce potentials `P` of shape `(N - 1, U)`. Direct gathers obtain each requested
resistance from its two diagonal and two cross-potential entries. This avoids a
terminal-by-terminal output matrix and a graph-by-terminal-by-terminal product.
The ground vertex has zero potential and coincident terminals are explicitly
zeroed.

Requested pairs do not make the solve storage negligible. The existing implicit
solve adjoint retains primal potentials and solves an equally shaped adjoint;
both have storage proportional to `N * U`. Sparse-edge gradient accumulation has
the same order on this grid. Neural activations and complete-group MLPE storage
are additional costs. Bounding a later species batch must therefore bound the
number of simultaneously live differentiated solves as well as feature/surface
buffers. A chunked forward loop alone does not establish that backward bound.

Float64 graph solves, configured tolerances and explicit numerical failures are
preserved. Tiny independent graph pseudoinverses and finite-difference objectives
check the selected-pair predictions and optimizer gradients in eager and JIT
workflows. Conditional predictions retain the independent joint-Gaussian checks;
query observations never enter calibration or fitting.
