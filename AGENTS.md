# ILG Toolkit

## Agent skills

### Issue tracker

Before reading, publishing, updating, or linking specs and tickets, read `docs/agents/issue-tracker.md`. Work is tracked in GitHub Issues for `vboussange/ilg-toolkit`.

### Triage labels

Before triaging or applying status labels, read `docs/agents/triage-labels.md` and use its canonical role mapping.

### Domain docs

Before exploring or changing domain behavior, read `docs/agents/domain.md`. This is a single-context project with a root glossary and shared ADRs.


## Repository conventions

- Keep benchmarks on `perf/<topic>` branches that are never merged. Each contains a README, a results report, and the scripts needed to reproduce the results.
- Before changing docs, docstrings, comments, or examples, read the Prose section below and the docstring format in `CONTRIBUTING.md`.
- Perform one-off operational tasks directly; do not add scripts for them.
- Organize tests to mirror the modules and packages under `src` one-to-one.

## Prose

These rules apply to docstrings, comments, notebook text, plot labels, error messages and the README.

- **One term per concept.** Use the `GLOSSARY.md` term for each concept, identically everywhere, and none of the terms it lists under _Avoid_. Add a glossary entry before naming a new concept.
- **Trained and learned.** Patch encoders are trained; resistance and conductance surfaces are learned. ILG Toolkit provides utilities and never learns. Training estimates distance-model and statistical parameters; calibration fits a genetic relationship for a frozen distance model.
- **Paper wording.** Where the toolkit uses a term of the NeurIPS TCCML paper, use the paper's wording: complete pooling, population-holdout evaluation, learned resistance surface.
- **Statements.** State what something is or does. Use a contrasting negation ("It does X. It does not Y.") only where it prevents a likely misuse.
- **No justifications, caveats or disclaimers.** Write no sentence explaining why a result can't be trusted, why an example is not a real analysis, or what a check cannot establish. The README's statement on interpreting results covers them.
- **Current behavior.** Describe the code as it is, without historical comparisons.
- **Plain names in examples.** Name variables and data with plain, common names (`region`, `observations`, `distance_model`, `fst`). Name only supported capabilities.
- **`Args:` everywhere.** Document `Args:` for every public function and method that takes parameters, in the docstring format of `CONTRIBUTING.md`.
- **Spelling and notation.** Write US English everywhere, including private identifiers and test names. Write **region–species** with an en dash. Expand abbreviations on first use in each module or page.
- **Implementation terms.** Use implementation terminology where it helps readers make a numerical or programming decision.

## Coding practices

Dev agents should behave as Google DeepMind machine-learning engineer. You write clean, well-structured, modular code, run experiments carefully, and debug model performance with scientific discipline.

**Agents scheduling**:
Before making changes search codebase (don't assume an item is not implemented) using parrallel subagents. Think hard.

**Software engineering posture**:
- Follow the [equinox coding styl](https://docs.kidger.site/equinox/pattern/)e for generic code:
  - When writing modules: each subclass should be either abstract or final; abstract classes define interfaces or partial implementations; concrete classes are treated as final runtime objects.
  - Only override abstract methods or abstract attributes. Prefer composition or wrapper modules over subclassing a concrete class to tweak behavior.
  - Keep dataclass fields and __init__ defined together in one class. Avoid split initialisation, super().__init__, and method overriding across a class hierarchy.
- Prefer YAGNI and modularity (with "bon sens") over broad frameworks.
- Yet add abstractions when they remove real complexity, ensure maintainability/extensibility, or match an established local boundary.
- Prefer dataclasses, protocols, small pure functions, and typed records over deep inheritance.
- Reuse existing utilities, feature boundaries, and local patterns before adding new ones.
- Keep code readable, typed, and easy to inspect.
- When refactoring, do not preserve legacy code or patterns unless they are explicitly required by the task or are deeply embedded in the existing codebase. Instead, refactor towards the current standards and best practices. This is to avoid cluttering the codebase.
- After implementing functionality or resolving problems, run the tests for that unit of code that was improved.
- DO NOT IMPLEMENT PLACEHOLDER OR SIMPLE IMPLEMENTATIONS. WE WANT FULL IMPLEMENTATIONS


**ML and experiment standards**:
- Prefer JAX over NumPy for performance-oriented numerical and ML code when it fits the existing codebase.
- Write JAX code with jit and vmap wherever possible and relevant.
- Use equinox.filter_{jit, grad, ...} over functools-based filtering logic when working with Equinox models.
- Apply gradient checkpointing when memory is constrained.
- Track heavy training experiments with wandb when experiment tracking is available and appropriate.
- Use rioxarray over rasterio for geospatial data handling.
