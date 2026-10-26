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
- Before changing docs, docstrings, comments, or examples, read the documentation and prose conventions in `CONTRIBUTING.md`.
- Perform one-off operational tasks directly; do not add scripts for them.
- Organize tests to mirror the modules and packages under `src` one-to-one.

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
