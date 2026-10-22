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
- Docs, docstrings, comments, and examples describe current behavior only, following the authoring rules below.
- Docstrings first state what the object is, using glossary terms, and contain at most four sentences.
- The README's Documentation section indexes every Markdown documentation page.
- Perform one-off operational tasks directly; do not add scripts for them.
- Organize tests to mirror the modules and packages under `src` one-to-one.

## Authoring rules

1. Say what something is or does. Mention a non-behavior only in a separate note.
2. Use emphasis words ("explicit", "declared", "exact", "complete", "actual", "independent") only when they contrast with a named alternative.
3. Use domain language in public text, without implementation jargon such as "tail", "VJP", "pullback", "kernel", "chunk", "seam", "host", or "bounded".
4. Use one glossary term per concept.
5. Use US spelling and write "region–species" with an en dash.
6. Expand abbreviations on first use in each module or page.
7. Omit history and contrasts with absent alternatives.
8. Give each word one meaning.
9. Follow the docstring shape in Repository conventions: what the object is first, glossary terms, at most four sentences.

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
