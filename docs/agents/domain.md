# Domain docs

This is a single-context inverse landscape genetics toolkit.

## Before exploring

Read the root `GLOSSARY.md` and the relevant decisions under `docs/adr/`. If a domain document does not exist, proceed with the task; domain-modeling creates documents lazily when terminology or decisions are resolved.

## Vocabulary and decisions

Use the glossary's canonical domain terms in issues, design discussions, and tests. Distinguish sampling units, pairwise genetic observations, landscape scores, and genetic calibration; relatedness is not automatically a genetic distance.

Surface a conflict with an existing ADR explicitly rather than silently overriding it. Use domain-modeling to resolve terminology gaps and record consequential decisions.

## Layout

- `GLOSSARY.md`: project-wide domain vocabulary.
- `docs/adr/`: shared architectural decisions.
- `docs/agents/`: engineering-skill consumer instructions.
