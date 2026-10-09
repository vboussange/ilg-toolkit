# Issue tracker: GitHub

Specs and implementation tickets live in [GitHub Issues for vboussange/ilg-toolkit](https://github.com/vboussange/ilg-toolkit/issues). Publishing to the tracker means creating a GitHub issue.

## Repository selection

Use `--repo vboussange/ilg-toolkit` with GitHub CLI issue commands. Use the same explicit repository with the authenticated GitHub connector when an equivalent operation is available. This remains the destination when working from a staging directory or another checkout.

The `gh` CLI requires its own authentication; a connected GitHub app does not imply that `gh` is authenticated.

## Issue operations

- Create: `gh issue create --repo vboussange/ilg-toolkit --title "..." --body-file BODY_FILE`. Apply the triage role required by the invoking skill using the configured mapping.
- Read: `gh issue view NUMBER --repo vboussange/ilg-toolkit --json number,title,body,labels,comments`.
- List: `gh issue list --repo vboussange/ilg-toolkit --state open --json number,title,body,labels,comments`, with appropriate filters.
- Comment: `gh issue comment NUMBER --repo vboussange/ilg-toolkit --body-file BODY_FILE`.
- Add/remove labels: `gh issue edit NUMBER --repo vboussange/ilg-toolkit --add-label LABEL` or `--remove-label LABEL`.
- Close only when authorized by the task: `gh issue close NUMBER --repo vboussange/ilg-toolkit --comment "..."`.

Write multiline bodies to a file and use `--body-file`; preserve actual newlines and literal text.

## Parent issues and blockers

Publish tickets in dependency order. When a source spec is a tracker issue, attach each implementation ticket as its sub-issue.

Use GitHub's native sub-issue API when supported: post to `repos/vboussange/ilg-toolkit/issues/PARENT/sub_issues` with `sub_issue_id` equal to the child's numeric database ID. If unavailable, include `Part of #PARENT` in the child body.

Use native blocking relationships when supported: post to `repos/vboussange/ilg-toolkit/issues/CHILD/dependencies/blocked_by` with `issue_id` equal to the blocker's numeric database ID. Database IDs are distinct from issue numbers and node IDs. If the client cannot set native relationships, put explicit `Blocked by: #NUMBER` references in each ticket body.

A ticket can be started when all its blockers are complete. Publishing a spec or tickets does not imply approval to implement or close their parent.

## Pull requests as a triage surface

**PRs as a request surface: no.**
