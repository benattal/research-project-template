# Plans

A plan defines intended multi-phase work: the goal, the fixed contracts, the
staged progression, and the completion criteria. It is what the user reads to
see what is done and what comes next. Operational state (jobs, numbers, dated
results, the resume point) belongs in the matching tracker under
`docs/tracking/`, which has the same file stem.

## Conventions

- Keep one active plan per workstream, and as few workstreams as possible.
- Keep a plan under 500 lines. Put **Immediate todos** at the top, numbered
  in the order they are worked; the next todo starts only when the current
  one is done.
- Write a plan without dates. It is self-contained: the rules that bind the
  work, the method or recipe as deployed, the staged progression, and the
  completion criteria.
- When a stage lands, update the plan's todo list and status paragraph in the
  same commit that lands it.
- Update the plan only when the intended work, dependencies, or acceptance
  gates change. Evidence and chronology go in the tracker.
- Link the plan and its tracker to each other in their first lines.
- When a plan is complete, promote its durable decisions into concepts,
  implementation guides, or [provenance](../papers/provenance.md), then move
  the plan and tracker into `docs/archive/<date>/` (see
  [the archive](../archive/README.md)).

## Skeleton

```markdown
# <Workstream>

[Latest status and resume point](../tracking/<stem>.md) ·
[Concept](../concepts/<topic>.md) · [Production lineage](../production-configs.md)

One paragraph: the goal and what "done" means.

## Immediate todos

Work these in order, one at a time.

### Done

1. **<Stage>.** What landed and the gate it passed.

### Open

2. **<Stage>.** What to do, the control that decides it, and the gate.
3. ...

## Rules

Contracts the work must not break while it proceeds.

## Method as deployed

The recipe in a paragraph, with the preset names that implement it.

## Completion criteria

The measurable gates that close the plan.
```

## Active plans

TODO(template): list the active plans here, each with its tracker.
