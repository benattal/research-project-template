# Live tracking documents

`docs/tracking/` contains the live execution state for active plans. Use one
tracking document per active plan and give both files the same stem:

- `docs/plans/<stem>.md` defines the goal, execution order, fixed contracts,
  and remaining tasks;
- `docs/tracking/<stem>.md` records current status, evidence, active jobs, and
  the exact resume point.

Update the tracking document when work lands or operational state changes.
Treat it as a stack: add the newest dated entry at the top, before older
entries. Consolidate superseded operational notes into a short current-state
handoff rather than growing a transcript of individual probe calls. Stable
findings belong in the appropriate concept or guide, not in the live tracker.

An entry records what happened, forward: the direction that started it (quoted,
with its date), what was launched (job IDs, presets, commits), what each run
reached, and what comes next. A disproved hypothesis recorded as disproved is a
result. Early readings of a running job are labelled as readings ("step N of
M"), not verdicts.

Machine-readable inventories and cleanup manifests live under
[`docs/tracking/data/`](data/README.md); keep the tracking root for
human-readable state.

When a plan is complete, archive its plan and tracker under
`docs/archive/<date>/` after promoting durable decisions into the maintained
documentation.

## Entry skeleton

```markdown
## YYYY-MM-DD HH:MM — <one-line status>

User direction (YYYY-MM-DD): "<quoted direction>".

- Launched: <preset> as job <id> on <host/GPU>, commit <sha>.
- Reading at step N of M: <numbers, with the setting that produced them>.
- Next check: <what, and when>.
```

## Active trackers

TODO(template): list the active trackers here, each with its plan.
