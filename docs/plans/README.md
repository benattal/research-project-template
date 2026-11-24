# Plans

A plan defines intended multi-phase work: the goal, the fixed contracts, the
staged progression, and the completion criteria. It is what the user reads to
see what is done and what comes next. Operational state (jobs, numbers, dated
results, the resume point) belongs in the matching tracker under
`docs/tracking/`, which has the same file stem.

## Conventions

- Keep one active plan per workstream, and as few workstreams as possible.
- Keep a plan under 500 lines. Put the tasks near the top, numbered in the
  order they are worked; the next task starts only when the current one is
  done.
- Organise a plan by use: constraints, levers and tasks each in their own
  section ([Organising a plan](#organising-a-plan)).
- Write a plan without dates. It is self-contained: the rules that bind the
  work, the method or recipe as deployed, the staged progression, and the
  completion criteria.
- When a stage lands, update the plan's task table and status section in the
  same commit that lands it.
- Update the plan only when the intended work, dependencies, or acceptance
  gates change. Evidence and chronology go in the tracker.
- Link the plan and its tracker to each other in their first lines.
- When a plan is complete, promote its durable decisions into concepts,
  implementation guides, or [provenance](../papers/provenance.md), then move
  the plan and tracker into `docs/archive/<date>/` (see
  [the archive](../archive/README.md)).

## Organising a plan

A plan is organised by use, not by the order things happened. The user's
test (2026-11-24): "How easy is it to (a) understand the constraints, (b)
understand tasks, (c) understand levers to use?" A reader answers each from
one section, in about a minute:

- **(a) What is fixed?** The constraints.
- **(b) What is next, and what decides it?** The tasks.
- **(c) What may be changed, and what has been tried?** The levers.

The sections, in this order:

1. **The goal and the bar.** What done means: one score beside its
   reference, and one line of status.
2. **Constraints.** What must not change. One table, one row a constraint,
   with the config key or the check that holds it. Only fixed things go
   here.
3. **Levers.** What may be changed. One table: the lever, its key, its
   current value, what the user expects of it (empty when the user has said
   nothing), and what is known (untried, the run testing it, the result).
   Below it, one line on what is not a lever: settings a reader might take
   for one that the constraints rule out.
4. **Tasks.** In the order they are worked. Each is a question, the lever it
   turns, its runs, the bar that passes it, and a status. No results inline.
5. **Where it stands.** Short: what is established and what is open, with a
   link to the tracker.
6. **Reference.** The method as deployed in a few lines, the keys'
   definitions, the agent's own choices. A longer derivation goes in a
   concept note (`docs/concepts/`) that the plan links: "You can put more
   involved derivations in docs/concepts" (the user, 2026-11-24).

The rules that keep it so:

- **One kind a section.** Before writing a statement down, sort it: a
  constraint, a lever, a hypothesis to test, a task, a rule on how the work
  is done, a result, or a choice the agent made. A requirements list that
  mixes them hides which items are fixed.
- **A new direction goes where its kind lives,** not at the end of a list.
  Numbers stay stable once given (`R7`, `L3`, `B2`), since trackers and
  commits cite them; the order on the page is by kind.
- **The agent's choices stand apart from the user's requirements,** each
  marked open to change. A lever whose current value the user did not set
  says so.
- **Each fact in one place.** Other sections cite it by its number. A task
  names its lever; a lever names the requirement behind it.
- **Results leave the plan as they age.** A task's result goes to the
  tracker. When the base method changes, the runs on the old base move to
  the archive with what they showed, and the levers table marks what must be
  measured again.
- **A lever is not a config key.** The keys' table belongs to the reference;
  the levers table says what turning each one is expected to do and what has
  been seen.

## Skeleton

```markdown
# <Workstream>

[Latest status and resume point](../tracking/<stem>.md) ·
[Concept](../concepts/<topic>.md) · [Production lineage](../production-configs.md)

One paragraph: the idea.

## The bar

What done means: the score, its reference, and one line of status.

## Constraints

| | Constraint | Held by |
|---|---|---|
| R1 | <what is fixed> | `<config key or check>` |

How the work is done: the rules on process, a line each.

## Levers

| | Lever | Key | Now | What the user expects of it | What is known |
|---|---|---|---|---|---|
| L1 | <what may be changed> | `<key>` | <value> | <quote, or empty> | untried |

Not levers: <settings the constraints rule out>.

## Tasks

Work these in order, one at a time.

| | Question | Lever | Runs | Pass | Status |
|---|---|---|---|---|---|
| A1 | <what the run decides> | L1 | <runs> | <the gate> | open |

## Where it stands

What is established and what is open; the evidence is in the tracker.

## Method as deployed

The recipe in a paragraph, with the preset names that implement it, and the
keys' definitions. The derivation is in the concept note.

## My choices, open to change

The agent's own settings, numbered; none is a requirement.

## Completion criteria

The measurable gates that close the plan.
```

## Active plans

TODO(template): list the active plans here, each with its tracker.
