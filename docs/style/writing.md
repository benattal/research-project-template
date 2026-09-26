# Writing and math style

The conventions for **all prose in this repository**: the markdown under
`docs/`, `README.md`, the agent files, commit messages, code comments where
they run to sentences, and chat replies about any of them. They override
default habits.

[Code style](code.md) governs source organization and comment mechanics;
[results style](results.md) governs figures, tables, metrics, and provenance.
This guide governs prose and mathematical notation.

## Audience

TODO(template): describe the reader. For example: someone who wants to
re-implement or extend the method, comfortable with the field's standard tools
at a graduate level, but not already familiar with this method's formulation,
its specific models, or its estimators. Name the terms that may be used bare
and the ones that must be explained at first use.

Write for that reader: a capable newcomer to *this* method, not an expert in
it. Techniques from the literature may be cited, but their *use here* is
derived, not assumed.

## Writing principles

**Read like a reference.** Each section should be understandable on its own.
Explain every term the first time it appears, and define notation before using
it. Assume the reader is following linearly and has not seen the idea before.

**No jargon.** If a technical word is unavoidable, define it in plain language
at first use. If it is avoidable, avoid it. Never use an inflated word where a
plain one works.

**Each section tells one clear story.** A reader should come away knowing:

- what the goal is,
- what is wrong with the existing or obvious approach,
- what the new approach is and why it fixes the problem.

These three beats do **not** need explicit headings, but each must be clearly
explained, in order, so the motivation is never a mystery.

**Each paragraph has one job.** Open with the point the paragraph makes, then
spend the rest of the paragraph making it. A reader skimming the first
sentence of each paragraph should get the through-line.

**Claims match the implementation, and name the branch.** Names, defaults, and
behavior in prose must match the code as written: quote the identifier and its
actual default rather than describing it. A claim that holds only on one
branch says so. When prose and code disagree, fix one of them; do not paper
over the gap with a qualifier.

**Numbers stay honest.** Report what was measured, with the setting that
produced it (preset, dataset, budget, learning rate, seed) and the scheduler
job where one exists. A result that did not reach its target says so plainly,
and a preliminary finding is labelled preliminary.

## No hedging, no litigating the previous version

This is the rule most often broken, so it gets its own section.

**When asked to change something, make the change.** Present the new text as
if it had been written that way from the start. Do not explain why the earlier
text was wrong, do not mark what changed, and do not frame the new prose as a
correction.

None of these belong in a document:

- "This is *not* actually X, as stated above."
- "Contrary to the earlier description, ..."
- "**Correction:** the model predicts the ratio, not the density."
- "Note that a previous version of this section claimed ..."
- "(Updated: this now reflects the two-phase pipeline.)"
- Struck-through or commented-out old text left in place "for reference."

Each of these should be the correct statement, standing alone. The reader did
not see the old version and does not benefit from its ghost.

**The same applies to chat replies.** After an edit made at the user's
direction, show or describe the new text. Skip the recap of what was wrong
before, skip "you're right that ...", and skip the apology. If the change has a
consequence the user should know about (something else now needs updating, or
the fix revealed a problem elsewhere), say *that*, because it is new
information.

**Version history lives in git.** A document records the current state of the
world; the diff records how it got there. Do not duplicate the diff in prose.

### Where a record of the change *is* the content

Three exceptions, all narrow.

**The paper cycle.** `docs/papers/submission/` records how the work responded
to review. Quoting a reviewer's objection and answering it *is* the document;
so is a "Corrections" section that states what the submitted version got
wrong. The submitted artifacts are frozen on purpose.

**Live tracking and investigation records.** `docs/plans/` defines intended
work. Matching files under `docs/tracking/` record live status, evidence,
accepted and rejected artifacts, active jobs, and exact resume points.
Concluded findings are promoted into `docs/papers/provenance.md` or a focused
concept. Status lines, dated entries, job IDs, and hypotheses that were tried
and abandoned are the content of trackers: a disproved hypothesis recorded as
disproved is a result. Even so, write entries **forward** as a record of what
happened, not backward as corrections of what earlier entries claimed.

**Old identifiers that are still in the code.** A note mapping an old name to
a new one earns its place while the old strings still appear in the tree,
because a reader will meet them. That makes it navigation, not a changelog, and
it goes when the identifiers do.

**Don't lean on negative intuitions either.** Prefer stating what is true to
refuting what is false. Do not raise a claim only to knock it down ("you might
think X, but actually not-X"), and do not assert that an intuition is false
*unless* it is one this reader would arrive at and be misled by. Stating the
real defect of the obvious or existing approach (the second story beat above)
is different, and still required.

## Words and phrases to avoid

- Do **not** use "annihilate" for a term being cancelled. Say "cancels,"
  "removes," or "sends to zero."
- Do **not** use "the price of." Say what is traded plainly: "in exchange
  for," "at the cost of."
- Do **not** write "x is genuinely y." State it.
- Do **not** call a parameter a "knob." Say "parameter," or name it (the
  learning rate, the grid resolution `base_resolution`).
- More generally, cut clever or unusual phrasing in favor of the plain
  version. If a phrase would make a careful reader pause to decode it, rewrite
  it.

TODO(template): add project-specific words to avoid, with their plain
replacements.

### Specific clever or colloquial phrases

Each has a plain replacement. The list is illustrative, not exhaustive: the
rule is "say the literal thing."

- "does double duty" → "serves two purposes," or state the two purposes.
- "razor-edge optimum" → "the optimum."
- "X is ideal" (of a method on a problem) → "X works well here," and say why.
- "not cosmetic" → state plainly what it does.
- "the pleasant surprise" / "remarkably" / "the punchline" as throat-clearing
  → state the fact.
- Throat-clearing openers ("It is worth noting that," "Notice that," "The key
  observation is that") → state the point directly.

### Prose structures and tics to watch

Beyond individual phrases, these recurring *patterns* make writing feel
mannered. Use them sparingly; when one shows up several times on a page,
rewrite most of them plainly.

- **Em-dash appositive stacking.** One parenthetical pair per sentence at
  most; prefer commas or parentheses, or split the sentence.
- **Emphatic intensifiers used as filler.** "exactly," "entirely,"
  "precisely," "automatically," "just," "simply." Keep them where they are
  literally true ("the estimator is unbiased," "recovers the density exactly
  when the reference is analytic"). Cut them when they are only for emphasis.
- **Colon and appositive pile-ups.** Define a term once, in its own short
  sentence, and move on.
- **Dramatic one-line fragments.** Fine once per section for emphasis; not a
  habit.
- **Heavy personification** ("the model wants," "samples that try to
  escape"). A little aids intuition; a lot reads oddly. Prefer the literal
  mechanism.
- **"not X but Y" / "X, not Y" inversions** used for drama. Fine
  occasionally; don't lean on it.
- **Over-asserting that two things are the same** ("verbatim," "identical in
  form"). Say *what* is the same and *what differs*.

## Mechanics for the markdown docs

- **Math.** GitHub's Markdown renderer has real limits. Follow the
  [LaTeX and math](#latex-and-math) rules below for display math and
  commands.
- **Links.** Cross-reference other docs by relative path. Every doc should be
  reachable from [index.md](../index.md); add a pointer there (and to
  `mkdocs.yml`) when you add a file.
- **Paths and names.** Refer to code as `src/core/training/schedules.py` or
  `` `LearningRateSchedule` ``, not "the schedule module" in isolation. Keep
  names in sync with the package when it changes.
- **Code blocks** show commands that actually run, in the form the drivers
  expect: `python -m src.apps.<application> <preset-name> [overrides]`, and
  `sbatch scripts/launchers/train_slurm.sh ...` for cluster work. Name the
  preset rather than describing it.
- **One responsibility per document.** Concepts explain the method,
  implementation docs describe the code, environment guides say what to run,
  trackers log investigations, plans propose work. A section that has drifted
  into the wrong file moves.

## LaTeX and math

GitHub renders math with MathJax but runs its Markdown parser **before**
handing off to the math renderer in some cases. This causes subtle rendering
failures that are hard to debug.

### The `\text{XXX}_Y` problem (most common)

**Do not write** `$\text{label}_subscript$` in inline math.

When `_` is preceded by `}` (not an alphanumeric), GitHub's CommonMark parser
can treat it as an emphasis marker. This consumes the `_`, breaks the math
expression, and, because it also unbalances the `$` delimiter count, corrupts
all subsequent inline math on the same line.

**Bad:**

```
$\text{gen}_a$, $\text{kv}_0$, $\hat{x}_t$, $\bar{g}_a$
```

**Good:** ensure `_` is always preceded by a letter or digit:

```
$g_a$, $v_0$, $x_t$, $G_a$
```

This applies to any `\command{...}_subscript` pattern, including `\text{foo}_x`,
`\hat{x}_t`, `\bar{g}_a`, `\tilde{z}_k`, `\mathbf{v}_i`, and `\mathcal{L}_t`.

**Rule of thumb:** if the character immediately before `_` in the raw LaTeX
source is `}`, rename the variable to a single letter or move the subscript
inside the braces if the semantics allow it.

### The `$(...)$` problem

**Do not start** inline math with `$(`. GitHub may not recognize `$(` as
opening a math expression (it looks like a dollar amount).

**Bad:** `the pair $(y_{t-1}, y_{t+1})$`

**Good:** `the pair $y_{t-1}$ and $y_{t+1}$`

### Spacing commands: `\!`, `\;`

- `\!` (negative thin space): avoid in inline math; it can confuse some
  renderers.
- `\;` (thick space): safe in display math `$$...$$`, but can fail in inline
  `$...$`. Use `,` or `,\,` instead.

### Display math: always surround with blank lines

```markdown
Preceding text.

$$x_t = t \cdot x_0 + (1-t) \cdot \varepsilon$$

Following text.
```

A `$$` block without a blank line before or after may render inline instead
of as a display equation.

### The `^*` problem (bare asterisks in math)

**Do not write** bare `*` inside `$...$` inline math when another `*` appears
on the same line. GitHub's parser matches `*` pairs as emphasis markers
**before** the math renderer sees them, which breaks all math on that line.

**Bad:** `$p^*(x)$ and $q^*(x)$`

**Good:** `$p^{\ast}(x)$ and $q^{\ast}(x)$`

### The `\{` / `\}` problem (curly braces in math)

GitHub's math rendering can be unreliable with `\{` and `\}` for set braces.
Use the named commands instead.

**Bad:** `$\{x_1, \ldots, x_n\}$`

**Good:** `$\lbrace x_1, \ldots, x_n \rbrace$`

### Images

A document's images live beside it and stay small. Large or generated images
belong to a page under `docs/pages/<page>/`, rebuilt by its builder, and are
not committed.
