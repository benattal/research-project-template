# Code style and organization

Code structure and visual quality are part of correctness in this repository.
The goal is not uniform small files; it is a tree whose names and ownership let
a contributor find behavior without guessing.

## Semantic ownership

- Put application-specific behavior under `src/apps/<application>/` with its
  configuration, presets, entries, models, training, objectives, and
  artifacts.
- Put behavior under `src/core/` only when at least two applications consume
  the same semantic contract.
- Organize by what code owns, not by historical call order or file size.
- A deeper directory must represent a real multi-file subdomain. Fold
  one-file packages and compatibility wrappers into their owner.
- Never create generic `common`, `base`, `helpers`, `misc`, or `runtime`
  catch-alls when a concrete domain name exists.

## Cohesive files

Long files are encouraged when the enclosed behavior shares one state or
lifecycle. Do not split code merely to meet a line target.

Treat these as refactoring prompts:

- sibling `<name>_<subname>.py` families;
- `checkpoint.py` beside `checkpoints.py`;
- `ema.py` beside `ema_scope.py`;
- one implementation fragmented into `model`, `cache`, `reference`, and
  `common` files that cannot be understood independently;
- a wrapper file whose only job is importing and calling another entry.

Split a long file only when the extracted concern has an independently useful
public contract, state owner, or multiple consumers.

## Shared lifecycles

- One shared lifecycle has one loop.
- Application differences in a shared lifecycle are primarily data: rates,
  budgets, dimensions, schedules, paths, and numerical parameters configure
  the same concrete algorithm.
- Applications customize it with typed callbacks, providers, parameter
  bindings, optimizer slots, and artifact hooks.
- Share infrastructure only while those differences remain data-driven and
  the shared control flow stays easy to state. If supporting a behavior needs
  nested modes, pervasive branches, or a callback that replaces the lifecycle,
  stop and rethink the ownership. Prefer a separate workflow or entry script
  inside the owning application when the lifecycle is different.
- Do not replace a boolean mode with a string, enum, optional component, or
  typed union while retaining the same config-driven branch tree. Resolve
  concrete components and bounded callbacks before entering the loop. The
  loop invokes their stable interfaces; configuration does not choose phase
  ordering or repeatedly dispatch algorithm branches.
- A callback is bounded: it supplies one operation at an established lifecycle
  point. If it replaces phase ordering, state ownership, or most of the loop,
  it is a separate workflow inside the owning application, not a callback.
- Do not select implementation behavior through trainer mode strings.
- Do not copy a loop to change one objective or estimator.
- Avoid behavior-mixin diamonds. Install a callback contract or use explicit
  composition.
- A callback mutates only state it owns, through public operations.

## Naming and layout

- Files, directories, classes, variables, and configuration fields must be
  self-explanatory in their local context.
- Use domain nouns for state and domain verbs for operations. Avoid anonymous
  `data`, `result`, `thing`, or `mode` when the quantity has a name.
- Preserve units and measures in names or docstrings: a density per area, a
  probability per solid angle, a per-pixel mass, and a per-sample weight are
  not interchangeable.
- Keep imports grouped and deduplicated. Keep phases visually ordered in the
  sequence they execute.
- Use whitespace to expose structure, not to create large mechanical gaps.
- Comments explain invariants, measures, ownership, or non-obvious device
  behavior. Remove timeline narration after its durable lesson is documented.
- Name variants precisely. When two architectures differ in one structural
  property, put that property in both names rather than calling one of them
  plain.

## Interfaces and configuration

- Internal interfaces fail explicitly when a required value or operation is
  absent. Do not hide an ownership error behind a passive alias.
- Use typed, frozen dataclasses for numerical configuration. Use concrete
  component factories or callbacks for implementation choices rather than
  config-driven runtime modes.
- Preset names are user-facing compositions, not implementation modes.
- Delete dead flags and unreachable variants after a reachability check.
- Checkpoint schemas stamp semantics, mapping, support, and model state;
  loaders do not guess or silently repair incompatible state.

### Explicit configuration ownership

- One field owns one behavior. A boolean, mode string, objective selector,
  sentinel value, convenience path, or callback must never silently rewrite,
  infer, disable, clamp, replace, or reinterpret another explicit field.
- Independent concerns receive independent fields. Changing one model's
  objective cannot change another model's representation, choosing a
  checkpoint cannot silently change an iteration budget, and selecting a
  workflow cannot replace a configured loss after parsing.
- Do not implement precedence chains such as "use A unless B is nonempty" when
  both A and B are public configuration. Give the resolved behavior one typed
  owner, or reject the conflicting combination with a useful error.
- Avoid magic sentinels (`-1` means inherit, `0` means disabled, empty means
  load instead, `None` means use another field) when they alter another
  parameter's meaning. Prefer an explicit strategy type or separately named
  quantities such as completed versus remaining iterations. If a sentinel is
  the value of the same quantity, document and validate it locally.
- Runtime code treats parsed configuration as immutable. Path normalization or
  derived runtime state belongs in an explicitly named resolved copy; saved
  provenance records the resolved values. Never mutate the requested config
  to make it agree with what the runtime decided to execute.
- A strategy may make some parameters inapplicable, but it must not silently
  ignore contradictory non-default values. Use a typed strategy dataclass
  where practical; otherwise validate the compatibility table before any
  allocation or training.
- Compatibility is validation, not inference. An objective and a
  representation may be checked for compatibility, but neither selects the
  other. Persisted contracts stamp both independently.
- Audit resolved registries as well as source literals. Tests must show that
  changing one independent field leaves the others unchanged and that invalid
  combinations fail before work begins.

### Explicit optimization schedules

- Every optimizer owner receives one typed schedule object in its preset
  (`src/core/training/schedules.py`). The saved configuration states the
  warmup, peak factor, anneal, minimum factor, and the phase budget that
  supplies the schedule horizon.
- A loop consumes that schedule directly. Do not add boolean switches such as
  `use_anneal`, `reset_schedule`, or `enable_warmup` that synthesize, replace,
  multiply, or silently override a schedule at runtime.
- Do not represent one schedule partly as a dataclass and partly as scalar
  fields (`*_warmup_iters`, `*_anneal`, `*_min_factor`). Combine the fields
  into the owning schedule, or give independent parameter families separate
  named schedules.
- Presets with different horizons construct different schedule values
  explicitly. A schedule inherited from a long recipe must not be reused by a
  short diagnostic recipe without an explicit decision in that preset.
- For a finite optimization, compare the integrated nominal step budget with
  the requested parameter span and keep explicit margin for noisy or canceling
  gradients.
- Parameter values, optimizer reset policy, EMA policy, and sampling budgets
  are separate concerns. Each is visible in the preset and applied once. Name
  a constant parameter-family multiplier `*_lr_scale`, not `*_lr_factor`, so
  it cannot be mistaken for a schedule.
- Add a static contract test that resolves representative presets and checks
  the effective schedule and parameter values. A test that a flag exists is
  not enough: the resolved configuration is the contract.

## Randomness and performance

- Calls that promise fresh samples advance independent named streams.
- Diagnostics cannot perturb training stream positions.
- Changing seeds updates device state without changing compiled structure.
- Keep data exchange between frameworks on device and scalar materialization
  outside the hot loop.
- Detach persistent autograd state between iterations.
- Performance changes require matched numerical output, stable compilation,
  and flat memory, not only a shorter wall time.

## Editing discipline

1. Read the owning concept and implementation document.
2. Characterize behavior before a structural or numerical change.
3. Move callers directly; do not leave forwarding modules.
4. Run syntax and missing-import checks and `scripts/infra/check_source_tree.py`.
5. Run focused behavioral tests and the relevant numerical or performance gate.
6. Check the tree again for names, depth, and one-consumer fragments.
7. Update durable documentation and remove stale workflow text.

## Review checklist

- Can a new contributor find this functionality from the tree alone?
- Is this the only implementation of the behavior?
- Does the filename name its actual concern?
- Is every additional directory layer justified?
- Are physical quantities and units distinguishable?
- Are random streams independent and reproducible?
- Does the hot path avoid host round trips, graph retention, and shape drift?
- Do checkpoint and artifact names preserve provenance?
- Did the change delete the old path and update every caller and document?

The enforceable ownership map is in
[implementation architecture](../implementation/architecture.md). The source
tree checker encodes objective findings, but a zero count does not replace
human judgment about semantics and aesthetics.
