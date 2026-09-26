# Codebase audits and configuration hygiene

The training codebase behind these notes went through several whole-codebase
audits: parallel review passes over each subsystem, hand verification of
every high-impact claim, and fixes grouped by how they change results. Most
findings belonged to a few recurring classes that any research training
codebase accumulates. The first section covers running an audit, the second
the recurring findings, the third configuration hygiene, and the fourth
checks that keep a refactor or a documentation site from breaking silently.

## Running an audit

### Review by subsystem in parallel, then verify every high-impact claim by hand

Split the review by subsystem (core loss and trainer, each application's
training loop, models and datasets, evaluation, scripts), then read the exact
code path for every finding before recording it. Give each finding a
confidence: certain (the code path was read), measured (benchmarked), or
likely (mechanism verified, magnitude or trigger estimated). Record the
false positives rejected during verification with the reason, and the areas
checked and found clean.

**Why.** Review passes produce plausible findings that are wrong. Recording
the rejected ones stops the next audit from re-raising them, and the list of
clean areas shows what the audit covered.

**Evidence.** One pass reported that the "clean" training variants silently
trained on noisy data because a view index for the clean signal was never
implemented. Verification found that the datasets appended the clean view at
that index by construction; the claim went into the rejected list with that
reason.

### Group fixes by what they change, and list the behavior changes

Batch the fixes so that each batch can be approved and applied on its own:

- correctness fixes that leave healthy runs unchanged;
- changes to training dynamics, random streams or the objective on
  configurations in active use, which need explicit sign-off because they
  break comparability with earlier runs;
- repairs to configurations and launch scripts;
- analysis tooling, latent bugs and documentation.

After landing, list every deliberate behavior change: which runs are no
longer seed-comparable, which frozen random realizations changed, and which
live runs must not resume across the change.

### Test each fix, not only each finding

Write a unit test per fix, run a one-epoch smoke of every configuration a fix
touches, a two-GPU smoke for distributed changes, and one end-to-end
evaluation.

**Evidence.** Testing the fixes found two further bugs: a kernel that
rejected a keyword argument the masked-data path always passed, which
crashed one configuration at its first step, and an inner optimization loop
that crashed when called from generation code running under
`torch.no_grad()`. Code that needs gradients inside a caller's `no_grad`
region must open `torch.enable_grad()` itself.

## Recurring findings

### Values that are computed but never read

A documented correction, an effective normalization constant combining
several feature spaces, was computed every step and stored in a result
dictionary that no code read. The kernels kept the constant captured when
they were built, so they ran too sharp whenever extra feature channels were
added. For each derived value, search for its readers. A closure that
captures a constant at construction ignores later updates; pass the value per
call or read it from an explicit mutable holder.

### Mirrored code paths drift apart

Two trainers written as mirrors of each other (one per data dimensionality)
had each received fixes the other lacked: one had a non-finite-gradient guard
and the other had none, the other restored EMA weights in a `finally` block
and the first did not, and the two calibrated a normalization constant under
different policies. When two paths mirror each other, port each fix to both
in the same change, or move the shared logic into one place.

### Ignored parameters and stale comments

A configuration parameter that is accepted and then ignored misleads every
reader of the configuration. Audits found command-line parameters that
nothing used, a comment describing a weight that "ramps in" whose ramp length
was zero, and docstrings with wrong tensor shapes. Make retired parameters
fail on any non-default value, and fix comments in the same change as the
code they describe.

### Configurations that share an output directory

Two registered configurations for different architectures shared one output
directory, and launch scripts named for one variant launched another.
Automatic resume would have loaded one architecture's checkpoint into the
other. Test the registry for unique output directories, and check that
every preset name that appears in a launch script exists in the registry:

```python
pattern = re.compile(r"PRESET=[\"']?([a-z0-9][a-z0-9_-]+)")
missing = sorted(
    f"{path}: {name}"
    for path in Path("scripts").rglob("*.sh")
    for name in pattern.findall(path.read_text().replace("\\\n", " "))
    if name not in registry)
assert not missing, "unresolved presets:\n" + "\n".join(missing)
```

### Scripts that compute the repository root by counting parent directories

After a reorganization moved scripts one directory deeper, six of them
resolved the repository root to the scripts directory. Some failed at once;
others continued silently, wrote a dataset into the wrong directory, or
created the virtual environment inside the scripts directory. A watcher that
evaluated milestone checkpoints called a stale path and marked each
milestone done without checking the evaluation's exit status. Find the root
by a marker instead of a count, and check exit status before recording work
as done:

```python
ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "pyproject.toml").is_file())
```

### Launch commands that fail at argument parsing

Command-line parsers for nested configurations accept a subcommand and its
fields only in their exact form. A sweep script that selected a model variant
and set one of its fields in a single misspelled flag waited in the queue and
then exited at argument parsing. Parse every maintained launch command
without running it, as part of the static checks, and use the entry point's
real call form: a later audit dry-parsed two launchers through a different
call form, reported them broken, and proposed a fix that the real entry
point would have rejected.

### Evaluation conditioned differently from training

A metric tool sampled each binary conditioning attribute independently from
its marginal frequency, while training conditioned on real attribute rows,
whose attributes are strongly correlated. Generation was evaluated on
attribute combinations that never occur in training, while the tool's
docstring said its conditioning matched training. Relative comparisons under
the one protocol stayed consistent, which is why nothing looked wrong;
absolute scores moved (FID 255.6 with real rows against 282.8 with
independent marginals, on the same checkpoint and 512 samples). Sample
evaluation conditioning from the same distribution as training, and rebuild
the model from the configuration stored in the checkpoint. A protocol fix
re-baselines every absolute score, so run the old and new protocols on one
checkpoint before switching.

## Configuration hygiene

### Keep configuration typed, nested and owned

- Use dataclasses for configuration, never untyped dictionaries, and list
  every parameter of a core interface explicitly rather than accepting
  `**kwargs`.
- Put each field in the sub-configuration that owns it. Moving a field
  changes its command-line flag, so update every launch script in the same
  change.
- Keep one registry of named configurations per application, so that
  starting one application does not construct the other's configurations.
- A chain driver forwards any command-line overrides to every link, not
  only the first; a resumed link that omits them continues under the named
  configuration's values.
- Read the configuration file saved with a run, not the dataclass defaults,
  to see what executed; named configurations override many defaults.

### Gate structure with a static test

A static architecture test enforced ceilings on module and function length,
with an exact list of existing exceptions that a change may shorten but
never extend (the test also fails on an exception that is no longer needed),
and checked that production packages never import scripts, contain no
`__main__` blocks, and expose a small declarative surface from each
`__init__.py`. It also hashed the sorted list of registered configuration
names, so that renaming or dropping a configuration fails until the expected
digest is updated on purpose. Delete dead code only after checking
whole-tree references, reachability from registered configurations, and the
checkpoint schemas that might still name it.

## Refactor and documentation checks

### Compare a refactored training step with the committed original

Load the pre-refactor module from git under a separate module name, run it
and the refactored code on a tiny deterministic dataset with the same seeds,
and require every intermediate to be equal with zero tolerance: processed
inputs, loss targets, losses, the skip decision of the step guard,
parameters and EMA shadows after the step, samples from a fixed seed, the
epoch log line and the checkpoint dictionary. The same script can gate
performance: the comparison required the refactored step's median time to
stay within 1.2 times the original's and its peak memory within 2 MiB.

```python
def load_revision(revision: str, path: str, name: str):
    source = subprocess.run(["git", "show", f"{revision}:{path}"],
                            check=True, capture_output=True, text=True).stdout
    module = types.ModuleType(name)
    module.__file__ = f"{revision}:{path}"
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module

legacy = load_revision(BASELINE, "pkg/train.py", "legacy_train")
torch.testing.assert_close(current_loss, legacy_loss, rtol=0, atol=0)
```

Before moving shared numerical modules, record a digest of their outputs on
fixed inputs (rounded to 10 decimals and hashed), and compare after the
move.

### Build the documentation strictly in CI

Build the documentation site with `mkdocs build --strict` on every pull
request that touches documentation, and deploy it from the default branch
through GitHub Pages. The strict build turns broken links and missing
navigation entries into failures.

```yaml
on:
  pull_request:
    paths: ["docs/**", "mkdocs.yml", "requirements-docs.txt"]
  push:
    branches: [main]
    paths: ["docs/**", "mkdocs.yml", "requirements-docs.txt"]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
      - uses: actions/setup-python@v6
        with: {python-version: "3.11", cache: pip}
      - run: python -m pip install -r requirements-docs.txt
      - run: python -m mkdocs build --strict
      - if: github.event_name != 'pull_request'
        uses: actions/upload-pages-artifact@v4
        with: {path: site}
```

A separate deploy job runs `actions/deploy-pages` with `pages: write` and
`id-token: write` permissions. The repository's Pages source must be set to
GitHub Actions before the first deployment.
