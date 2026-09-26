# Production configurations and lineages

This is the authoritative registry of the accepted presets, checkpoints, and
open caveats for every application. A newer timestamp, a convenient checkpoint,
or a preset that loads successfully is not evidence that it is
production-best. Sections follow the applications; the procedure and retention
rules at the end apply to all of them.

When the project has results to register, keep a code registry beside this
document (for example `src/apps/<application>/lineages.py`) holding the exact
preset names, run directories, and commits, and update both in the same
commit. Page builders and evaluation scripts read the code registry; people
read this page.

## Example

TODO(template): replace with the project's first accepted configuration.

| task | exact preset | run | commit | status |
|---|---|---|---|---|
| polynomial fit | `example-default` | not yet registered | | reference only |

Contracts: what the accepted configuration stores, the mapping and support its
checkpoint stamps, and which presets may load it.

## Accepting a replacement

1. Run the candidate from scratch on the current code with the exact preset.
2. Pass the gate recorded in the active plan, on the full horizon.
3. Record the preset, run directory, commit, and gate numbers here and in the
   code registry.
4. Move the superseded entry to a "Retired" list with the reason and date.

## Retention

Registered runs and their parent checkpoints are protected from cleanup and
are kept in the durable store described in `DATA.md`.

## Retired

None yet.
