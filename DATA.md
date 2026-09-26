# Experiment data and outputs

Experiment inputs that are too large for git, and every run's results, live
in an S3-compatible object store (Cloudflare R2, AWS S3, MinIO, ...) under one
project prefix. They are **not stored in git**. `outputs/`, `logs/`, and large
assets under `data/` are gitignored; the bucket is the durable copy. Scratch
space gets purged; the bucket does not.

TODO(template): fill in the three placeholders used throughout this page:

- `<remote>`: the rclone remote name configured on each machine;
- `<bucket>`: the bucket;
- `<project>`: this project's prefix inside the bucket.

Do not write credentials, account IDs, or endpoints into this file or anything
else that is tracked.

---

## Quick start

```bash
# one-time setup (see "Setup" below), then pull ONE run, not the whole tree:
rclone copy <remote>:<bucket>/<project>/outputs/<run-dir>/ \
            ./outputs/<run-dir>/ --progress
```

---

## Setup (once per machine)

### 1. Credentials

Keep the access key, secret, and endpoint in a credentials file outside any
repository, or in `.data-credentials` at the repository root (gitignored,
mode 600). Copy it from another machine or mint new keys in the storage
provider's dashboard; it is not recoverable from the repository.

```bash
chmod 600 .data-credentials
set -a; . ./.data-credentials; set +a   # exports S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY, S3_ENDPOINT
```

### 2. rclone remote

`rclone` is the supported tool. Check for a working remote before writing a
new one, so an existing configuration is not overwritten:

```bash
rclone lsd <remote>:<bucket> 2>/dev/null && echo "remote OK"
```

If that fails, generate the configuration from the credentials:

```bash
set -a; . ./.data-credentials; set +a
mkdir -p ~/.config/rclone
cat >> ~/.config/rclone/rclone.conf <<EOF
[<remote>]
type = s3
provider = Cloudflare
access_key_id = $S3_ACCESS_KEY_ID
secret_access_key = $S3_SECRET_ACCESS_KEY
endpoint = $S3_ENDPOINT
region = auto
EOF
chmod 600 ~/.config/rclone/rclone.conf
rclone lsd <remote>:<bucket>        # verify
```

Set `provider` to match the storage service.

---

## Layout under `<bucket>/<project>/`

| Prefix | Contents |
|---|---|
| `outputs/` | One directory per run, named `<experiment>_<YYYYmmdd_HHMMSS>_<job>`: final artifacts, checkpoints, configs, provenance, metrics. |
| `data/` | Inputs needed to *run* experiments that are too large for git. |
| `tables/` | Aggregate score tables. |

Inside a run directory the artifacts that survive are typically
`config.json`, `provenance.json`, `metrics.json`, final predictions and
checkpoints, and reference renders or arrays.

---

## Pulling

**Pull selectively.** Inspect sizes before copying a large prefix:

```bash
rclone size <remote>:<bucket>/<project>/outputs --fast-list

# list runs of one experiment
rclone lsd <remote>:<bucket>/<project>/outputs/ | grep <experiment>

# one run
rclone copy <remote>:<bucket>/<project>/outputs/<run-dir>/ \
            ./outputs/<run-dir>/ --progress

# only the cheap files (configs and metrics) across every run
rclone copy <remote>:<bucket>/<project>/outputs/ ./outputs/ \
            --include '*/config.json' --include '*/metrics.json'

# input data needed to run anything
rclone copy <remote>:<bucket>/<project>/data/ ./data/ --progress
```

`copy` never deletes; it skips files already present with matching size and
modification time, so re-running is cheap and interrupted transfers resume.

---

## Pushing new results

```bash
rclone copy ./outputs/<run-dir>/ \
            <remote>:<bucket>/<project>/outputs/<run-dir>/ \
            --progress --transfers=16
```

- Use `rclone copy`, **not `rclone sync`**. `sync` deletes anything on the
  remote that is absent locally, which wipes other machines' results.
- Push from the checkout where runs actually land.
- **Strip intermediate iterates before pushing** (see the retention policy
  below). This filter keeps every file whose path *below* the run directory
  has no 4-or-more-digit iteration index, leaving the run directory's own
  timestamp alone:

```bash
cd outputs
MANIFEST="$(mktemp)"
find <run-dir> -type f -printf '%P\n' \
  | awk '{p=$0; i=index(p,"/"); rest=(i?substr(p,i+1):p); if (rest !~ /[0-9]{4,}/) print "<run-dir>/" p}' \
  > "$MANIFEST"
rclone copy . <remote>:<bucket>/<project>/outputs/ \
        --files-from "$MANIFEST" --progress --transfers=16
```

### Verify before local cleanup

Take the upload manifest as the contract. Check the same paths against the
bucket, then record the remote object count and bytes before touching the
local copy:

```bash
rclone check ./outputs/ <remote>:<bucket>/<project>/outputs/ \
    --files-from "$MANIFEST" --one-way
rclone size <remote>:<bucket>/<project>/outputs/<run-dir>/
```

`rclone check` compares hashes where the backend supplies a usable hash and
reports any missing or changed object. Do not infer success from
`rclone copy`'s exit code alone. Keep the manifest until verification
finishes.

Cleanup is a separate, explicit action. First confirm that no active job uses
the run, that no live plan or page build needs local access, and that the
remote check passed. Move eligible runs into a same-filesystem quarantine
directory, record their paths and byte counts, and delete the quarantine only
after a later review. Never remove an incomplete run merely because it lacks a
final artifact; its intermediate files may be its only result.

---

## What gets stored (retention policy)

**Stored:** final results, trained models and checkpoints, predictions,
reference renders and arrays, run configurations and provenance, timings,
metrics and score tables, and the input data needed to *run* experiments.

**Not stored: intermediate training outputs, whenever a final artifact
exists.** A per-iteration snapshot that a final artifact supersedes is
reproducible from the configuration plus the final state. Exclude
per-iteration directories and files (`iter_<NNNNN>/`, `<name>_<NNNNN>.png`,
`<name>_<NNNNN>.npy`) when the run has the corresponding final artifact.

**Exception: runs with no final artifact keep their full iteration history.**
For those the iterates *are* the result. The rule is "drop intermediates *if*
a final exists", not "drop intermediates".

**Also not stored:**

- logs and scheduler output (`logs/`, `*.log`, `*.err`, `*.out`);
- code: git already has it;
- virtual environments and environment archives;
- compilation caches.

---

## Notes

- `copy` is idempotent; it is safe to re-run after an interrupted transfer.
- Keep the bucket private.
- Never commit `.data-credentials`. It is gitignored; keep it that way.
- The `upload-data` agent skill in `.claude/skills/` is the short version of
  this page for coding agents.
