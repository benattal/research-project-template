---
name: upload-data
description: Upload this project's run outputs and large input data to the project's object-storage bucket, or pull them back. Use when asked to push, back up, archive, sync, or fetch run outputs, checkpoints, predictions, or data. Covers stripping intermediate per-iteration snapshots before upload and verifying before any local cleanup.
---

# Upload project data to object storage

Results and large inputs live in an S3-compatible bucket under the project
prefix, never in git. `DATA.md` in the repository root is the full
human-facing version of this skill and names the placeholders:
`<remote>`, `<bucket>`, `<project>`.

TODO(template): replace the placeholders in `DATA.md` once; this skill reads
them from there.

## 1. Check the remote

```bash
rclone lsd <remote>:<bucket> 2>/dev/null && echo "remote OK"
```

If it fails, follow "Setup" in `DATA.md`. Never print credentials into logs,
commit them, or upload them to the bucket.

## 2. Strip intermediate snapshots

Upload finals, not trajectories, whenever the run has a final artifact. Build
an explicit manifest with the filter in `DATA.md` ("Pushing new results") and
read its file count before uploading. A run with no final artifact is
uploaded in full.

## 3. Upload with copy, never sync

```bash
rclone copy . <remote>:<bucket>/<project>/outputs/ \
    --files-from "$MANIFEST" --progress --transfers=16
```

`sync` deletes remote files that are absent locally and would wipe other
machines' results.

## 4. Verify

```bash
rclone check ./outputs/ <remote>:<bucket>/<project>/outputs/ \
    --files-from "$MANIFEST" --one-way
```

Report the object count and bytes uploaded. Do not delete or move local
copies as part of an upload: cleanup is a separate action that the user
authorizes, following "Verify before local cleanup" in `DATA.md`.
