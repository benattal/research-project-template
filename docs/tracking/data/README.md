# Tracking data

Machine-readable artifacts supporting the human tracking documents: output
inventories, reviewed cleanup manifests, gate threshold tables, and similar
JSON records.

Scripts under `scripts/infra/` or `scripts/scoring/` own these schemas and are
named in each file's first field. JSON data belongs here rather than beside the
Markdown trackers. Arrays and model weights never belong here; see `DATA.md`.

TODO(template): list each data file and the script that owns its schema.
