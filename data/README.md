# Data

Repository inputs live here:

- one directory per dataset or scene, `data/<name>/`, holding tracked
  definitions and small source assets. Large downloaded assets share the same
  tree but remain ignored;
- `resources/`, an ignored machine-local cache of pretrained weights and other
  downloaded resources, fetched ahead of time so compute nodes need no
  outbound network access.

TODO(template): name the download command, for example:

```bash
rclone copy <remote>:<bucket>/<project>/data/ ./data/ --progress
```

Do not commit downloaded assets, model weights, or NumPy arrays. Any
deliberately tracked binary exception is listed by exact path in `.gitignore`.
No executable code lives under `data/`; asset generators belong in `scripts/`.
