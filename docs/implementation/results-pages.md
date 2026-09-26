# Results pages

Results pages are generated artifacts. Builders live under `scripts/pages/`;
generated HTML, JSON, copied panels, and derived images live under
`docs/pages/<page>/` and remain ignored.

`docs/pages/` is the local-preview exception to tracked documentation: the
builder is source, while its generated page tree is not.

Read [results presentation style](../style/results.md) before changing a page.

## Ownership

- `build_<page>.py` owns page content and artifact selection;
- a shared page module, when there is more than one page, owns structure and
  styling;
- `bundle.py` owns the standalone ZIP of a page directory;
- page-specific tracked templates and assets stay beside their builder;
- each generated page owns everything beneath its `docs/pages/<page>/` root.

Production code never imports page builders or figure scripts.

## Build workflow

1. Resolve source runs through the production registry or an explicit
   builder map.
2. Load metrics from full-precision arrays.
3. Validate component, reference, sample budget, shape, and checkpoint
   lineage.
4. Generate figures with one reference-derived exposure or normalization.
5. Write page HTML plus a manifest of source arrays and presentation
   metadata.
6. Bundle the page (`python scripts/pages/bundle.py docs/pages/<page>`).
7. Inspect the generated page and grep for unresolved template placeholders.

Run a builder from the repository root. Its module docstring and CLI are
authoritative for arguments and output path.

## Standalone pages

A page directory is self-contained: every runtime asset lives beneath it and
every URL is page-relative. Copying the directory, opening its `index.html`
from disk, or extracting its ZIP gives the same page. A page never reaches into
`outputs/`, another page, or a server-root path.

A live page prints its build stamp beside each source run's newest record, so
a stale build is visible on the first screen.

TODO(template): list the project's maintained pages and their builders here.
