# Building the documentation site

Markdown under `docs/` is the source of truth. The HTML site is generated with
MkDocs; never hand-maintain an HTML copy beside a Markdown file.

```bash
bash scripts/infra/build_docs.sh
```

The command uses a documentation-only virtual environment (override its
location with `DOCS_VENV=...`) and writes the static site to the gitignored
`docs/pages/documentation/` tree. Open
`docs/pages/documentation/index.html` in a browser. The API reference is built
by static analysis (`scripts/infra/gen_api_pages.py`), so the build never
imports the project or initializes a GPU.

The build also creates
`docs/pages/documentation/downloads/documentation-site.zip`, a standalone
archive of the complete site under one `documentation/` root. It excludes the
downloads directory itself so repeated builds cannot nest prior bundles.

The site includes concepts, implementation, style, environments, papers,
plans, trackers, the archive, and every generated source API page. Only
`docs/pages/` is excluded, because it is generated output and including it
would recursively ingest the site.

## Adding a document

1. Write the Markdown file in the directory that owns its responsibility.
2. Link it from [the index](../index.md).
3. Add it to the `nav` in `mkdocs.yml`.
4. Run the build and read its warnings: unresolved links and pages missing
   from the nav are reported there.

Published experiment-result pages are a separate artifact family. Build those
with `scripts/pages/build_*.py` and follow [results pages](results-pages.md);
do not place them in the MkDocs source tree.
