"""Generate one API-reference page per ``src`` module at MkDocs build time.

- Runs inside the mkdocs-gen-files plugin; pages are virtual, nothing is
  written to ``docs/``.
- Also emits ``api/SUMMARY.md``, which mkdocs-literate-nav turns into the API
  nav subtree.
- mkdocstrings reads the source statically, so the build never imports the
  project or initializes a GPU.
"""

from pathlib import Path

import mkdocs_gen_files

# repo root = parent of scripts/
root = Path(__file__).resolve().parents[2]
src = root / "src"

nav = mkdocs_gen_files.Nav()

for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(root).with_suffix("")
    parts = list(module_path.parts)

    if "__pycache__" in parts:
        continue
    # a package's page comes from its __init__; drop the suffix
    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = Path(*parts, "index.md")
    # command shims have no API surface
    elif parts[-1] == "__main__":
        continue
    else:
        doc_path = module_path.with_suffix(".md")

    full_doc_path = Path("api", doc_path)
    ident = ".".join(parts)

    # nav key is the dotted-path tuple; value is the page path relative to api/
    nav[tuple(parts)] = doc_path.as_posix()

    # the page body is one mkdocstrings directive; griffe fills in the rest
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        fd.write(f"# `{ident}`\n\n::: {ident}\n")

    # "edit this page" points at the real source file
    mkdocs_gen_files.set_edit_path(full_doc_path, Path("..") / path.relative_to(root))

with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
