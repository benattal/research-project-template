#!/usr/bin/env bash
# Build HTML for every maintained doc plus the generated API reference.
#
# Usage: bash scripts/infra/build_docs.sh
#   DOCS_VENV=<dir>  override the documentation-only virtualenv location
#
# Output: docs/pages/documentation/ (gitignored) and a standalone ZIP at
# docs/pages/documentation/downloads/documentation-site.zip.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${DOCS_VENV:-${TMPDIR:-/tmp}/${USER}/docs_venv}"

# Create the docs venv on first use; it never reuses a training environment.
if [ ! -x "$VENV/bin/pip" ]; then
    mkdir -p "$(dirname "$VENV")"
    # stdlib venv needs ensurepip; fall back to virtualenv where it is missing
    python3 -m venv "$VENV" 2>/dev/null || virtualenv -p python3 "$VENV"
fi

# Install the build toolchain; a fast no-op when already satisfied.
"$VENV/bin/pip" install -q \
    mkdocs mkdocs-material 'mkdocstrings[python]' mkdocs-gen-files mkdocs-literate-nav

# Build into a temporary directory, then replace the generated tree in one
# move. No top-level site/ directory is created.
cd "$REPO"
# Silence the MkDocs 2.0 migration notice a plugin prints on every build, so
# the build's own warnings stay visible. (mkdocs-material prints a separate
# banner of its own that this variable does not affect.)
export DISABLE_MKDOCS_2_WARNING=true
BUILD_DIR="$(mktemp -d)"
trap 'rm -r -- "${BUILD_DIR:?}"' EXIT
"$VENV/bin/mkdocs" build --site-dir "$BUILD_DIR/site"

TARGET="$REPO/docs/pages/documentation"
if [ -e "$TARGET" ]; then
    rm -r -- "${TARGET:?}"
fi
mkdir -p "$REPO/docs/pages"
mv "$BUILD_DIR/site" "$TARGET"

# TODO(template): add a check for each page a reader must always find.
test -f "$TARGET/index.html"
test -f "$TARGET/style/code.html"
test -f "$TARGET/environments/index.html"
test -f "$TARGET/historical/index.html"
test -f "$TARGET/api/src/core/training/schedules.html"

# Keep the site downloadable and independent of this checkout.
"$VENV/bin/python" scripts/pages/bundle.py "$TARGET" \
    --archive-name documentation-site.zip --root-name documentation
test -s "$TARGET/downloads/documentation-site.zip"

echo
echo "Docs built. Open: file://$TARGET/index.html"
