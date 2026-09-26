"""Bundle one generated page directory into its own downloads folder.

Examples:

    python scripts/pages/bundle.py docs/pages/<page>
    python scripts/pages/bundle.py docs/pages/documentation \
        --archive-name documentation-site.zip --root-name documentation

The archive excludes ``downloads/`` so rebuilding cannot recursively include
an older bundle. Every other page-owned file is stored beneath one root
directory named after the page unless ``--root-name`` overrides it.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def bundle_page(page_directory: Path, archive_name: str = "",
                root_name: str = "") -> Path:
    """Create and return a standalone ZIP for ``page_directory``."""
    page = page_directory.resolve()
    if not (page / "index.html").is_file():
        raise FileNotFoundError(f"generated page has no index.html: {page}")
    archive_name = archive_name or f"{page.name}-page.zip"
    root_name = root_name or page.name
    downloads = page / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    archive_path = downloads / archive_name

    self_download = f"downloads/{archive_name}"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(page.rglob("*")):
            if not path.is_file() or downloads in path.parents:
                continue
            member = Path(root_name) / path.relative_to(page)
            if path == page / "index.html":
                source = path.read_text(encoding="utf-8")
                # The hosted page links to its own archive. The extracted copy
                # cannot contain itself, so its link points back at the page.
                source = source.replace(
                    f'href="{self_download}"',
                    'href="./index.html" '
                    'title="This extracted page is already standalone"')
                archive.writestr(member.as_posix(), source)
            else:
                archive.write(path, member)
    return archive_path


def main() -> None:
    """Parse the page path and rebuild its standalone archive."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("page_directory", type=Path)
    parser.add_argument("--archive-name", default="")
    parser.add_argument("--root-name", default="")
    args = parser.parse_args()
    print(bundle_page(args.page_directory, args.archive_name, args.root_name))


if __name__ == "__main__":
    main()
