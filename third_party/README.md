# Third-party code

Standalone baselines, reference implementations, and vendored dependencies.
Maintained source under `src/` never imports from here; a baseline runs
through its own entry point and environment.

Prefer a git submodule pinned to a commit (`git submodule add <url>
third_party/<name>`) over a copied tree. When code must be vendored, record
its upstream URL, commit, and license in a README beside it, and keep local
modifications in separate commits.

TODO(template): list each third-party component, why it is here, and how to
run it.
