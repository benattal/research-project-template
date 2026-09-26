# Integration tests

End-to-end runs of an application and device integration gates: a short
preset trained into a temporary directory, repeated-run reproducibility, and
any test that needs a GPU.

A GPU test selects its device explicitly, never at import time of a shared
module, and runs on a free GPU bound by the launcher with per-job caches
(`scripts/lib/job_cache.sh`). Skip, rather than fail, when an optional
dependency is absent (`pytest.importorskip`).
