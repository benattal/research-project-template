# Troubleshooting

Common issues and their fixes. Each entry states the symptom a user sees, the
cause, and the fix, so it can be found by searching for the symptom.

TODO(template): add entries as the project meets them. Keep each entry durable:
the investigation that found it belongs in the tracker.

---

## Progressively slower iterations

**Symptom**: each training iteration takes longer than the previous one;
eventually the process runs out of memory.

**Cause**: an array that persists across iterations (an accumulator, a running
statistic, a buffer) retains its autograd graph, so each iteration's graph
references all earlier ones.

**Fix**: evaluate and detach persistent state at the end of each iteration
(`x = x.detach()` in PyTorch, or the framework's equivalent).

---

## Out of memory after a snapshot

**Symptom**: training runs, then fails with an out-of-memory error shortly
after writing a snapshot or evaluation.

**Cause**: evaluation allocated large temporary buffers that the allocator
caches.

**Fix**: run evaluation under the framework's no-gradient context, release
references to its outputs, and collect garbage before resuming training.

---

## Compilation-cache errors when two jobs start together

**Symptom**: a job fails at startup with a cache or database error from a JIT
compiler, typically when another job started on the same node at the same
time.

**Cause**: both jobs write one shared on-disk compilation cache.

**Fix**: source `scripts/lib/job_cache.sh` and call `job_cache_setup` in every
launcher; each job then compiles into its own directory, removed at exit.

---

## "No module named ..." on a login node

**Symptom**: an import that works inside cluster jobs fails in an interactive
shell.

**Cause**: the dependency comes from an environment module that the launcher
loads, not from the virtual environment.

**Fix**: load the launcher's module set in a login shell before running, as
documented in [environments](../environments/index.md).
