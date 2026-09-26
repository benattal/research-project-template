# Shell safety in job and cleanup scripts

Several lost runs and one deleted directory tree came from shell mechanics
rather than from the programs being run: `set -e` ending a job without
output, a second `trap` replacing the first, a `pkill` pattern that matched
its caller, and a `find` guard that failed without a message. Each entry
states the rule, then the mechanism.

## `set -euo pipefail` can end a job before it prints anything

Under `set -e` with `pipefail`, an assignment from a command substitution
whose pipeline fails ends the script at that line, with no message. Give
probe commands whose failure is informative an explicit fallback
(`|| true`), then test the result and report the problem. Print a banner
(host, job ID) as the first action of every job script, so an empty log
means the script never started.

```bash
echo "job ${SLURM_JOB_ID:-local} on $(hostname) at $(date -Is)"
NVLIBS=$(ls -d "$SITE"/nvidia/*/lib 2>/dev/null | tr '\n' ':' || true)
[ -n "$NVLIBS" ] || echo "WARNING: no bundled CUDA libraries under $SITE" >&2
```

**Evidence.** An unmatched glob made `ls` exit with status 2 inside such an
assignment. The jobs left zero-byte `.out` and `.err` files and looked like
a flaky node.

## Compose traps; a second `trap` replaces the first

`trap cmd EXIT` replaces any handler already armed for `EXIT`. When a sourced
helper and the script both need exit handlers, append to the armed handler
instead:

```bash
trap_append() {        # usage: trap_append <function> <signal>...
    local cmd=$1 sig prev
    shift
    for sig in "$@"; do
        prev=$(trap -p "$sig")
        prev=${prev#trap -- \'}
        prev=${prev%\' $sig}
        if [ -n "$prev" ]; then trap -- "$prev; $cmd" "$sig"
        else trap -- "$cmd" "$sig"; fi
    done
}
trap_append sync_outputs EXIT TERM INT      # runs first
trap_append cleanup_caches EXIT TERM INT    # runs second
```

Composed handlers run in the order they were armed, so arm the output copy
before the cache cleanup. Keep handler bodies to function names: a body that
contains a single quote does not survive the `trap -p` round trip. A helper
that normally arms its own trap needs an option to defer it, so the caller
can place it after its own.

## A handler on `EXIT` and `TERM` runs twice when the job is signalled

On `SIGTERM`, bash runs the `TERM` handler; when the script then exits, the
`EXIT` handler runs as well. Make every handler idempotent: check whether
its work is done, and clear the list of paths it deletes once they are
deleted. A `TERM` handler that returns without calling `exit` also lets the
script continue after the interrupted command. [Output retention and
cleanup](output-retention-and-cleanup.md) describes the false failure report
a non-idempotent handler produced.

## Kill by PID; never `pkill -f` a pattern from your own command line

`pkill -f pattern` and `pgrep -f pattern` match against full command lines,
including that of the shell running the `pkill`. A tool or wrapper that runs
commands as `bash -c "<command>"` carries the pattern in its own arguments,
so it kills itself.

**How to apply.** Record the PID at launch and kill that PID. To find a
process, use the bracket form of the pattern, which does not match its own
command line, and kill the listed PIDs after reading them:

```bash
nohup python -m app ... > logs/run.log 2>&1 &
echo $! > logs/run.pid
ps -eo pid,cmd | grep "[r]un_chain"
kill "$(cat logs/run.pid)"
```

**Evidence.** Both `pkill -f <script>` and a loop over `pgrep -f <script>`
killed the calling shell mid-command, which exited with status 144.

## `find` may be `bfs`, which rejects relative `-newermt` dates

Check `find --version` on a new machine. `bfs`, which some systems install as
`find`, rejects relative date strings such as `-newermt '-2 days'` or
`-newermt '2 days ago'` with `Invalid timestamp`; GNU `find` accepts them.
Use `-mtime -2` or `-mmin -45`, or an absolute timestamp:

```bash
find "$DIR" -mindepth 1 -maxdepth 1 -newermt "$(date -d '-2 days' -Iseconds)"
```

**Evidence.** A cleanup meant to keep directories modified in the last two
days ran its guard as `$(find ... -newermt '-2 days' 2>/dev/null)`. The
error vanished, the guard saw "nothing recent", and all 127 directories were
deleted, recent ones included.

## Do not let a guard read an error as "nothing found"

`2>/dev/null` inside a command substitution that feeds a guard turns a
failure into an empty result, and for a cleanup guard, empty usually means
"safe to delete". Check the command's exit status, and run every delete loop
first as a dry run that prints its keep and delete counts:

```bash
recent=$(find "$d" -maxdepth 2 -mmin -45 -print -quit) ||
    { echo "guard failed for $d" >&2; exit 1; }
```

## Guard every variable in a delete path

`rm -rf $T/*` with `T` unset or empty becomes `rm -rf /*`, and a glob delete
also removes files nobody has looked at. Prefer writing into a fresh
directory to clearing an old one. When a delete is needed, name the files and
make an empty variable fatal:

```bash
rm -f -- "${T:?}/paper.aux" "${T:?}/paper.log"
rm -rf -- "${OUT_LOCAL:?}"
```

Scripts that delete computed paths should also check the path's form before
deleting: a `case` on the basename, or a check that the target is a direct
child of the expected directory.

## Pass commands to launchers as argument vectors

A generic launcher should take the command as `"$@"` and run it unchanged,
not as a string that it `eval`s. Callers then need no quoting tricks, and
the command's exit status reaches the scheduler unchanged. Log the command
with `%q`, so the log shows what ran, quoted:

```bash
printf 'Command:'; printf ' %q' "$@"; printf '\n'
if "$@"; then
    echo "done at $(date -Is)"
else
    status=$?
    echo "failed with exit status $status" >&2
    exit "$status"
fi
```
