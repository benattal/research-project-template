# Staging a Python environment onto compute nodes

A cluster that does not provide a project's dependencies as environment
modules needs the Python environment shipped to every node that runs a job.
The pattern that held up is one compressed archive on the shared filesystem,
extracted once per node onto local disk under a lock, checked with a real
import before each use, and built so that it does not depend on the node's
system Python. Each entry below is one piece of that pattern and the failure
that motivated it.

## Ship one archive and extract it to node-local disk

Keep only a compressed archive of the virtualenv on the shared filesystem.
Each job extracts it onto node-local disk, or reuses the copy an earlier job
on the same node extracted, and runs the interpreter from there.

**Why.** A GPU deep-learning virtualenv holds tens of thousands of small
files, and every import stats and reads many of them. A network filesystem
serves small-file metadata slowly, and many jobs importing at once load its
metadata server for every user. One large file transfers at close to full
bandwidth. Copying the tree file by file (`rsync`, `cp -r`) pays the same
small-file cost as importing from it.

**How to apply.**

```bash
ARCHIVE=$SHARED/envs/project_venv.tar.zst   # built once; see the last entries
STAGE=/tmp/$USER/project_venv               # refined in the next entry
mkdir -p "$STAGE"
zstd -dc "$ARCHIVE" | tar -C "$STAGE" -xf -
PY=$STAGE/venv/bin/python                   # call it directly; no activate
```

**Evidence.** The environment held about 26,000 files and needed about 10 GB
extracted. The shared filesystem moved about 350 MB/s on large files and an
effective 120 KB/s on workloads of many small files.

## Key the stage directory by archive identity and modification time

Derive the stage directory from the archive's name and modification time, so
two different archives, or two versions of one archive, never extract into
the same directory.

**Why.** With one shared stage directory, a job that resolves a different
archive (a portable build, a build for another GPU generation, a rebuilt
archive) finds a ready marker that does not match its archive and deletes the
tree to re-extract it. Another job may be running from that tree. A lock
serializes one extraction against another; it cannot protect a job that is
already running and holds no lock. When the path encodes the archive
identity, a different archive extracts somewhere else and never deletes a
live tree.

**How to apply.** Every launcher that stages the environment must use the
same rule, so put it in one sourced helper. Old stage directories stay on
the node after the archive changes; remove them only when no job on the node
is using them.

```bash
STAMP=$(stat -c %Y "$ARCHIVE")
STAGE=/tmp/$USER/$(basename "$ARCHIVE" .tar.zst)_$STAMP
VENV=$STAGE/venv
READY=$STAGE/.ready
mkdir -p "$STAGE"
```

**Evidence.** Jobs finished training, then a second `srun` step in the same
job died at import with

```text
RuntimeError: Unable to find torch_shm_manager at <stage>/venv/.../torch/bin/torch_shm_manager
```

The environment had been deleted under the live job. Ten runs of one sweep
lost their final evaluation this way, and a diagnostic launcher that kept the
old shared stage path reproduced the failure after the training launcher was
fixed.

## Serialize extraction with a lock

Hold an exclusive `flock` on a file inside the stage directory while checking
the ready marker and extracting.

**Why.** Two jobs that start together on one node both find the marker
missing and both extract into the same directory. The second job's
`rm -rf` before extracting removes files the first job is writing.

**How to apply.**

```bash
exec 9>"$STAGE/.lock"
flock 9                  # waits while another job on this node is staging
# check the marker, probe imports, extract if needed (next entry)
flock -u 9
```

**Evidence.** Two jobs that started together on one node raced this way, and
one lost its environment mid-extraction.

## Trust a real import, not a ready marker

Before reusing a staged environment, import its heaviest packages with the
staged interpreter. Re-extract when the import fails, even if the marker says
the stage is current.

**Why.** A marker proves that one extraction finished; it says nothing about
the files afterwards. On some nodes, `/tmp` cleaners remove files by access
time. They leave directories in place, and files that every job reads, such
as the marker, keep a recent access time and survive. The files of a rarely
imported package go first. Other clusters never purge `/tmp`; plan for
both.

**How to apply.** Run this while holding the lock:

```bash
ok=0
if [ "$(cat "$READY" 2>/dev/null)" = "$STAMP" ] &&
   "$VENV/bin/python" -c "import torch, tqdm" >/dev/null 2>&1; then
    ok=1
fi
if [ "$ok" != 1 ]; then
    rm -rf -- "${VENV:?}"
    zstd -dc "$ARCHIVE" | tar -C "$STAGE" -xf -
    echo "$STAMP" > "$READY"
fi
```

**Evidence.** A job died at import with one small package's files removed
from the staged tree and the ready marker intact.

## Check free local disk before extracting

Measure free space on the stage filesystem before extracting, and exit with a
message that names the node and the shortfall.

**Why.** `tar` on a full disk exits with status 2 and a terse message, and a
partly extracted tree then looks like a broken environment. Local `/tmp`
sizes differ across a heterogeneous cluster.

**How to apply.** Give each placement check its own exit status, so a
wrapper can tell a placement failure from a failed run.

```bash
need_kb=$((12 * 1024 * 1024))            # extracted size plus margin
avail_kb=$(df -Pk "$STAGE" | awk 'NR == 2 {print $4}')
if [ "${avail_kb:-0}" -lt "$need_kb" ]; then
    echo "ERROR: $(hostname) has $((avail_kb / 1048576)) GB free in /tmp;" \
         "staging needs 12 GB" >&2
    exit 3
fi
```

**Evidence.** Without the check, jobs placed on nodes with small `/tmp`
partitions failed with a bare `tar` exit status 2.

## A virtualenv's interpreter is a symlink to the build host's Python

`python -m venv` and `virtualenv` create `bin/python` as a symlink to the
interpreter that built the environment, and install packages under
`lib/pythonX.Y/site-packages`. On a node without that `X.Y`, the symlink
dangles or resolves to another version, which does not search
`lib/pythonX.Y`, and nothing imports.

**How to apply.** Compare the environment's version with the interpreter the
node resolves, before anything else, and report the node. The lasting fix is
the portable build in the next entry.

```bash
want=$(basename "$(ls -d "$VENV"/lib/python3.* | head -1)")    # python3.10
have=$("$VENV/bin/python" -c \
    'import sys; print("python%d.%d" % sys.version_info[:2])' 2>/dev/null ||
    echo none)
if [ "$want" != "$have" ]; then
    echo "ERROR: $(hostname) resolves $VENV/bin/python to $have" \
         "($(readlink -f "$VENV/bin/python")); the venv needs $want" >&2
    exit 4
fi
```

**Evidence.** Most GPU nodes of one heterogeneous cluster shipped only
Python 3.12, while the environment was built with 3.10. Jobs placed there
left empty `.out` and `.err` files, because a `set -e` script died on an
empty glob before printing anything (see [shell safety](shell-safety.md)).
A sweep whose jobs all landed on one such node looked like a flaky node, and
a node that seemed to "eat jobs during staging" had the same cause.

## Bundle the interpreter and standard library into the archive

Make the environment independent of the node's Python: copy the real
interpreter binary and its standard library into the virtualenv, point
`bin/python` at the copy through relative symlinks, and set `home` in
`pyvenv.cfg` to the copy's directory.

**Why.** The CPython binary links only against libraries every Linux node
has (`libc`, `libm`, `libz`, `libexpat`; confirm with `ldd`), so a copied
binary runs anywhere. The standard library must sit where that binary
searches for it: under `base/<platlibdir>/pythonX.Y`, where `<platlibdir>`
is the build interpreter's `sys.platlibdir` (`lib` on Debian and Ubuntu
builds, `lib64` on Fedora and RHEL builds). When the library is not there,
the interpreter falls back to its compiled-in prefix, `/usr`, without an
error. That works on nodes that have the system Python and fails on the
nodes the bundle was built for.

**How to apply.** Build on a node that has the source interpreter. Write the
result as a new archive under a new name, and rename it into place only
after it is complete, so jobs running from the old archive are unaffected.

```bash
SYS_PY=$(readlink -f "$(command -v python3.10)")
STDLIB=$("$SYS_PY" -c 'import sysconfig; print(sysconfig.get_paths()["stdlib"])')
LIBDIR=$("$SYS_PY" -c 'import sys; print(sys.platlibdir)')
mkdir -p "$VENV/base/bin" "$VENV/base/$LIBDIR"
cp "$SYS_PY" "$VENV/base/bin/python3.10"
cp -a "$STDLIB" "$VENV/base/$LIBDIR/python3.10"
for n in python python3 python3.10; do
    ln -sfn ../base/bin/python3.10 "$VENV/bin/$n"
done
cat > "$VENV/pyvenv.cfg" <<CFG
home = $RUNTIME_VENV/base/bin
include-system-site-packages = false
CFG
tar -C "$BUILD" -cf - venv | zstd -T0 -3 -o "$OUT.tmp" -f
mv "$OUT.tmp" "$OUT"
```

A node that has `python3.X` but not the distribution's `python3.X-venv`
package cannot run `python3.X -m venv`; it fails with "you need to install
the python3-venv package". Create the environment with `--without-pip` and
bootstrap pip with `get-pip.py`, which needs no root access.

**Evidence.** The portable archive made the Python-3.12-only nodes usable. A
layout test with the standard library under `lib` for a `lib64` build loaded
`/usr/lib64/python3.9/os.py` instead of the bundled copy, with no warning.
With the correct directory name, the same test tree kept working after it
was moved to a path other than the one `home` named, because the interpreter
found its library from the relative symlink's target.

## Put the torch wheel's bundled CUDA libraries on the loader path

Add every `site-packages/nvidia/*/lib` directory of the staged environment to
`LD_LIBRARY_PATH` before running an extension that loads CUDA libraries at
run time.

**Why.** Pip-installed torch ships CUDA libraries such as `libnvrtc.so.12`
under `site-packages/nvidia/` and loads them by path itself. A separately
built extension that opens them later, such as tiny-cuda-nn compiling
kernels at run time, searches only the loader path.

**How to apply.** Ask the interpreter for its site-packages directory rather
than hard-coding a Python version, and tolerate an empty glob:

```bash
SITE=$("$PY" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
NVLIBS=$(ls -d "$SITE"/nvidia/*/lib 2>/dev/null | tr '\n' ':' || true)
[ -n "$NVLIBS" ] || echo "WARNING: no nvidia/*/lib under $SITE" >&2
export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:-}"
```

**Evidence.** Without these directories on the loader path, tiny-cuda-nn
could not load `libnvrtc.so.12` for its run-time kernel compilation.
