# Long runs as chains of resumed links

A run longer than its host allows runs as consecutive links. Each link is a
fresh process that resumes the previous link's exact checkpoint, advances a
fixed number of steps, and saves a new checkpoint. The limit may be a login
node's CPU-time limit or a scheduler's wall-clock limit; the chain works the
same way under both. These entries cover what a checkpoint must hold, how a
driver chooses the next link, and how settings get lost between links.

## Chain long login-node work; do not move it to the scheduler for length alone

Login nodes that permit GPU work often limit each process to about one hour
of CPU time. Run longer work there as links, rather than routing it to the
scheduler only because it is long.

**Why.** A CPU-time limit (`RLIMIT_CPU`) counts the CPU time of all of a
process's threads, so a GPU job with busy host threads reaches it before one
hour of wall-clock time. At the soft limit the kernel sends `SIGXCPU`, which
ends the process with exit status 152 (128 + 24) and no Python traceback.

**How to apply.** Read the limits with `ulimit -St` and `ulimit -Ht` (in
seconds). If both are unlimited, the site enforces its limit another way;
read its documentation. Size each link to about half the limit, measured
from a short run's throughput, and save periodic checkpoints often enough
that a lost link costs little.

**Evidence.** Long GPU runs on a login node died with exit status 152, no
traceback and no out-of-memory record, while a twin under the same settings
kept running.

## An exact checkpoint holds everything the next step reads

A link boundary must not change the run. The checkpoint that ends a link
holds the model weights, their EMA copies, every optimizer's state, the
global step, the random streams' state, and any other value the next step
reads. Resuming from it reproduces the uninterrupted run.

**How to apply.**

- Save optimizer state with the weights; an optimizer resumed with fresh
  moment estimates takes different steps.
- Turn off actions that happen once at a phase boundary, such as resetting
  optimizer state, when a link resumes past that boundary.
- Keep "stop at step N" separate from the run's nominal horizon. A link that
  stops early must leave schedules defined over the full horizon unchanged.
- Write progress checkpoints into two alternating slots, so a crash during a
  write leaves the other slot intact.
- Mark a checkpoint resumable only after all of its files are written, and
  check that the optimizer-state file is nonempty before resuming from it.

## Choose the next link from what is on disk

A local chain driver loops: find the newest valid checkpoint of the
experiment, launch one link from it, and require that the link advanced the
checkpoint.

**How to apply.** Handle each point at which a link can die:

- no checkpoint: start the run;
- died during the first phase: resume from the newer of the two progress
  slots;
- died after the first phase ended but before the first chain checkpoint:
  start the second phase from the saved end of the first;
- otherwise: resume the newest chain checkpoint for another `LINK_STEPS`.

After each link, read the newest checkpoint again. If its step did not
advance, stop; otherwise a link that crashes at start relaunches forever.
Bound the loop count as a second guard.

```bash
for _ in $(seq 1 200); do
    read -r dir step <<< "$(newest_checkpoint "$EXPERIMENT")"
    [ -n "$dir" ] && [ "$step" -ge "$STOP" ] && exit 0
    run_link "$dir" "$step" || exit $?
    read -r _ next <<< "$(newest_checkpoint "$EXPERIMENT")"
    [ "${next:-0}" -gt "${step:-0}" ] ||
        { echo "link wrote no new checkpoint; stopping" >&2; exit 1; }
done
```

Check the GPU again before every link: another workload can claim the device
after the previous process releases it (see
[login nodes and shared hosts](login-nodes-and-shared-hosts.md)).

## Scheduler chains: read allocation-level accounting

A driver that continues a run across scheduler jobs polls each job's final
state with `sacct -X`, which reports the allocation record instead of the
batch and extern steps.

```bash
sacct -X -n -P -j "$JOB" --format=JobIDRaw,State,ExitCode
# <jobid>|COMPLETED|0:0
```

Take the first word of the state and strip a trailing `+` before comparing
(`CANCELLED by <uid>`, `CANCELLED+`). When `sacct` times out or fails, keep
polling the same job. A failed observation must never resubmit or cancel
anything.

## Treat a timeout as a link end and every other failure as a stop

Classify final states explicitly:

- `COMPLETED` with exit code `0:0`: validate the saved checkpoint, then
  submit the next link;
- `TIMEOUT`: resume from the newest periodic checkpoint that job wrote,
  losing at most one save interval; stop if it wrote none;
- `FAILED`, `CANCELLED`, `NODE_FAIL`, `OUT_OF_MEMORY`, `PREEMPTED`,
  `BOOT_FAIL`, `DEADLINE`, `REVOKED`, or `COMPLETED` with a nonzero exit
  code: stop the chain for diagnosis;
- any other state (`PENDING`, `RUNNING`): wait.

**Why.** A wall-clock timeout is an expected event in a chain, and the
link's periodic checkpoints are complete. A crash needs diagnosis before the
chain spends more allocations reproducing it.

## Validate a link's output before submitting the next

Before continuing, check the finished link's saved configuration and
checkpoint, not only its exit state:

- select the output directory by the job ID in its name, and require
  exactly one match;
- the executed horizon, seed and target file equal the chain's request;
- every explicit override the chain carries appears unchanged in the saved
  configuration;
- the checkpoint's step equals the stop the link was asked for;
- an intermediate stop carries a nonempty optimizer state and the resumable
  mark, and the final link carries the final result.

Allow the shared filesystem about a minute to show the files after
accounting reports completion. Retry only on missing files, and stop at once
on any mismatch.

## A chain driver forwards only the settings it knows about

Command-line overrides given to the first link are lost at the first resume
unless the driver forwards them. Register every per-run variation as a named
configuration, or make the driver forward an explicit list of overrides and
check each link's saved configuration for them.

**Why.** The driver builds each continuation command from its own
arguments. An override it does not know about is absent from the next link,
which runs with the default and reports no error.

**Evidence.** A learning-rate scale passed on the first link's command line
reverted to its default at the first continuation, without a message.

## Run the driver where it can submit, detached from the session

A scheduler chain driver submits every continuation, so it must run on a
host that accepts those submissions, and it must survive the interactive
session that started it.

```bash
nohup setsid bash -l chain_driver.sh ... > logs/chain_driver.log 2>&1 &
```

Have the driver write its own PID at start (`echo $$ > logs/driver.pid`):
`setsid` forks when its caller leads a process group, so `$!` can name a
parent that has already exited. Anything that has to outlive a session, such
as a driver or a loop that waits for jobs and then scores them, belongs in a
script in the repository; an ad-hoc background shell dies with the session
and leaves no record. When the host running the driver goes down, the driver
dies with it; restart it once the host is back.

## When jobs cannot submit jobs, pre-submit the chain with `afterok`

Some clusters reject `sbatch` from inside a compute job, with an error that
reads in part "cannot be done from this node or from a compute job; use a
login node instead". A link then cannot submit its own continuation. Submit
every link up front from a login node, each held by
`--dependency=afterok:<previous job>`. The scheduler releases a link only
when the previous one exits with status 0, so each link's exit status
controls the chain.

**How to apply.** The submitting script refuses to run inside a job and
passes the same arguments to every link:

```bash
[ -z "${SLURM_JOB_ID:-}" ] ||
    { echo "run from a login node, not inside a job" >&2; exit 1; }
prev=${FIRST_DEP:-}
for i in $(seq 1 "$N"); do
    jid=$(sbatch --parsable ${prev:+--dependency=afterok:$prev} \
                 link.sh "$CONFIG" "$@")
    echo "link $i/$N: job $jid"
    prev=$jid
done
```

Each link then decides its exit status:

- exit 0 after a clean stop at its time budget, or when the newest
  checkpoint advanced during the link;
- exit nonzero when no checkpoint advanced, so that a crash loop stops the
  chain instead of consuming every pre-submitted link;
- when training finishes, write a completion marker in the output
  directory; later links check it first and exit 0 at once instead of
  holding a GPU allocation to rediscover that the run is done;
- count links in the output directory and stop past a maximum;
- resolve the output directory from the named configuration, and fail the
  link if that lookup fails: a fallback name would start the run fresh in
  another directory partway through the chain.

**Why.** A pre-submitted chain has a fixed length and no driver process.
Exit statuses and files in the output directory are the only state that
passes between links. Links that depend on a failed link stay in the queue
with reason `DependencyNeverSatisfied` unless the site removes them; cancel
them with `scancel`.

## Derive the in-process time budget from the job's time limit

A trainer that stops cleanly at an epoch boundary (see
[checkpoints and resume](../training-engineering/checkpoints-and-resume.md#stop-at-an-epoch-boundary-before-the-time-limit))
needs a budget below the job's wall-clock limit. Read the limit inside the
job instead of hard-coding it, and leave a margin for the last checkpoint
and sample writes:

```bash
walltime_to_sec() {    # D-HH:MM:SS, HH:MM:SS, MM:SS or minutes; else 0
    awk -v t="$1" 'BEGIN { d = 0
        if (t ~ /-/) { split(t, p, "-"); d = p[1]; t = p[2] }
        n = split(t, a, ":")
        if (n == 3) s = a[1] * 3600 + a[2] * 60 + a[3]
        else if (n == 2) s = a[1] * 60 + a[2]
        else if (n == 1 && t ~ /^[0-9]+$/) s = t * 60
        else { print 0; exit }
        print d * 86400 + s }'
}
if [ -z "${MAX_RUNTIME_SEC:-}" ]; then
    limit=$(scontrol show job "$SLURM_JOB_ID" |
            grep -oE 'TimeLimit=[^ ]+' | head -1 | cut -d= -f2)
    wall=$(walltime_to_sec "$limit")
    budget=3300                          # fallback when the limit is unknown
    [ "${wall:-0}" -gt 0 ] && budget=$((wall - 300))
    [ "$budget" -lt 600 ] && budget=600
    export MAX_RUNTIME_SEC=$budget
fi
```

When the caller sets the budget, compare it with the requested wall time at
submission and warn if it is not smaller: the trainer's stop then never
fires, and the scheduler kills every link partway through an epoch.

**Evidence.** A fixed default budget of 3,300 s, sized for one-hour links,
capped six-hour links at about 55 minutes of work without any message.
