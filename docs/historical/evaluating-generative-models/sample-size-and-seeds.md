# Sample size, seeds and provenance

A metric value is an estimate. It has a floor set by the number of samples,
a spread set by the training seed, and a meaning set by the evaluator code,
the reference set and the weights it was computed from. A comparison is
valid only when these are fixed across the arms and the difference exceeds
the noise. This file covers finite-sample floors, seed spread, checkpoint
selection, EMA weights and the pipeline checks that confirm an evaluation
measured what it claims.

## Every distance has a finite-sample floor

Compute each distance between two independent clean draws of the evaluation
size, and report that floor beside every model value. A model value at or
below the floor is indistinguishable from the target.

**Why.** Empirical distances between two finite samples of the same
distribution are positive. Their size depends on the sample count, the
dimension and the metric. Without the floor, a small value looks like a
recovery and a gap between two small values looks like a ranking.

**How to apply.** Draw the two clean sets with seeds different from the
generator's evaluation seed. Recompute the floor whenever the sample count,
the projection count or the bandwidth rule changes. Treat values below the
floor as ties with the target, not as better than it.

**Evidence.** On a toy spiral with 8,192 samples, the clean-versus-clean
floors were SWD .0054, energy distance .00027, MMD .00034 and Fréchet
distance .000056. On 1D mixture signals, a clean-trained model scored SWD
.0435 and energy distance .0334 against floors of .0460 and .0415: a tie with
the data, not a better-than-data result. In a different setting of the
same evaluator the SWD floor was .0546, so a floor belongs to one
evaluation setting.

## Fix sample count, reference set and evaluator across a comparison

Compare values only when they were computed with the same number of
generated samples, the same reference set, the same random projections and
the same evaluator version. When any of these changes, re-evaluate every
arm in the table.

**Why.** Most distances are biased by the sample count, so the same model
gives different values at different sizes. A median-heuristic bandwidth
depends on the pooled samples, so it shifts with the arms being compared.
A change to the evaluator code can change values without any change to the
models.

**How to apply.** Put the sample count, the reference seed and the evaluator
version in every table header. Keep one frozen reference set per task.
Re-score old checkpoints with the current evaluator before comparing them
with new ones.

**Evidence.** The same model scored SWD .062 with 512 samples per
condition and .044 with 1,024 class-matched samples. Real ImageNet images
scored FID 26.2 with 2,000 samples and 9.5 with 10,000 against the same
reference statistics. A schedule looked better than the baseline until the
baseline, scored by an older evaluator, was re-evaluated with the current
one.

## Compare the seed spread with the effect size

Run each arm with at least three training seeds and report the mean and
standard error. When arms share seeds, report the paired difference and its
standard error. A single-seed win smaller than the seed spread is not a
result.

**Why.** Training seeds change initialization, data order and sampling
noise. For small effects this variation is often as large as the effect.
Pairing removes the variation shared by both arms and gives a tighter
comparison than two independent means.

**How to apply.** Decide the seed count before launching. Report per-seed
values in an appendix and the paired mean difference with its standard
error in the main table. Call two arms tied when the paired difference is
within about one standard error.

**Evidence.** A kernel-width effect on one scene was about the size of the
seed spread (plus or minus .02 in TV). A method that won on seed zero had a
three-seed paired difference of +.0020 with standard error .0093 against the
averaging baseline; the three-seed means were .173, .171 and .171 for the
method and two baselines, a tie.

## Seed before model construction and trust the saved config

Set the random seed before building the model, the data pipeline and the
noise draws, and read a run's seed from its saved config, not from its
directory name.

**Why.** Weight initialization happens at construction. A seed set after
construction pairs the data streams but not the starting weights. Directory
names are typed by hand and drift from the config that ran.

**How to apply.** Seed at the start of the entry point, before any module is
created. Write the resolved config, including the seed, into the run
directory and into each checkpoint. Tabulate from the saved config.

**Evidence.** Unseeded model construction made nominally paired runs start
from different weights; every table built on those pairs had to be rerun.
In another campaign a directory named for seed 0 contained a run whose saved
config said seed 42.

## Best-over-run selection distorts rankings; report converged values

Report the value at convergence, defined by a rule stated in advance, not
the best value observed during the run. Selecting the best of many
evaluations picks a favorable noise draw and hides instability, and it
favors some arms more than others.

**Why.** A noisy metric evaluated at many checkpoints has a minimum below
its typical value. Arms that oscillate more, or are evaluated more often,
gain more from this selection, so the ranking can move in either direction.
A run that dips and then collapses looks good under best-over-run.

**How to apply.** Define convergence explicitly, for example "no
improvement in TV of more than .002 over the last 50 evaluation epochs, and
the density images have stopped thinning". Train every arm on the same
horizon and schedule, report the final value and show the trajectory. Keep
best-over-run values, if at all, in a secondary column. Judge stability over
the whole trajectory, as in the
[reproducing-baselines note](reproducing-baselines.md).

**Evidence.** Under best-over-run selection a method appeared only to tie an
averaging baseline. On converged values it beat both baselines on every
seed and width setting (TV .18 against .21 and .23). A two-moons run ended
with SWD .018 and coverage .993 but had sustained regressions over three
separate epoch ranges and was rejected.

## Evaluate EMA weights for results and name the weights used

Report results from the exponential moving average (EMA) of the weights,
state in every table and figure whether live or EMA weights were used, and
check that the swap to EMA weights covers every component.

**Why.** EMA weights are usually better once training has settled, but they
lag the live weights early in training, and with a high decay they can stay
far behind for thousands of steps. A metric computed on half-swapped models
(generator on EMA, decoder or a submodule on live weights) matches neither.

**How to apply.** Show live and EMA samples side by side during training.
Implement the swap as one function that covers all modules with EMA shadows
and restores the live weights in a `finally` block. Restore buffers before
copying EMA parameters. When early-training EMA samples look worse than
live ones, compare the EMA horizon with the step count before suspecting the
method.

**Evidence.** At step 2,000 of a class-conditional ImageNet-32 run, the live
weights produced recognizable objects while EMA weights with decay .999
produced brown and green texture; the EMA metrics at step 2,500 were FID
187, precision .96, recall .002 and class consistency .001. In one pipeline
the FID path did not apply a decoder's EMA shadow while the training grids
did. An EMA swap without `try`/`finally` left EMA weights live after an
exception.

## Rebuild the evaluated model from the checkpoint and the intended code

Construct the model for evaluation from the config stored in the checkpoint,
and check that the evaluation imports the code version that trained it.

**Why.** Loaders that rebuild from a current default config silently drop
fields added later. Import paths set up for one working copy can load the
code from another. Resuming can overwrite the saved config with the resumed
command line.

**How to apply.** Embed the full resolved config in each checkpoint and
build the evaluated model from it. Print the imported module path at the
start of an evaluation. Never let a resume rewrite the original config
without keeping the old one.

**Evidence.** A loader omitted newly added model fields until models were
rebuilt from checkpoint-embedded config. A path setup in a separate working
copy imported the main copy's code. A resume overwrote the saved config of
the run it continued.

## Confirm that the evaluation ran and its outputs are finite

Check the exit status and the output of each evaluation, not only that it
was scheduled. Check that sample grids are not blank and that metrics are
finite before reading them.

**Why.** Evaluation queues fail quietly: a watcher can mark a milestone done
when the job crashed, pruning can delete a checkpoint before it is scored,
and non-finite weights produce valid image files that contain nothing.

**How to apply.** Require an exit status of zero and a result file for a
milestone to count. Copy checkpoints that must be scored to a protected
location before pruning can reach them. Flag image files whose size or
variance is near zero, and stop a run whose loss or weights are not finite.

**Evidence.** A milestone watcher marked FID milestones done without
checking the exit status, and no FID had run. With a keep-last-three
pruning rule, an evaluation queue raced checkpoint deletion until
checkpoints were copied aside first. A run with non-finite weights wrote
solid-black 133-byte grid images for about 80 epochs before anyone noticed.

## Keep one metric row per step across resumes

Deduplicate metric logs by step, keeping the latest row, and rewind the log
to the resume step when a run restarts.

**Why.** An append-only log extended by a resumed run contains two rows for
every step between the checkpoint and the crash. Plots show loops and
averages count those steps twice.

**How to apply.** On resume, truncate the metric log to rows at or before
the resumed step. In analysis, group by step and keep the last row.

**Evidence.** An append-only metrics file duplicated rows after each resume
until the reader kept the latest row per step and the writer rewound on
restart.
