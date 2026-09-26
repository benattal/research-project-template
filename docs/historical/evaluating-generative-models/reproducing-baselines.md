# Reproducing baselines and running fair comparisons

A comparison between a new method and a baseline says something only when
the baseline is working as well as it can, both arms run under the same
protocol and budget, and the numbers are read at convergence over the whole
trajectory. This file covers reproducing a published result before building
on it, tuning baselines and ceilings, matching budgets, judging training
trajectories, and handling runs that were trained under known bugs.

## Reproduce the published result, component by component

Before comparing against a published method, reproduce it: check each
component numerically against the official implementation, then reproduce
its evaluation protocol, and only then its headline number.

**Why.** A reimplementation that differs in one component (a resize filter,
a normalization, a feature network) is a different method, and its failure
says nothing about the original. Component parity tests find those
differences in minutes; a full training run finds them, if at all, days
later and without saying where.

**How to apply.** For each component (loss, feature extractor, generator
architecture), feed identical inputs and weights to both implementations
and compare outputs to a stated tolerance. Check parameter counts against
the paper. Reproduce the evaluation exactly: sample count, the source of
class labels, the Inception weights and resize, and the reference
statistics file. Keep a numbered list of deliberate deviations, each with
its reason, and close entries as they are tested.

**Evidence.** Reproducing a published one-step class-conditional
ImageNet-256 generator, the loss matched the official implementation to a
maximum relative error of about $5 \times 10^{-6}$, the feature encoder to
about $3 \times 10^{-5}$, and both model sizes matched the paper's parameter
counts exactly. A second feature branch disagreed in every layer group until
a missing antialiasing step in an image resize was fixed; afterwards all
groups matched within $3.4 \times 10^{-5}$. The FID protocol was 50,000
samples with labels drawn from the validation loader, the benchmark's
Inception network with bilinear resize to 299 pixels, and the benchmark's
reference statistics. Deviations such as 8-bit storage of feature banks
and a normalizer computed over smaller groups were listed with reasons.

## Tune every baseline and the clean ceiling

Give each baseline and each ceiling the same tuning effort as the proposed
method, and make the strongest simple baseline mandatory. A method succeeds
only if it beats that baseline quantitatively and visually under a matched
protocol; approaching the clean ceiling is not enough.

**Why.** Simple baselines are often strong. Averaging several noisy
observations and training on the average, for example, removes much of the
noise at no cost in method complexity. An untuned baseline flatters the
method, and an untuned ceiling misstates the gap left to close.

**How to apply.** Sweep the baseline's main parameters (bandwidth, learning
rate, horizon) with the same seeds and budget as the method. Tune the clean
ceiling the same way; its value is an upper bound only after tuning. Report
the selected settings of every arm.

**Evidence.** On a noisy spiral, a tuned averaging-of-three-captures
baseline reached SWD .0098 while the proposed estimator reached .015 to .019
over three seeds. On an easier setting, three-seed paired TV was .173 for
the method against .171 for the averaging baseline and .171 for a baseline
that used a single capture, a tie.

## Match optimizer steps as well as data passes

When arms differ in batch size, state whether they are matched in epochs or
in optimizer steps, and run the step-matched comparison before drawing a
conclusion about batch size or sample efficiency.

**Why.** Doubling the batch at a fixed epoch count halves the optimizer
steps. The larger-batch arm then has less optimization, and a difference in
the result mixes the effect of batch noise with the effect of fewer steps.

**How to apply.** Hold the step count, learning rate and schedule fixed and
change only the batch. Report steps and epochs for every arm. When a method
change also changes the per-step cost, match wall time as a separate,
labeled comparison.

**Evidence.** A batch-size test at a fixed epoch count gave the larger batch
25,600 optimizer steps against 51,200 for the smaller. It was rerun with
both arms at 51,200 steps; there the larger batch reached TV .450 against
.445, confirming a probe that had found the error dominated by bias.

## Compare at convergence, not early

Read comparisons at convergence under a stated rule, after every arm has
finished the same schedule. An early lead does not predict the final order.

**Why.** Biased estimators often converge fast to a worse point, while
estimators with less bias and more variance improve more slowly and keep
improving. An early checkpoint rewards the first kind.

**How to apply.** Train every arm on the same full schedule. Plot the
metric against epoch for all arms together and report the final values.
Treat early readings as a check that runs are healthy, not as results.

**Evidence.** On 1D mixtures, an averaging baseline led at epoch 60 (SWD
.098 against .124), was behind by epoch 120 (.105 against .097), and
plateaued near .107 while the proposed method kept improving to .090. On a
2D spiral, a baseline variant that looked better at 100 epochs (SWD .0119)
converged to .0166 at 400 epochs, behind the variant it was compared with
(.0098).

## Judge the whole trajectory with a stability gate

Accept a run only if it reaches a good state and stays there. Define "good"
and "regressed" numerically and reject runs with sustained regressions, even
if their last checkpoint is good.

**Why.** A method whose samples collapse and recover during training is
unreliable: the recovery may not happen on another seed, dataset or
horizon. Selecting only the last or best checkpoint hides this.

**How to apply.** A workable gate: the first good checkpoint is the first
with SWD at or below a threshold and coverage at or above a threshold.
After it, a checkpoint is bad when SWD exceeds
$\max(1.5\,b, b + .01)$, where $b$ is the best SWD so far, when energy
distance exceeds the analogous bound with $+.002$, or when coverage falls
more than .1 below its best. Two or more consecutive bad checkpoints form a
sustained regression. Report the first good epoch and every regression
interval.

**Evidence.** A two-moons run finished with SWD .018 and coverage .993 but
had sustained regressions over three separate epoch ranges and was
rejected. The rule for the campaign was stated in its plan: do not hide a
collapse by selecting a later checkpoint.

## Reproduce a failure under current code before blaming a change

When a failure is suspected to be caused by a bug that has since been
fixed, rerun the failing configuration on the current code. When a failure
appears after a code change, bisect over commits with an identical config
and compare internal quantities at step zero.

**Why.** A fix and a failure can coincide in time without being related.
Rerunning separates "the bug caused it" from "the method does it". Step-zero
quantities with the same seed isolate what a code change altered before
training dynamics amplify it.

**How to apply.** Keep the exact config of failing runs so that they can be
rerun. For a bisect, run each commit with the same config and seed and log
the loss terms and kernel statistics at the first step. Predict the size of
the change from the diff before looking.

**Evidence.** A collapse observed in an archived image run was rerun after
a batch of bug fixes and reproduced faster (two draws per condition differed
by 1/255 by epoch 10), which ruled out the fixed bugs as its cause. In a
separate regression, a bisect with identical configs found that one side of
a kernel computation matched to seven decimals at step zero while the other
shrank from .649 to .540, a distance inflation of 1.43, close to the
$\sqrt{2}$ predicted by a normalizer that changed from .5 to .25.

## Retire runs trained under known bugs

When a bug is fixed, tag every run trained under it with the reason and
remove those runs from comparisons. Keep them for reference but never mix
them into tables with post-fix runs.

**Why.** Runs from before a fix look like valid data points, and a table
that mixes them compares different code. Tags on the run itself survive
longer than notes in a log.

**How to apply.** Add a status suffix or metadata field naming the bug to
each affected run directory. Rerun the arms a table needs.

**Evidence.** Cohorts were retired for a misalignment between images and
conditioning attributes, for a resume bug that broke random-state
determinism and made the loss drift upward across resumed segments, and for
reference runs made with the wrong noise protocol.

## Run the new pipeline in the regime where the reference works

When a new pipeline fails in the target regime, run it in the regime where
an established reference succeeds. If it fails there too, the problem is the
pipeline, not the regime.

**Why.** A hard regime has its own information limit. Without the easier
control, a pipeline bug and a physical limit look the same.

**How to apply.** Keep one configuration in which a reference is known to
work, and run every new pipeline there first. Compare epoch by epoch.

**Evidence.** A new reconstruction pipeline produced blurry results at a
low photon count. Rerun at the higher photon count where an earlier pipeline
was sharp, it was blurry there too, which pointed at the pipeline: an
encoder that saw too few pixels per view. After the fix it tracked the
earlier pipeline epoch by epoch at the higher count (29.3 against 29.8 dB by
epoch six), and the remaining softness at the low count was the measurement
limit.
