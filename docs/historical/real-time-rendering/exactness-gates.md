# Gating speed changes on exact numerics

A speed change to a training kernel is accepted only when it leaves the
training numerics unchanged, or changes them by no more than two runs of the
unchanged code differ from each other. Training an online cache makes this
hard to check: a hash-grid backward pass with float atomics is not
reproducible run to run, and the optimizer hides small input changes. These
notes record which comparisons actually detect a changed kernel and which do
not.

## A one-update film cannot prove two sample streams equal

**Insight.** Comparing the image rendered after one optimizer update does not
detect ulp-level changes in the training samples.

**Mechanism.** On Adam's first step the bias-corrected update is
$m/\sqrt{v} = g/|g|$, a sign per parameter. Inputs that differ by an ulp give
the same signs, the same parameters and a byte-identical film. Longer fits do
not help either, because atomic-order differences make two runs of the same
code diverge after a few hundred updates.

**How to apply.** Compare the sampled training records themselves (or the
gradients they produce) across the two code versions, byte for byte.

**Evidence.** A persistent-warp loop around an unchanged per-lane path
sampler changed fused multiply-add contraction in the inlined code and moved
3 to 36% of gather records by ulps. Every 0- and 1-update film stayed
byte-identical; 200-update films differed even between two runs of the old
code.

## Dump raw records from both source trees

**Insight.** The reliable gate for a change to the training trace is a raw
dump of its records from the old and new source trees, compared as bytes.

**Mechanism.** The trace is deterministic given the scene, the seed and the
update index, so equal code gives equal bytes. The dump isolates the trace
from the nondeterministic fit that follows it.

**How to apply.** Build a small dump tool against each header set with the
production compile flags (copy them from the build directory's generated flag
files so both builds match the viewer). Run both on every scene package in one
GPU job and compare several sequences of a full batch (four sequences of
131,072 records per package worked). Treat any differing byte as a numerical
change until explained.

## An in-test reference kernel is not the production reference

**Insight.** A copy of a device function compiled into a test kernel does not
reproduce the bits of the same function inside the production kernel.

**Mechanism.** Inlining exposes different neighbouring operations to the
compiler, which contracts multiplies and adds into fused multiply-adds
differently. The source is identical; the arithmetic is not.

**How to apply.** Compare against the production binary built from the old
tree, not against a reference written inside the new test.

**Evidence.** An in-test per-lane copy of the path sampler disagreed with the
production kernel by one ulp on 3,125 photon records of one scene.

## Moving where a value is materialized moves its bits

**Insight.** Evaluating an intermediate at a different point, or joining two
passes into one, is a numerical change even when the formula is unchanged.

**Mechanism.** GPU compilers fuse `a*b + c` into one FMA when both operations
sit in one kernel. Writing `a*b` to memory and adding `c` in another kernel
rounds twice. About a quarter of random lanes differ between the two forms.

**How to apply.** Treat every added or moved evaluation, lane compaction or
kernel fusion as non-bitwise until a per-lane comparison against production
state shows zero differing lanes. Build the comparison so the reference sees
the same lazy or materialized inputs as production does.

## Make the reduction deterministic where the platform allows it

**Insight.** Summing hash-grid gradients in fixed point makes a training update
bitwise reproducible, and turns every later parity check into an exact one.

**Mechanism.** Float atomics add in an unspecified order, so each run rounds
differently. Integer addition is associative: accumulate each gradient as a
64-bit fixed-point value, scaled per level from the batch's largest
$|\partial L/\partial y|$, and round to float once in the optimizer kernel.

**How to apply.** Where only 32-bit integer atomics exist (as in WebGPU), add
the low word and then the carry into the high word. Write NaN to every
parameter of a level whose gradient bound is non-finite, so the failure stays
visible. With deterministic updates, gate speed changes by bitwise equality of
the whole fit.

**Evidence.** In the WebGPU backend, fixed-point grid sums reproduced the
reference three-seed fits and made repeated updates bitwise equal. The native
backend's float atomics differ from it by 1e-7 to 1e-6 relative L2 in the
grid moments, which is the atomic-order floor.

## Use the run-to-run spread as the tolerance when bits cannot match

**Insight.** When a change alters only the order of float atomics, accept it if
its fits sit inside the spread between repeated runs of the unchanged code.

**Mechanism.** The unchanged code already has an atomic-order floor. A change
that only reorders additions (a float2 atomic in place of two scalar ones, a
second stream) cannot be bitwise, but its effect is of the same kind as that
floor when every other operation is unchanged.

**How to apply.** Run several seeds of old and new code on each scene and
compare the image error to the reference. Require the traces (by record dump)
and the MLP (by plane-by-plane comparison) to be bitwise equal, so the only
remaining difference is the atomic order. Measure the floor as the relative
L2 of the saved state after a fixed number of updates (64 worked), with the
fraction of bitwise-equal entries.

**Evidence.** Floors after 64 updates ranged from 2e-8 to 8e-7 relative L2 for
an fp32 field, 1.2e-5 to 1.7e-5 for an fp16 field, and 1e-3 to 7e-3 for an
fp16 field trained at a 4× batch; 54 to 82% of entries were bitwise equal.

## Compare SASS when a change should not touch device code

**Insight.** When a change only reorganizes host scheduling, identical machine
code for every kernel proves the arithmetic did not move.

**How to apply.** Dump SASS for every kernel from both builds
(`cuobjdump -sass`), normalize the compiler's anonymous-namespace names, and
compare function by function.

**Evidence.** Overlapping the next update's trace with the current fit left
447 of 447 fp16 functions and 193 of 193 fp32 functions identical.

## Test atomic kernels where their result is order-independent

**Insight.** A float atomic kernel can still be tested bitwise, on inputs where
the order cannot matter.

**Mechanism.** Float addition of two terms is commutative, so an entry that
receives at most two addends has the same value in any order.

**How to apply.** Build a sparse-gradient batch in which every table entry
receives at most two contributions and compare the kernel bitwise against the
library's (768 nonzero entries in 3D and 1,528 in a packed 4D grid were
enough).

## Gate in tiers when the full fit cannot be bitwise

**Insight.** When half-precision or float atomics make whole fits differ run
to run, stack gates from exact to statistical.

**How to apply.**

1. Zero-update renders from the same checkpoint must be byte-identical.
2. One-update record counts, gradient statistics and MLP parameters must be
   identical.
3. Multi-update fits must stay inside the measured repeat envelope of the
   unchanged code.

**Evidence.** Two runs of one fp16 binary changed about 215,000 of 31.2M
values after one update, and a rebatching change moved a fit by .0258 against
a repeat envelope of .0250. The atomic-order floor comes in discrete modes: a
7.8e-5 gap between two schedules was matched by two runs of one binary
(8.0e-5).

## When summation order must change, compare the field itself

**Insight.** A change that deliberately reorders a reduction needs a
statistical gate, and the gate should compare the learned field, not only the
image.

**Mechanism.** A bright, unchanged camera-path prefix can dominate an image
and hide a change in the field underneath it.

**How to apply.** From one checkpoint, run two fits of the old code and one of
the new, compare at several points (1,000, 4,000 and 10,000 updates), and
evaluate the field on identical retained query points at two viewing angles.
Accept a difference within about twice the repeat variation. Bound later
differences by a multiple of the identical-repeat spread; do not treat
growing divergence as a bug by itself (a grid difference of 2.7e-8 at update 1
grew to 7.2e-3 by update 100 between identical repeats).

## Keep the old path as a permanent reference

**Insight.** When replacing a library kernel, keep the library path callable
and make every new option default to the old behaviour bit for bit.

**How to apply.** Keep the library's fit and inference entry points in the
build and gate each replacement against them. An option that is absent must
run the old code on host and device, verified against a verbatim copy, and is
stamped into checkpoints only when set. Test that a diagnostic leaves weights,
optimizer moments, EMA and random counters unchanged, and that a batched call
(`train_steps(3)` then `train_steps(5)`) gives the same records as eight
single steps. Poison lanes that a compaction should skip with NaN, so a lane
read by mistake shows up.

**Evidence.** Three gate failures during the speed work were test bugs, not
kernel bugs: an oracle that still expected a gradient-clear pass the new code
correctly removed, a test that read records after a render that uses the
record buffer as scratch, and a zero-update film that needed a checkpoint.

## Separate exact speedups from recipe changes

**Insight.** Label every change as either behaviour-preserving or a new recipe,
and never let a precision, batch or schedule change pass as a speedup.

**Mechanism.** fp16 instead of fp32, a larger batch, a different network
backend or a different random generator can each be faster and each changes
the trajectory. Calling them speedups hides a quality change.

**How to apply.** Stamp precision, backend, batch sizes, schedule, random
stream version and representation in every checkpoint and report, and reject a
checkpoint whose stamps do not match the executing binary. Validate a recipe
change against the reference on quality; validate a speedup by the gates
above.
