# Precision and bitwise parity

Most performance work on a DrJit program promises unchanged numbers, and
checking that promise needs a clear idea of where bits come from. This file
covers the three device behaviors that move bits without any change to the
mathematics (`FastMath`, fused multiply-add contraction, and hardware texture
filtering), then how to build comparisons that tell a real change from these
effects.

## Device arithmetic

### `FastMath` is on by default and approximates divide and square root

DrJit enables `dr.JitFlag.FastMath` by default, which lowers `x / y` and
`dr.sqrt` to the hardware approximations. On the CUDA backend this costs about
1e-6 relative accuracy, in double precision as well as single.

**Why it matters.** A host/device comparison written at a tolerance near
double-precision epsilon measures the approximation, not the arithmetic under
test.

**How to apply.** Turn the flag off for the scope of an exact comparison and
restore it afterwards, so later kernels keep production arithmetic:

```python
with dr.scoped_set_flag(dr.JitFlag.FastMath, False):
    device = kernel_under_test(mi.Float64(x))
np.testing.assert_allclose(device.numpy(), reference(x), rtol=1e-12)
```

In pytest, wrap the same save-and-restore in a fixture used by those tests
only. Do not turn it off globally to make a test pass.

**Evidence.** A double-precision parity test failed at a maximum absolute
error of 8.7e-7 with the flag on and passed at 2.2e-16 with it off.

### Evaluation boundaries change rounding

DrJit's CUDA backend contracts a multiply followed by an add into one fused
multiply-add inside a kernel. Evaluating `a * b` in one kernel and adding `c`
in another rounds twice instead of once, so moving an evaluation point
changes bits.

**Why.** Within one kernel the compiler is free to fuse `a * b + c`; across
an evaluation boundary the product is stored in memory, rounded, and
reloaded. The fused result matches FMA rounding exactly and the split result
matches unfused rounding exactly; neither is wrong.

**Consequences.** All of these are numerical changes, however small:

- adding or moving a `dr.eval` before a query;
- merging two queries into one call when the merge materializes inputs that
  were lazy;
- reordering reads so an input is materialized earlier;
- diagnostic instrumentation that evaluates intermediate records to
  attribute time (this also changes kernel fusion and the kernel count).

The same holds outside DrJit: the same inline CUDA function compiled into two
different kernels can round differently, so a "reference" kernel written for
a test is not a bit-level reference for production.

**Evidence.** Splitting `a * b + c` across an evaluation changed the result on
about 23% of random lanes. A merge of two network queries moved 1.5 to 2.4% of
lanes by at most 2.4e-5 of the peak value. Evaluating intermediate records for
a timing breakdown changed a loop's stable kernel count from 311 to 287.

### Hardware texture filtering limits finite differences of bitmaps

In the CUDA variants, a bitmap texture's lookup runs through the GPU texture
unit, which interpolates with 8-bit fractional weights, while the reverse-mode
derivative with respect to texture coordinates is the exact bilinear slope.
A central difference of the lookup therefore has a relative error of about
`(1/256 texel) / step`.

**How to apply.** When checking a texture derivative against a finite
difference:

- step at least 0.4 texel;
- keep only lanes whose three sample points fall in the same texel cell;
- treat a residual of about 1% as the hardware weights, not a pairing bug.

**Evidence.** On a 360 by 234 texture, a step of 1e-6 in texture
coordinates gave pure noise (median relative error 1.0), 0.05 texel about 8%,
and 0.4 texel about 1% (median 0.3%, maximum 3.7% on same-cell lanes).

### Accumulate long sums in double precision when they must repeat

Single-precision running sums formed by parallel scans do not repeat between
constructions and lose resolution on long tables. Form such sums on the host
in double precision, or in double precision on the device normalized by
their own endpoint, and validate the result on the value ranges in use. See
[randomness and independence](randomness-and-independence.md)
for the `mi.DiscreteDistribution` case.

## Building comparisons

### Compare in process, per lane, from production's own state

A parity probe must feed the reference path the same lazy or evaluated state
that production feeds the candidate. A probe that evaluates captured inputs
before calling the reference changes the reference's rounding and can report
a bitwise match for a change that moves production bits.

**How to apply.** Run both paths in one process on the same inputs and seed,
compare per lane, and count differing lanes. When lanes differ, wrap the
reference's queries so they evaluate their inputs, their outputs, or both at
the query boundary. If the difference drops to zero lanes under one of these
wrappings, it is fusion, not a bug; if it does not, look for a real change.

**Evidence.** A probe that evaluated its inputs first reported a merge as
bitwise while the production merge moved about 1.5% of lanes. Once the
reference also evaluated the bounced hit's position, normal and direction
before its network read, the difference was zero lanes.

### Restructured reductions differ at reassociation level

Any change that reorders a sum (merging batches, replaying a recorded
computation, contracting a Jacobian in a different order) changes results at
the level of floating-point reassociation, and a bitwise gate cannot accept
it. Define the tolerance from the spread of an unchanged repeat instead.

**Evidence.** A replayed gradient matched the raw computation to 1e-7 to
3e-6 relative, against a raw-versus-raw spread of 1e-7 to 2e-6; its largest
lane difference was 3.5e-5 of the peak. Multiplying the network weights by
`1 + 1e-6 * N(0, 1)` moved the same gradient by 4.6e-5 relative, a useful
scale for how much a rounding-level difference matters downstream.

### One-update comparisons cannot see rounding changes

A comparison of outputs after a single optimizer update is blind to
last-bit changes in the gradients, because Adam's first step is close to
`lr * sign(g)`. Comparisons after many updates differ even between two runs
of the same code, because of atomic accumulations in training (see
[randomness and independence](randomness-and-independence.md)).

**How to apply.** Gate a change that promises identical numbers by dumping
the raw per-lane records it produces (samples, weights, queries) from the old
and new code and comparing their bytes.

**Evidence.** A restructured loop around an unchanged per-lane function
changed 3 to 36% of its records by one or a few ulps through changed FMA
contraction, while every image after zero or one update was byte-identical.
A record dump caught it.
