# Geometry and BSDF conventions

Mitsuba gives every surface interaction two normals and evaluates BSDFs with
a built-in cosine, and a path tracer that adds its own estimators on top has
to decide which normal and which cosine each quantity uses. This file records
what Mitsuba's APIs return, the normal convention that keeps area measures
consistent, how to keep surface interactions complete, and how to
differentiate a learned field that is only supervised on surfaces.

## What Mitsuba returns

The facts in this section were checked on Mitsuba 3.8.0.

### Neither normal is flipped toward the ray

`scene.ray_intersect` returns `si.n` following the triangle's winding and
`si.sh_frame.n` following the authored vertex normals, even when the two
point to opposite sides, and neither is flipped toward the incoming ray.
`si.wi` is expressed in the shading frame.

**Evidence.** On a triangle wound toward $+z$ with vertex normals of $-z$,
rays from either side returned `si.n = (0, 0, 1)` and
`si.sh_frame.n = (0, 0, -1)`.

**How to apply.** Compute which side a ray arrives from explicitly,
`dr.dot(si.n, -ray.d) > 0`, and use `dr.abs` for area cosines, which do not
depend on winding.

### `sample_position` returns the interpolated vertex normal

`Mesh.sample_position` returns the interpolated vertex normal in `ps.n` when
the mesh has vertex normals. An emitter cosine or area conversion computed
from `ps.n` is therefore geometric only on flat-shaded meshes. Compute the
face normal separately when a geometric normal is needed on a mesh with
vertex normals.

### `spawn_ray` and `spawn_ray_to` offset along `si.n`

Mitsuba offsets spawned rays along the geometric normal. Hand-written ray
offsets should do the same (with the sign chosen from the side of arrival),
not use a flipped shading normal.

### `BSDF.eval` includes the outgoing cosine

`bsdf.eval(ctx, si, wo)` returns `f_s * abs(wo.z)`, the BSDF times the
projected-solid-angle cosine in the shading frame. Path-tracing integration
wants this product. A quantity that is already a density per unit area (an
irradiance estimate, a photon density) needs `f_s` alone:

```python
def eval_bsdf_value(bsdf, si, wo, active=True, ctx=mi.BSDFContext()):
    """f_s without the projected-solid-angle cosine."""
    cosine = dr.abs(wo.z)
    ok = active & (cosine > 0.0)
    value = bsdf.eval(ctx, si, wo, ok)
    return dr.select(ok, value / dr.select(ok, cosine, 1.0),
                     mi.Spectrum(0.0))
```

**Evidence.** Two versions of the same pipeline, one contracting densities
with `bsdf.eval` and one with `f_s`, disagreed by the mean of $1/\cos$ over
the receivers: a total-weight ratio of 1.47 and a flux ratio of 1.40. Each
version was self-consistent; comparing a result from one against a reference
from the other looked like a large error.

### The diffuse BSDF gates on the shading frame and ignores the transport mode

Mitsuba's `diffuse` returns `albedo / pi * cos(wo)` and returns zero when
`si.wi.z <= 0` or `wo.z <= 0`, both measured in the shading frame. It applies
no adjoint correction in `mi.TransportMode.Importance`. `twosided` chooses
which side's BSDF to evaluate from the sign of `si.wi.z`, again in the
shading frame.

**Consequence.** On a one-sided BSDF with interpolated normals, a point whose
face is front-facing by `si.n` can be black because the interpolated normal
faces away from the viewer. The image then has a discontinuity inside a
face, where the interpolated normal crosses the view direction, which
estimators that assume a smooth interior miss.

### Choose the transport mode from the direction of the walk

Camera walks use `mi.TransportMode.Radiance`; walks that start at a light and
carry flux use `mi.TransportMode.Importance`, set in the `mi.BSDFContext`,
at every vertex including dielectric crossings. The mode does not change the
sampled directions or their probabilities. It changes the weight's units: at
a dielectric transmission, Mitsuba multiplies the radiance-mode weight by
$1/\eta^2$, where $\eta$ is the relative index of refraction of the
crossing, and leaves the importance-mode weight unscaled. Entering glass of
index 1.5 at normal incidence, the sampled transmission weight is 0.444 in
radiance mode and 1 in importance mode (Mitsuba 3.8).

### Test lobes with the BSDF flags

`mi.has_flag(bsdf.flags(), mi.BSDFFlags.Diffuse)` (and the `Glossy` and
`Delta` flags) works on a per-lane BSDF pointer array, so vertices can be
classified inside a vectorized walk. A BSDF that combines a diffuse and a
glossy lobe (a rough plastic, for example) falls into both classes; reject it
explicitly if the code treats the two classes differently.

### Updating vertex positions recomputes vertex normals from the winding

For a mesh with vertex normals, writing `vertex_positions` and calling
`params.update()` recomputes the vertex normals from the face winding,
replacing any normals loaded from the file. A file whose normals point
against its winding flips its normals at the first update.

Mitsuba's change detection does not register a write of the parameter's own
buffer (`params[key] = mi.Float(params[key])`). To force the recomputation,
write a different buffer, for example a fresh `dr.zeros` buffer filled by
gather and scatter.

**How to apply.** At initialization, write every mesh whose normals matter
through a fresh buffer, so all meshes are in the recomputed state before the
first render, not only those a later update happens to touch.

**Evidence.** A glass slab loaded from files whose normals opposed their
winding had its front face updated at initialization and its back face left
alone, so one face carried recomputed normals and the other the file's. The
dielectric treated both interfaces as exits, and the slab transmitted 4.67
instead of 0.92, a factor of $1.5^4$ from the index of refraction.

### Moving a mesh rebuilds its area distribution

`Mesh.surface_area()` and `Mesh.sample_position()` read an area distribution
that Mitsuba rebuilds whenever `vertex_positions` changes, with a parallel
scan whose last bits vary between rebuilds. See
[randomness and independence](randomness-and-independence.md).

## Normals and measures

### The geometric normal owns measures; the shading normal owns BSDF evaluation

Use the geometric normal `si.n` for every area measure and density, and the
shading normal `si.sh_frame.n` only inside BSDF evaluation.

The geometric normal belongs to:

- solid-angle to area conversions and the densities built on them;
- connection cosines to a sensor or emitter that stand for an area measure;
- surface sample anchors and their densities;
- labels for which side of a surface a point or ray is on;
- ray offsets and edge orientation.

The shading normal enters only BSDF evaluation and one ratio: a flux that
arrives per unit geometric area becomes the shading model's irradiance when
multiplied by the ratio of the shading to the geometric incidence cosine:

```python
def incidence_ratio(n_shading, n_geometric, toward_source):
    d = dr.normalize(toward_source)
    shading = dr.maximum(dr.dot(n_shading, d), 0.0)
    geometric = dr.abs(dr.dot(n_geometric, d))
    return dr.select(geometric > 1e-6, shading / geometric, 0.0)
```

The ratio is one on flat-shaded meshes, which is why mixing the normals goes
unnoticed until a mesh with authored vertex normals appears. Test estimators
on such a mesh.

**Evidence.** On a mesh with authored vertex normals, using the shading
normal for a sensor-connection cosine produced a 25% pointwise error, and
applying the missing ratio to a light-side weight moved a shadow-boundary
derivative from 0.0249 to 0.0334.

### Importance transport with shading normals needs the adjoint factor

When a light-started walk connects to the sensor from a scattering vertex
with interpolated normals, the connection weight is
`abs(dot(n_g, wo)) * f_s * cos_sh(wi) / cos_geo(wi)`, where `wo` points to the
sensor and `wi` is the arrival direction (Veach's adjoint BSDF for shading
normals). Mitsuba's `bsdf.eval(ctx, si, wo)` returns `f_s * cos_sh(wo)`,
which equals it only when the shading and geometric normals coincide.

### Apply the incidence ratio where the arrival direction is known

A photon deposit carries flux per unit geometric area. Its conversion to
shading irradiance needs the arrival direction, so multiply by the incidence
ratio at deposit time. A field that stores position-only irradiance cannot
recover the factor at query time, because the directions were marginalized
when the deposits were pooled.

## Surface interactions

### Carry the complete `SurfaceInteraction3f`

A `mi.SurfaceInteraction3f` constructed from a position and a normal has no
shape pointer, primitive index, texture coordinates, shading frame or
incident direction. Code that reads it does not fail: `si.shape` is null, so
support tests and `si.bsdf()` lookups mask the lane out, and a zero normal
falls back to a default orientation, which gives plausible values from the
wrong side.

**How to apply.** Store and move the whole interaction produced by the walk.
DrJit gathers, selects and scatters structures directly:

```python
si_live = dr.gather(mi.SurfaceInteraction3f, si, index)
si_first = dr.select(take, si, si_first)
```

**Evidence.** A walk record that rebuilt its first interaction from position
and normal alone lost the shape pointers of all 7,513 valid endpoints in a
batch. Every one failed the support test, so a training stream received no
samples, with no error raised.

### Never re-trace to recover a vertex the walk already found

Keep the endpoint a sampled walk produced; do not trace the same camera ray
again to find it. At a dielectric the second trace samples reflection or
refraction independently and can land on a different surface, pairing one
path's value with another path's point. Fitting to such pairs converges to a
spatially warped solution with ghost caustics.

### Offset ray origins, not query points

When a visibility test needs clearance from a surface or an edge, move the
ray's origin or aim point. Evaluate fields at the recorded surface point or at
a re-intersected hit, never at the offset point, which lies off the surface
the field was fitted on.

## Derivatives on surfaces

### A field supervised on surfaces has no meaningful normal derivative

A neural field fitted only at points on surfaces is unsupervised along the
surface normal: its value there is hash-grid extrapolation, and the normal
component of its input gradient is arbitrary. Contract the field's input
gradient only with the tangential part of a velocity:

```python
n = dr.normalize(mi.Vector3f(si.n))           # geometric normal
v_tangent = v - n * dr.dot(v, n)
rate = dr.dot(grad_x, v_tangent)
```

For a point on a moving object, hold its material coordinates fixed: the
world velocity of a hit has a normal component, and the part of the motion
that changes the field's input is the tangential slide of the hit across
the surface.

### Finite differences that move the surface read the same extrapolation

A finite difference that re-traces the scene with the object displaced by
$\pm h$ and evaluates the field at the new hits reads the field off the
surface it was fitted on, by $h$ times the normal component of the motion. It
then agrees with the unprojected AD contraction, and neither is the physical
derivative. Validate against a finite difference of a rendered or path-traced
quantity, or evaluate the field at material points on the current surface.
A finite difference of the field along world axes compared with `dr.grad`
validates the AD implementation, not the physics.
