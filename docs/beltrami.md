# Beltrami measurements on supplied maps (0.6)

Two class methods now expose the numerical measurement stage without the
original tracking, Ricci-flow, filtering, and plotting workflows:

```python
spatial = tubi.measureBeltramiCoefficient(reference_mesh, {
    0: vertices_at_time_0,   # N × 3, ordered by reference vertex ID
    3: vertices_at_time_3,
})
planar = tubi.measureBeltramiCoefficientPullbackToPullback(reference_mesh, {
    0: chart_at_time_0,      # N × 2, ordered by reference vertex ID
    3: chart_at_time_3,
})
```

Both methods accept keyword-only `time_points`, defaulting to all acquired
times. The second argument must be an explicit timepoint-to-array mapping.
The reference is a `Mesh` with a UV chart. All target rows must correspond to
the **same reference vertex IDs**; calculations use the reference triangles.
Correspondence is a caller-supplied contract. The code can verify sizes and
geometry, but it cannot prove that those rows track the same material points.
Equal vertex counts or equal grid resolutions are not evidence of tracking.

The result dictionary contains:

| Key | Meaning |
| --- | --- |
| `mu` | Complex T × F raw Beltrami coefficients |
| `dilatation` | T × F ratios of largest/smallest Jacobian singular values |
| `reference_mu` | F coefficients of reference UV → reference XYZ for the spatial method; reference UV → itself for the planar method |
| `timepoints`, `times`, `time_unit` | Acquisition coordinates and physical elapsed times |
| `target_dimension` | 3 for spatial maps, 2 for chart maps |

Selection and caches are preserved; no files are written. Each coefficient
belongs to a face in the supplied ordering. The methods do not create charts,
track vertices, smooth fields, average face values onto vertices, or calculate
time derivatives. Only a fixed supplied chart and corresponding target points
are required; no ImSAnE metadata is involved.

## Meaning and formulas

The pure function is
`quasiconformal.beltrami_coefficient(reference, mapped_coordinates=None)`.
It returns a `BeltramiResult` with read-only `mu`, `dilatation`, and
`target_dimension`. Omitting coordinates measures the reference's own embedding.

On each triangle the map is affine. For a planar map `f=x+iy`,
`mu = f_zbar / f_z`. For a spatial map, let `E`, `F`, `G` be its induced metric;
then `mu = (E-G+2iF)/(E+G+2*sqrt(EG-F²))`. The implementation evaluates the area
term using a cross product instead of subtracting nearly equal squared terms.
It normalizes each Jacobian before taking products and uses singular values
for the dilatation ratio.

Both fields are dimensionless and unchanged by uniform target scale.
`mu=0`, `dilatation=1` indicates local angle preservation. An X stretch of two
relative to Y gives `mu=1/3`, `dilatation=2`. These quantities do **not** measure
isotropic growth, area change, or a full strain tensor. The phase of mu depends
on the reference chart axes.

The original spatial class method subtracts a scalar real-mean coefficient for
certain reference-chart choices. Python returns the **raw coefficient** and
`reference_mu` separately; it does not silently subtract a reference offset.
If the reference chart is nonconformal, raw mu includes that chart distortion
even when the surface has not moved. Subtracting coefficients is not generally
the Beltrami coefficient of a composed deformation. Computing a conformal
reference or a true relative material deformation is separate work.

## Orientation, validity, and limits

Planar maps must have positive determinant relative to the reference UV on
every triangle. Reflected or folded triangles are rejected. This is a deliberate
improvement over the original pullback-to-pullback code, which embeds the target
chart in Z=0 and uses the 3D metric, losing its orientation information.
For orientation-preserving planar maps, the two formulas agree.

Spatial metric coefficients are invariant under a mirror of the embedding;
they cannot diagnose a 3D reflection or surface self-intersection. Neither
routine proves global injectivity or detects distant triangles overlapping.
The reference must be an oriented manifold with nondegenerate UV triangles;
target triangles must be nondegenerate. Maps with singular-value ratios at or
above 1e12, nonfinite coordinates, or numerically unresolved mu are rejected.

## Validation and source scope

The kernels adapt `utility/geometry/metric/bc_metric.m`. The class methods adapt
only the measurement portion of `measureBeltramiCoefficient.m` and
`measureBeltramiCoefficientPullbackToPullback.m`. Loading generated pathlines,
Ricci parameterization, and mode filtering remain pending.

`scripts/generate_beltrami_reference.m` runs the unchanged original `bc_metric`
on nonuniform planar and curved spatial maps. It also compares a planar map
embedded in Z=0, matching the original pullback-to-pullback formula. Source,
generator, and fixture hashes are in `tests/data/beltrami_provenance.json`.
Comparisons use `atol=1e-12, rtol=1e-11`. Independent tests verify analytic
stretch/shear, chart-rotation phase, target similarity invariance, reflection
behavior, face ordering, invalid maps, and unchanged experiment state.

```matlab
addpath('/path/to/pyTubULAR/scripts');
generate_beltrami_reference('/path/to/tubular', ...
    '/path/to/pyTubULAR/tests/data/matlab_beltrami_reference.mat');
```
