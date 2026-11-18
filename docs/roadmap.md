# Translation milestones and acceptance criteria

This is the full-program design. Milestone 1 and the class-interface slice
below are implemented; full real-data interchange remains incomplete. Each stage
must have a runnable non-GUI API before a GUI exposes it. There is no reliable
schedule estimate until representative real data and its expected MATLAB
outputs are available.

| Stage | Work | Gate before proceeding |
| --- | --- | --- |
| 1. Numerical foundation | Mesh contracts, geometry, charts, explicit MATLAB import, synthetic pullback | Original-MATLAB comparisons plus analytic/invariance/input-validation tests; implemented in this snapshot |
| 2. Real-data interchange | PLY/OBJ mesh adapters, TIFF/HDF5 volume adapters, project manifest, APDV transforms | Same landmarks and overlays in MATLAB/Python; round-trip IDs, units, timestamps, winding, anisotropic voxel tests |
| 3. Tubular topology | Centerline extraction, cap selection/removal, seam cutting, correspondence maps | Expected boundary loops and Euler characteristics; no lost fields/vertices; manually inspected real-data cuts |
| 4. Parameterization | Orbifold-Tutte or documented replacement; UV and s-phi construction | Original map comparison modulo explicit gauge alignment, boundary constraints, seam continuity, Jacobian/area distortion reports, no foldovers |
| 5. Dynamic pullbacks | Surface sampling stacks, texture/geometric phase stabilization, temporal smoothing | Image/texture agreement with identical projection settings; known translations/rotations; no seam jumps or undocumented time interpolation |
| 6. Velocities/pathlines | PIV or optical-flow adapter, 2D-to-3D lifting, pathline integration, temporal averaging | Synthetic known motion, convergence with time step, missing-frame/nonuniform-time tests, tracking over periodic seams |
| 7. DEC and mechanics | DEC operators, Helmholtz-Hodge, curvature kinematics, strain and strain rate | `d*d=0`, adjoint/sign tests, Poisson nullspaces and boundary conditions, rigid-motion zero strain, analytic dilation/twist/taper cases, original outputs |
| 8. Biological reproduction | A representative midgut/heart dataset and a second geometry | Agreed pixel/field/statistical tolerances; reproduce selected published plots and document discrepancies |
| 9. GUI and distribution | Napari project/inspection prototype can start after stage 2; complete workflow after scientific stages | Linked 2D/3D selections, cancellation, save/reopen, offline operation, packaging tests on supported systems, researcher usability sessions |

## Next concrete slice

The `TubULAR` object now supplies time selection, mesh and volume sources,
APDV transforms, coordinate helpers, and an explicit method-coverage catalog.
Its 58 adapted MATLAB methods are documented in [class_api.md](class_api.md).
The next five after the initial class milestone add UV surface queries,
rectilinear pullback-field interpolation, experiment bounds, and existing hoop
centerline loading. Their outputs have been compared to an actual original
MATLAB `TubULAR` instance on synthetic inputs.
The 0.4 milestone adds centerline length and Levitt/Gauss writhe series, including
independent quadrature checks, original-MATLAB kernel comparisons, and explicit
rejection of intersecting curves. Polar writhe and the original smoothing
pipeline are still pending; see [centerline_writhe.md](centerline_writhe.md).
The 0.5 milestone adds explicit landmark-frame estimation and batch mesh
alignment. Its native constructor no longer simulates an ImSAnE experiment.
Anatomical landmark detection and endpoint adjustment remain separate future
work; see [landmark_alignment.md](landmark_alignment.md).
The 0.6 milestone adds vertex curvature and Beltrami measurements on supplied
corresponding vertices/charts, with original-kernel and analytic comparisons.
These do not implement tracking, new surface parameterization, or full strain.
The other 142 method names are explicit pending ports, including topology,
parameterization, velocity, strain, and plotting workflows.

Import one existing MATLAB `spcutMeshSm` and its matching raw image volume.
Keep the existing mapping fixed. Compare surface overlays, sampled pullback,
metric, and normal conventions at that timepoint. This isolates interoperability
and sampling issues before changing the surface parameterization algorithm.
Then extend to two timepoints with explicitly established correspondence.

## Dependency decisions still open

Do not translate bundled third-party MATLAB libraries indiscriminately.
Inventory their calls and licenses, choose a Python/native replacement or a
small validated port, and record the decision. In particular:

- GPToolbox/CGAL remeshing and geometric queries need explicit replacement and
  mesh-quality benchmarks; a generic nearest-neighbor function is not equivalent.
- Orbifold-Tutte and Ricci-flow methods cannot be substituted by a planar PCA
  projection or generic UV unwrap without revisiting their mathematical guarantees.
- DECLab operators require matched primal/dual conventions and boundary treatment.
- PIVLab behavior is not identical to a generic optical-flow implementation;
  retain a comparison dataset and identify interpolation/filtering differences.
- TexturePatch projection and microscopy axis conventions need their own
  reference cases, including mirrored and anisotropic acquisitions.

The source method inventory is a discovery aid. A method remaining in that
inventory is not a Python feature unless listed as implemented in the README.
