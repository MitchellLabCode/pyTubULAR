# pyTubULAR

A Python implementation of [TubULAR](https://github.com/npmitchell/tubular),
developed as an independent Mitchell Lab repository. The numerical core is
designed to work from scripts, notebooks, and eventually a desktop GUI.

**Status: experimental class API (0.9.0a1), not a replacement for the MATLAB pipeline.**
The translated geometry routines have been checked against outputs generated
by the original MATLAB functions. Full surface mapping, motion analysis, and
biological-dataset equivalence remain future milestones. There is no GUI yet.

`from tubular import TubULAR` now provides an experiment object with the original
MATLAB method names. **63 of 200 methods have working, adapted implementations**;
the remaining names raise `MethodNotPortedError`. Method names alone do not imply
equivalent options, file formats, or complete scientific workflows.

## What works

- A `TubULAR` object with time selection, stage-specific mesh caches, image
  channels, APDV transforms, coordinate helpers, and raw area/volume series.
- Landmark-defined APDV frames and mesh-series alignment, with explicit
  reflection orientation and preview/registration behavior. Experiment setup
  uses native Python inputs; ImSAnE compatibility is not a project requirement.
- UV-to-3D surface queries on the supplied triangles, tiled pullback-field
  interpolation, experiment bounds, and existing hoop-centerline loading.
- Centerline length, polygonal Levitt writhe, and Gauss writhe quadrature, with
  intersection checks and explicit MATLAB normalization compatibility.
- Vertex principal/mean/Gaussian curvature, with boundary and winding conventions.
- Beltrami distortion of supplied 3D trajectories and chart-to-chart maps,
  using explicit reference vertex correspondence.
- Annular seam cutting and constrained harmonic UV generation from an explicit
  seam path, with vertex correspondence and foldover/boundary checks.
- Periodic regular UV-grid resampling, optional supplied normals, and reusable
  vertex-field weights; original-triangle and MATLAB-style interpolation modes.
- Explicit endcap removal and shortest edge seams, followed by surface `(s,phi)`
  coordinates with field correspondence ([guide](docs/sphi_parameterization.md)).
- Single-pass PIV with translated PIVlab kernels, validity flags and an audited
  comparison of Python packages ([scope and package audit](docs/piv.md)).
- Validated triangular meshes with explicit coordinates, units, and indexing.
- Manifold/orientation checks, area, signed closed-mesh volume, and face normals.
- Face/vertex averaging, 2D–3D Jacobians, vector projection, induced metrics,
  and first/second fundamental forms.
- Rectilinear mesh connectivity and explicit periodic seam gluing.
- MATLAB v5–v7.2 mesh import/export and versioned, pickle-free NPZ snapshots.
- Trilinear volume sampling and 2D pullbacks **on an already supplied chart**.
- A command-line inspector and reproducible synthetic example.

## Install and run

Use an isolated Python environment. The core supports Python 3.9 or later;
new application/GUI environments should use a currently supported Python release.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
python -m pytest
tubular demo /tmp/tube.npz
tubular inspect /tmp/tube.npz
python examples/synthetic_pullback.py /tmp/pytubular-demo
python examples/class_workflow.py
python examples/surface_queries.py
python examples/centerline_writhe.py
python examples/landmark_alignment.py
python examples/surface_measurements.py
python examples/annular_parameterization.py
python examples/uv_grid_resampling.py
```

On Windows, activate with `.venv\Scripts\activate` and use a local output path.
Outputs must not already exist. The demo writes a mesh, sampled pullback arrays,
and a JSON geometry report; it does not launch a viewer.

## Use the TubULAR object

```python
from tubular import TubULAR
from tubular.grid import cylinder_mesh

tubi = TubULAR(time_points=[0, 2], opts={"timeInterval": 3, "timeUnits": "min"})
mesh = cylinder_mesh(17, 65, radius=6, length=24, length_unit="um")
tubi.register_mesh(0, mesh, kind="spcutMeshSm")
tubi.setTime(0)
current = tubi.getCurrentSPCutMeshSm()
metric, curvature = tubi.measure_current_geometry()
print(tubi.trueTime(2))  # 6 minutes relative to the first timepoint
print(TubULAR.method_status("measureStrainRate")["status"])  # not_ported
```

Existing MAT/NPZ meshes and NumPy volumes can be registered without copying the
original experiment directory. Getters load supplied artifacts; they do not
generate missing cuts or parameterizations. See the
[class API and migration guide](docs/class_api.md),
[complete method coverage](docs/method_coverage.md), and
[runnable two-timepoint example](examples/class_workflow.py).
The [surface query example](examples/surface_queries.py) maps UV points to 3D,
interpolates vertex fields, and measures bounds without creating output files.
The [centerline example](examples/centerline_writhe.py) measures length and
writhe over time. See [writhe conventions and limits](docs/centerline_writhe.md)
before comparing results with an existing MATLAB analysis.
The [alignment example](examples/landmark_alignment.py) defines an anatomical
frame from landmarks and aligns registered meshes. See
[alignment conventions](docs/landmark_alignment.md) for units and reflection.
The [surface measurement example](examples/surface_measurements.py) computes
cylinder curvature and distortion from a known stretch. See
[curvature conventions](docs/surface_curvature.md) and [Beltrami scope](docs/beltrami.md).
The [parameterization example](examples/annular_parameterization.py) creates a
new UV chart from an unparameterized annulus through `generateCurrentCutMesh`.
See [seam conventions and current limits](docs/annular_parameterization.md).
The [UV-grid example](examples/uv_grid_resampling.py) continues through periodic
resampling and field transfer. See [interpolation choices and validation](docs/uv_grid_resampling.md).

## Bring a MATLAB mesh across

```python
from tubular.io import load_mat_mesh, save_mesh
from tubular.geometry import fundamental_forms

mesh = load_mat_mesh("mesh.mat", variable="spcutMeshSm", length_unit="um")
g, b = fundamental_forms(mesh)
save_mesh("mesh.npz", mesh)
```

The selected MATLAB scalar struct must contain `v` (N×3), `f` (F×3, one-based),
and optionally `u` (N×2). Omit `variable` for top-level `v/f/u` arrays.
Faces become zero-based integers exactly once, at import. Physical units must
describe the coordinates already stored; setting a unit does not rescale them.
MATLAB v7.3/HDF5, gridded structs, PLY/OBJ, and complete experiment objects are
not yet supported. No file reader guesses axis order or repairs geometry.

## Design and verification

- [Architecture and GUI design](docs/architecture.md)
- [Translation conventions and source mapping](docs/translation.md)
- [Milestones and acceptance criteria](docs/roadmap.md)
- [Validation evidence and limits](docs/validation.md)
- [Original method inventory](docs/source_inventory.json)
- [Contribution workflow](CONTRIBUTING.md)

The preserved reference data in `tests/data` allow comparison tests to run
without MATLAB. Regenerating those references requires the original TubULAR
checkout and MATLAB; see the validation document.

## Attribution

Original TubULAR: Noah P. Mitchell and Dillon J. Cislo. The source snapshot used
for this translation is `9bd49dbffd38b199c32582b8cc089f611f584679`.
The original MIT notice is retained alongside this project's notice.
Individual mathematical routines retain their source attribution in
[the translation map](docs/translation.md). Bundled MATLAB dependencies have
not been copied wholesale into this repository. The curvature estimator adapts
Itzik Ben Shabat's BSD-licensed MATLAB code; its notice is preserved in
[`licenses/curvature-BSD.txt`](licenses/curvature-BSD.txt) and packaged with releases.

For the original method, see the citation supplied by TubULAR:
Mitchell, N.P. & Cislo, D.J., *TubULAR: Tracking in toto deformations of dynamic
tissues via constrained maps*, bioRxiv 2022.04.19.488840,
[doi:10.1101/2022.04.19.488840](https://doi.org/10.1101/2022.04.19.488840).
