# pyTubULAR

A Python implementation of [TubULAR](https://github.com/npmitchell/tubular),
developed as an independent Mitchell Lab repository. The numerical core is
designed to work from scripts, notebooks, and eventually a desktop GUI.

**Status: experimental foundation, not a replacement for the MATLAB pipeline.**
The translated geometry routines have been checked against outputs generated
by the original MATLAB functions. Full surface mapping, motion analysis, and
biological-dataset equivalence remain future milestones. There is no GUI yet.

## What works

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
```

On Windows, activate with `.venv\Scripts\activate` and use a local output path.
Outputs must not already exist. The demo writes a mesh, sampled pullback arrays,
and a JSON geometry report; it does not launch a viewer.

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
for this translation is `9bd49dbffd38b199c32552b8cc089f611f584679`.
The original MIT notice is retained alongside this project's notice.
Individual mathematical routines retain their source attribution in
[the translation map](docs/translation.md). Bundled MATLAB dependencies have
not been copied into this repository.

For the original method, see the citation supplied by TubULAR:
Mitchell, N.P. & Cislo, D.J., *TubULAR: Tracking in toto deformations of dynamic
tissues via constrained maps*, bioRxiv 2022.04.19.488840,
[doi:10.1101/2022.04.19.488840](https://doi.org/10.1101/2022.04.19.488840).
