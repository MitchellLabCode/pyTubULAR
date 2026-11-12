# Contributing to pyTubULAR

Translate one scientific operation at a time. Identify the original source and
its exact revision, document indexing/orientation/units, export expected values
by running the original routine, and compare the Python output. Add analytic
tests that can catch errors shared by both implementations.

For a deliberate mathematical change, document why the MATLAB output should
not be reproduced, test the new invariant, and keep the old/reference behavior
available when it is meaningful. Do not loosen tolerances just to make a failing
comparison pass. A missing reference fixture must fail, not skip.

Use pure functions for numerical calculations. Keep file writes, viewers, and
global experiment state out of the core. Distinguish face, vertex, chart-pixel,
and volume-voxel fields in every API. Return a correspondence map whenever
topology changes. Do not silently repair meshes, infer physical units, reverse
normal signs, or reuse coordinate-frame labels after transforms.

Before submitting changes:

```sh
python -m pip install -e '.[test]'
python -m pytest
python -m build
```

Scientific milestones additionally need real-data evidence as described in
`docs/roadmap.md`. GUI tests will be separate from the headless core tests.
New dependencies require a purpose and license review; importing a viewer
must never be necessary to import `tubular`.
