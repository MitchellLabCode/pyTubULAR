"""Create a new UV chart from an unparameterized annulus and a chosen seam."""

import numpy as np

from tubular import TubULAR
from tubular.grid import cylinder_mesh, glue_rectilinear_seam
from tubular.quasiconformal import beltrami_coefficient


def main():
    n_u, n_v = 17, 49
    # The helper only constructs the test geometry. Gluing discards its chart:
    # generateCurrentCutMesh receives an annulus with mesh.uv == None.
    demo = cylinder_mesh(n_u, n_v, radius=2, length=8, length_unit="um")
    annulus, _ = glue_rectilinear_seam(demo, n_u, n_v)
    assert annulus.uv is None
    # The correct conformal width for this polygonal cylinder is its physical
    # length divided by its polygon circumference (not the smooth circumference).
    width = 8/(4*(n_v-1)*np.sin(np.pi/(n_v-1)))
    tubi = TubULAR(time_points=[0])
    tubi.register_mesh(0, annulus, kind="cylinderMeshClean")
    tubi.setTime(0)
    result = tubi.generateCurrentCutMesh(np.arange(n_u), aspect_ratio=width, register=True)
    chart = tubi.getCurrentCutMesh()
    assert chart is result.mesh
    distortion = beltrami_coefficient(chart)
    # Any source per-vertex scalar/vector field follows the same explicit map.
    axial_position = annulus.v[:, 0]
    cut_field = axial_position[result.cut_to_uncut]
    print(f"Cut vertices: {len(chart.v)}; source vertices: {len(annulus.v)}")
    print(f"Seam pairs: {len(result.path_pairs)}; UV orientation: {result.embedding.orientation}")
    print(f"Clamped negative weights: {result.embedding.clamped_edges}")
    print(f"Linear residual: {result.embedding.linear_residual:.3g}")
    print(f"Maximum |Beltrami coefficient|: {np.max(np.abs(distortion.mu)):.3g}")
    print(f"Transferred axial field: [{cut_field.min():g}, {cut_field.max():g}] um")


if __name__ == "__main__":
    main()
