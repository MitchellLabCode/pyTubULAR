"""Annulus -> cut/solve -> periodic grid, including a reusable field map."""

import numpy as np

from tubular import TubULAR
from tubular.geometry import surface_area
from tubular.grid import cylinder_mesh, glue_rectilinear_seam


def main():
    n_u, n_v = 17, 49
    demo = cylinder_mesh(n_u,n_v,radius=2,length=8,length_unit="um")
    source,_ = glue_rectilinear_seam(demo,n_u,n_v)
    tubi = TubULAR(time_points=[0],opts={"nU":25,"nV":65})
    tubi.register_mesh(0,source,kind="cylinderMeshClean")
    tubi.setTime(0)
    # A diagonal source edge path, creating a seam that can curve in UV.
    cut = tubi.generateCurrentCutMesh(np.arange(n_u)*(n_u+1),register=True)
    # Exact periodic unit normals supplied explicitly for this synthetic tube.
    source_normals = np.column_stack((np.zeros(len(source.v)),source.v[:,1:]/2))
    cut_normals = source_normals[cut.cut_to_uncut]
    grid = tubi.generateCurrentUVCutMesh(cut.path_pairs,normals=cut_normals,register=True)
    assert tubi.getCurrentUVCutMesh() is grid.mesh
    axial_field = grid.interpolate(source.v[cut.cut_to_uncut,0])
    assert np.array_equal(grid.mesh.v[:grid.n_u],grid.mesh.v[-grid.n_u:])
    glued,_ = glue_rectilinear_seam(grid.mesh,grid.n_u,grid.n_v,tolerance=0)
    print(f"Regular UV grid: {grid.n_u} x {grid.n_v} (including duplicate seam)")
    print(f"Source faces: {len(source.f)}; grid faces: {len(grid.mesh.f)}")
    print(f"Source area: {surface_area(source):.6g}; grid approximation: {surface_area(glued):.6g} um^2")
    print(f"Transferred axial field: {axial_field.min():.6g} to {axial_field.max():.6g} um")
    print("Vertex weights are retained in grid.source_vertices and grid.barycentric.")


if __name__ == "__main__":
    main()
