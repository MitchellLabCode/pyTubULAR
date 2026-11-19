"""Closed triangular surface -> explicit caps/seam -> surface (s,phi)."""

import numpy as np

from tubular import Mesh, TubULAR
from tubular.grid import cylinder_mesh, glue_rectilinear_seam


def main():
    # Synthetic segmented surface, in micrometers. Real data enter as Mesh
    # vertices/faces, or via tubular.io.load_npz/load_mat; no voxel extraction.
    nu,nv = 9,25
    cut = cylinder_mesh(nu,nv,radius=2,length=8,length_unit="um")
    tube,_ = glue_rectilinear_seam(cut,nu,nv)
    a,p = len(tube.v),len(tube.v)+1
    faces = list(tube.f)
    for j in range(nv-1):
        nxt = (j+1)%(nv-1)
        faces.extend([[a,j*nu,nxt*nu],[p,nxt*nu+nu-1,j*nu+nu-1]])
    segmented = Mesh(np.vstack((tube.v,[0,0,0],[8,0,0])),np.asarray(faces),length_unit="um")
    tubi = TubULAR(time_points=[0],opts={"nU":17,"nV":49})
    tubi.register_mesh(0,segmented,kind="rawMesh"); tubi.setTime(0)
    opened = tubi.sliceMeshEndcaps([a],[p],register=True)
    # These boundary vertices explicitly fix axial direction and reference seam.
    chart = tubi.generateCurrentCutMesh(seam_endpoints=[0,nu-1],register=True)
    result = tubi.generateCurrentSPCutMesh(chart.path_pairs,register=True)
    # Transfer a field from the original segmentation through all three maps.
    source_field = segmented.v[:,0]
    cut_field = source_field[opened.vertex_sources][chart.cut_to_uncut]
    transferred = result.interpolate(cut_field)
    assert np.array_equal(result.mesh.v[:result.n_u],result.mesh.v[-result.n_u:])
    print(f"Grid: {result.n_u} x {result.n_v}, including duplicate seam")
    print(f"s: {result.sphi[:,0].min():.6g} to {result.sphi[:,0].max():.6g} um")
    print("phi: 0 to 1 turn; result.phi_radians provides radians")
    print(f"Field transfer error: {np.max(abs(transferred-result.mesh.v[:,0])):.3g}")


if __name__ == "__main__":
    main()
