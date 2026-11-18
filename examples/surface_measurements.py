"""Curvature and chart distortion on synthetic surfaces, with no output files."""

import numpy as np

from tubular import Mesh, TubULAR
from tubular.grid import cylinder_mesh, glue_rectilinear_seam


def run():
    cut = cylinder_mesh(11, 49, radius=2, length=8, length_unit="um")
    mesh, correspondence = glue_rectilinear_seam(cut, 11, 49)
    tubi = TubULAR(time_points=[0,3], opts={"timeInterval": 2},
                    mesh_sources={"spcutMeshSmRSC": {0:mesh}})
    result = tubi.measureCurvatures(time_points=[0])["curvatures"][0]
    interior = (mesh.v[:,0] > 1.5) & (mesh.v[:,0] < 6.5)
    print("Cylinder mean curvature (1/um):", np.median(result.mean[interior]))
    print("Cylinder Gaussian curvature (1/um^2):", np.median(result.gaussian[interior]))
    print("Boundary vertices:", result.boundary_vertices.sum())
    uv = np.array([[0.,0], [1.,0], [1.,1.], [0.,1.]])
    reference = Mesh(np.column_stack((uv, np.zeros(4))), np.array([[0,1,2],[0,2,3]]), uv, "um")
    positions = {0:reference.v, 3:reference.v*[2,1,1]}
    distortion = tubi.measureBeltramiCoefficient(reference, positions)
    print("Physical times:", distortion["times"], distortion["time_unit"])
    print("Spatial Beltrami coefficients:", distortion["mu"])
    print("Dilatation:", distortion["dilatation"])
    charts = {time:xyz[:,:2] for time,xyz in positions.items()}
    planar = tubi.measureBeltramiCoefficientPullbackToPullback(reference, charts)
    print("Planar/spatial agreement:", np.allclose(planar["mu"], distortion["mu"]))


if __name__ == "__main__":
    run()
