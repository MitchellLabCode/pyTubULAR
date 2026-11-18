"""Define an anatomical frame and align a mesh series without an image viewer."""

import numpy as np

from tubular import TubULAR
from tubular.grid import cylinder_mesh


def run():
    tubi = TubULAR(time_points=[0, 3], opts={"timeInterval": 2, "spaceUnits": "um"})
    # Geometry already uses physical XYZ coordinates; resolution stays at one.
    origin = np.array([12., -4., 7.])
    q = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    for time, radius in ((0, 2), (3, 2.5)):
        mesh = cylinder_mesh(9, 33, radius=radius, length=10, length_unit="um")
        tubi.register_mesh(time, mesh.with_vertices(mesh.v @ q.T+origin))
    tubi.computeAPDVCoords(origin, origin+q[:, 0]*10, origin+q[:, 2]*3)
    preview = tubi.alignMeshesAPDV()
    print("Aligned timepoints:", preview["timepoints"])
    print("Physical times:", preview["times"], preview["time_unit"])
    print("Rotation determinant:", np.linalg.det(preview["rot"]))
    result = tubi.alignMeshesAPDV(register=True)
    tubi.setTime(3)
    print("Aligned vertices:", len(tubi.getCurrentAlignedMesh().v))
    print("Face corner correspondence:", result["face_corner_order"])


if __name__ == "__main__":
    run()
