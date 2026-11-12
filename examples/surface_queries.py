"""Query an existing surface chart and bounds through TubULAR; no file writes."""

import numpy as np

from tubular import TubULAR
from tubular.grid import cylinder_mesh
from tubular.surface import locate_chart


def run():
    mesh = cylinder_mesh(9, 33, radius=2, length=6, length_unit="um")
    tubi = TubULAR(time_points=[0], opts={"rot": np.eye(3), "trans": [0, 0, 0],
                                         "resolution": 1, "normalShift": 0.5},
                   mesh_sources={"rawMesh": {0: mesh}, "spcutMeshSmRS": {0: mesh}})
    tubi.setTime(0)
    # U is normalized by max(U)=6 by uv2APDV; V stays in radians in this chart.
    queries = np.array([[0.25, np.pi/4], [0.5, np.pi/2], [1.2, np.pi]])
    xyz, faces, chart, weights = tubi.uv2APDV(queries, return_details=True)
    print("Mapped XYZ (outside points remain NaN):\n", xyz)
    print("Zero-based face IDs (-1 outside):", faces)
    field = 10 + mesh.v[:, 0] - 2*mesh.v[:, 1]
    print("Interpolated vertex field:", locate_chart(chart, queries).interpolate(field))
    raw, pixels, physical, buffered = tubi.getXYZLims()
    print("Physical XYZ bounds:\n", physical)
    print("Buffered XYZ bounds:\n", buffered)
    print("Implemented MATLAB method names:", len(tubi.available_methods()))


if __name__ == "__main__":
    run()
