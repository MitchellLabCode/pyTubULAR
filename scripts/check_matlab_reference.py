"""Print numerical error measurements against the committed MATLAB fixture."""

import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.io import loadmat

from tubular import Mesh, __version__
from tubular.geometry import (averaging_operators, face_normals, fundamental_forms,
                              induced_metric, jacobian_2d_to_3d, jacobian_3d_to_2d,
                              signed_volume, surface_area)


def report():
    fixture = Path(__file__).resolve().parents[1] / "tests/data/matlab_reference.mat"
    r = loadmat(fixture)
    mesh = Mesh(r["v"], r["f"].astype(np.int64) - 1, r["u"])
    g, b = fundamental_forms(mesh)
    v2f, f2v = averaging_operators(mesh)
    tet = Mesh(r["tetV"], r["tetF"].astype(np.int64) - 1)
    outputs = {"J23": jacobian_2d_to_3d(mesh), "J32": jacobian_3d_to_2d(mesh),
               "metric": induced_metric(mesh), "g": g, "b": b,
               "V2F": v2f.toarray(), "F2V": f2v.toarray(),
               "normals": face_normals(mesh),
               "legacyNormals": face_normals(mesh, convention="legacy_face_normals"),
               "area": surface_area(mesh), "volume": signed_volume(tet),
               "tetArea": surface_area(tet)}
    errors = {}
    for name, actual in outputs.items():
        expected = r[name].item() if np.ndim(actual) == 0 else r[name]
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12, err_msg=name)
        errors[name] = float(np.max(np.abs(actual - expected)))
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "pytubular": __version__,
            "matlab": str(r["matlabVersion"][0]), "rtol": 1e-11, "atol": 1e-12,
            "maximum_absolute_errors": errors}


if __name__ == "__main__":
    print(json.dumps(report(), indent=2))
