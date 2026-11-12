"""Expected values produced by the original MATLAB implementation."""

from pathlib import Path
import hashlib
import json
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.io import loadmat

from tubular import Mesh
from tubular.geometry import (averaging_operators, face_normals, fundamental_forms,
                              induced_metric, jacobian_2d_to_3d, jacobian_3d_to_2d,
                              signed_volume, surface_area)
from tubular.grid import rectilinear_faces


def test_reference_provenance():
    root = Path(__file__).resolve().parents[1]
    data = root / "tests/data"
    provenance = json.loads((data / "provenance.json").read_text())
    assert hashlib.sha256((data / "matlab_reference.mat").read_bytes()).hexdigest() == provenance["fixture_sha256"]
    assert hashlib.sha256((root / "scripts/generate_matlab_reference.m").read_bytes()).hexdigest() == provenance["generator_sha256"]


def test_matlab_reference():
    # Missing references fail; never quietly skip the comparison.
    r = loadmat(Path(__file__).parent / "data" / "matlab_reference.mat")
    mesh = Mesh(r["v"], r["f"].astype(np.int64) - 1, r["u"])
    assert_array_equal(rectilinear_faces(int(r["nU"].item()), int(r["nV"].item())), mesh.faces)
    g, b = fundamental_forms(mesh)
    v2f, f2v = averaging_operators(mesh)
    outputs = {"J23": jacobian_2d_to_3d(mesh), "J32": jacobian_3d_to_2d(mesh),
               "metric": induced_metric(mesh), "g": g, "b": b,
               "V2F": v2f.toarray(), "F2V": f2v.toarray(),
               "normals": face_normals(mesh),
               "legacyNormals": face_normals(mesh, convention="legacy_face_normals"),
               "area": surface_area(mesh)}
    for name, actual in outputs.items():
        assert_allclose(actual, r[name].squeeze() if name == "area" else r[name],
                        rtol=1e-11, atol=1e-12, err_msg=name)
    tetrahedron = Mesh(r["tetV"], r["tetF"].astype(np.int64) - 1)
    assert_allclose(signed_volume(tetrahedron), r["volume"].item(), rtol=1e-12)
    assert_allclose(surface_area(tetrahedron), r["tetArea"].item(), rtol=1e-12)
