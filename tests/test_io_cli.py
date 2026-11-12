import json

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.io import savemat

from tubular import Mesh
from tubular.analysis import analyze_mesh
from tubular.cli import main
from tubular.io import load_mat_mesh, load_mesh, save_mat_mesh, save_mesh


def test_npz_roundtrip_and_no_overwrite(plane, tmp_path):
    path = tmp_path / "mesh.npz"
    save_mesh(path, plane)
    result = load_mesh(path)
    assert_array_equal(result.vertices, plane.vertices)
    assert_array_equal(result.faces, plane.faces)
    assert_array_equal(result.uv, plane.uv)
    assert result.length_unit == "um"
    with pytest.raises(FileExistsError):
        save_mesh(path, plane)


@pytest.mark.parametrize("variable", [None, "spcutMeshSm"])
def test_matlab_index_conversion(plane, tmp_path, variable):
    path = tmp_path / "mesh.mat"
    data = {"v": plane.vertices, "f": (plane.faces + 1).astype(float), "u": plane.uv}
    savemat(path, {variable: data} if variable else data)
    result = load_mat_mesh(path, variable=variable, length_unit="um")
    assert_array_equal(result.faces, plane.faces)
    assert_array_equal(result.vertices, plane.vertices)
    assert_array_equal(result.uv, plane.uv)
    assert result.length_unit == "um"


def test_single_triangle_mat_roundtrip(tmp_path):
    mesh = Mesh([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], [[0, 1, 2]])
    path = tmp_path / "single.mat"
    save_mat_mesh(path, mesh)
    loaded = load_mat_mesh(path, variable="mesh")
    assert_array_equal(loaded.faces, mesh.faces)
    assert loaded.uv is None
    with pytest.raises(FileExistsError):
        save_mat_mesh(path, mesh)


@pytest.mark.parametrize("bad_faces", [np.array([[0., 1., 2.]]),
                                      np.array([[1., 2.1, 3.]]),
                                      np.array([[1., 2., np.nan]]),
                                      np.array([[1., 2., 99.]])])
def test_invalid_mat_faces(tmp_path, plane, bad_faces):
    path = tmp_path / "bad.mat"
    savemat(path, {"v": plane.vertices, "f": bad_faces})
    with pytest.raises(ValueError, match="one-based"):
        load_mat_mesh(path)


def test_mat_variable_must_be_explicit(plane, tmp_path):
    path = tmp_path / "mesh.mat"
    save_mat_mesh(path, plane)
    with pytest.raises(ValueError, match="fields"):
        load_mat_mesh(path)
    with pytest.raises(ValueError, match="not found"):
        load_mat_mesh(path, variable="wrong")


@pytest.mark.parametrize("metadata", [
    {"schema_version": 99},
    {"schema_version": 1, "index_base": 1, "coordinate_order": "xyz"},
    {"schema_version": 1, "index_base": 0, "coordinate_order": "zyx"},
    {"schema_version": 1, "index_base": 0, "coordinate_order": "xyz"},
    [],
])
def test_snapshot_schema_rejected(tmp_path, plane, metadata):
    path = tmp_path / "invalid.npz"
    np.savez(path, vertices=plane.vertices, faces=plane.faces, metadata=json.dumps(metadata))
    with pytest.raises(ValueError):
        load_mesh(path)


def test_snapshot_disallows_pickle(tmp_path, plane):
    path = tmp_path / "pickle.npz"
    np.savez(path, vertices=plane.vertices, faces=plane.faces, metadata=np.array({}, dtype=object))
    with pytest.raises(ValueError, match="Object arrays"):
        load_mesh(path)


def test_cli_demo_inspect(tmp_path, capsys):
    mesh_file, report = tmp_path / "cylinder.npz", tmp_path / "report.json"
    assert main(["demo", str(mesh_file)]) == 0
    capsys.readouterr()
    assert main(["inspect", str(mesh_file), "--output", str(report)]) == 0
    result = json.loads(report.read_text())
    assert result["topology"]["euler_characteristic"] == 1
    assert result["signed_volume"] is None
    assert result["area"] > 0 and result["length_unit"] == "um"
    with pytest.raises(SystemExit) as exc:
        main(["demo", str(mesh_file)])
    assert exc.value.code == 2


def test_analysis_volume(tetrahedron):
    result = analyze_mesh(tetrahedron)
    assert_allclose(result["signed_volume"], 1)
    assert result["topology"]["closed"]
