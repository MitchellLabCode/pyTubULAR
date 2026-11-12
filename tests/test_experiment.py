from pathlib import Path
import json

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.io import savemat

from tubular import Mesh, MeshSource, MethodNotPortedError, TubULAR
from tubular.experiment import MESH_KINDS
from tubular.io import save_mesh


def test_constructor_legacy_metadata_and_no_writes(tmp_path):
    output = tmp_path / "absent"
    xp = {"fileMeta": {"timePoints": [2, 4, 9], "stackResolution": [0.5]*3}, "expMeta": {}}
    tubi = TubULAR(xp, {"meshDir": output, "nU": 12, "nV": 25, "timeInterval": 3, "t0": 4})
    assert tubi.currentTime is None and tubi.dynamic
    assert_array_equal(tubi.xp.fileMeta.timePoints, [2, 4, 9])
    assert tubi.xp.tIdx[9] == 2  # Python positions, not MATLAB one-based positions.
    assert tubi.uvexten == "_nU0012_nV0025"
    assert tubi.APDV.resolution == 0.5
    assert not output.exists()
    xp["fileMeta"]["timePoints"][0] = 100
    assert tubi.timePoints[0] == 2
    assert_allclose(tubi.trueTime([2, 4, 9]), [-6, 0, 15])
    assert tubi.trueTime(9, True, return_label=True) == (0.25, "0.25 hr")
    assert tubi.t0set() == 4


def test_reference_time_file_and_explicit_override(tmp_path):
    (tmp_path / "t0_referenceTime.txt").write_text("# reference time\n7\n")
    tubi = TubULAR(opts={"meshDir": tmp_path}, time_points=[5, 7, 11])
    assert tubi.t0set() == 7
    tubi.t0set(9)
    assert tubi.t0set() == 9  # No surprising reload/reset from disk.
    assert TubULAR(opts={"meshDir": tmp_path, "t0": 8}).t0 == 8


@pytest.mark.parametrize("times", [[], [2, 1], [1, 1], [0, np.nan], [True], [[1, 2]]])
def test_invalid_timepoints(times):
    with pytest.raises(ValueError):
        TubULAR(time_points=times)


def test_time_switch_clears_all_caches_and_preserves_same_time(plane):
    tubi = TubULAR(time_points=[0, 2])
    tubi.setTime(0)
    for kind in MESH_KINDS:
        tubi.register_mesh(0, plane, kind=kind)
        assert tubi.get_mesh(kind) is plane
    tubi.register_data(0, np.ones((2, 3, 4)))
    data = tubi.getCurrentData()
    tubi.currentVelocity.piv3d["dummy"] = 1
    tubi.currentSegmentation.seg3d["dummy"] = 1
    tubi.currentCline.mss = 1
    tubi.setTime(0)
    assert tubi.getCurrentData() is data and tubi.currentMesh.ricciMesh is plane
    with pytest.raises(ValueError, match="not in"):
        tubi.setTime(1)
    assert tubi.currentTime == 0 and tubi.currentMesh.alignedMesh is plane
    tubi.currentTime = 2
    assert all(getattr(tubi.currentMesh, name) is None for name in MESH_KINDS)
    assert tubi.currentData.IV is None and not tubi.currentVelocity.piv3d
    assert not tubi.currentSegmentation.seg3d and tubi.currentCline.mss is None
    tubi.clearTime()
    with pytest.raises(ValueError, match="setTime"):
        tubi.getCurrentRawMesh()


def test_instances_do_not_share_state(plane):
    first, second = TubULAR(), TubULAR()
    first.setTime(0)
    first.register_mesh(0, plane)
    first.getCurrentRawMesh()
    first.setEndcapOptions({"a": [2]})
    assert second.currentTime is None and second.endcapOptions == {}
    assert second.currentMesh.rawMesh is None


def test_apdv_position_and_vector_transforms():
    r = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    for flip in (False, True):
        tubi = TubULAR(opts={"rot": r, "trans": [3., -4., 5.], "resolution": 0.25, "flipy": flip})
        x = np.array([[1., 2., 3.], [-3., 1., 8.]])
        original = x.copy()
        expected = np.array([[0.25, -0.75, 2.], [0.5, -1.75, 3.25]])
        if flip:
            expected[:, 1] *= -1
        assert_allclose(tubi.xyz2APDV(x), expected)
        assert_allclose(tubi.APDV2xyz(expected), x)
        dx = np.array([[0.3, -0.7, 0.1], [1., 0., 0.]])
        assert_allclose(tubi.dx2APDV(dx), tubi.xyz2APDV(x + dx) - tubi.xyz2APDV(x), atol=1e-15)
        assert_allclose(tubi.APDV2dxyz(tubi.dx2APDV(dx)), dx, atol=1e-15)
        assert_array_equal(x, original)
        returned, _ = tubi.getRotTrans()
        returned[:] = 0
        assert_array_equal(tubi.getRotTrans()[0], r)


def test_apdv_requires_explicit_valid_frame(tmp_path):
    tubi = TubULAR()
    with pytest.raises(ValueError, match="unset"):
        tubi.xyz2APDV([[1, 2, 3]])
    with pytest.raises(ValueError, match="orthonormal"):
        tubi.set_apdv_frame(np.diag([1, 2, 3]), [0, 0, 0])
    rot_path, trans_path = tmp_path / "r.txt", tmp_path / "t.txt"
    np.savetxt(rot_path, np.eye(3))
    np.savetxt(trans_path, [1, 2, 3])
    tubi.fileName.rot, tubi.fileName.trans = rot_path, trans_path
    assert_allclose(tubi.xyz2APDV([[0, 0, 0]]), [[1, 2, 3]])
    xp = {"fileMeta": {"stackResolution": [1, 1, 2]}}
    with pytest.raises(ValueError, match="anisotropic"):
        TubULAR(xp)
    assert TubULAR(xp, {"resolution": 1}).APDV.resolution == 1


def test_named_mat_mesh_and_legacy_path_cache(tmp_path, plane):
    filename = tmp_path / "mesh_0004.mat"
    data = {"v": plane.v, "f": plane.f + 1, "u": plane.u}
    savemat(filename, {"spcutMeshSm": data})
    tubi = TubULAR(time_points=[4], opts={"fullFileBase": {"spcutMeshSm": str(tmp_path / "mesh_%04d.mat")}})
    tubi.setTime(4)
    first = tubi.getCurrentSPCutMeshSm()
    assert_array_equal(first.f, plane.f)
    data["v"] = 2 * plane.v
    savemat(filename, {"spcutMeshSm": data})
    assert tubi.getCurrentSPCutMeshSm() is first
    assert_allclose(tubi.loadCurrentSPCutMeshSm().v, 2 * plane.v)
    tubi.register_mesh(4, MeshSource(filename, "spcutMeshSm", "um"), kind="spcutMeshSm")
    assert tubi.getCurrentSPCutMeshSm().length_unit == "um"


def test_stage_getters_and_register_invalidation(plane, tmp_path):
    path = tmp_path / "mesh.npz"
    save_mesh(path, plane)
    tubi = TubULAR(mesh_sources={"alignedMesh": {0: path}, "ricciMesh": {0: plane}})
    tubi.setTime(0)
    aligned = tubi.getCurrentAlignedMesh()
    assert_allclose(aligned.v, plane.v)
    assert tubi.getCurrentAlignedMesh() is aligned
    assert tubi.getCurrentRicciMesh() is plane
    moved = plane.with_vertices(plane.v + [2, 1, 0])
    tubi.register_mesh(0, moved, kind="alignedMesh")
    assert tubi.getCurrentAlignedMesh() is moved
    assert tubi.currentMesh.ricciMesh is plane
    tubi.register_mesh(0, MeshSource(path, length_unit="mm"), kind="alignedMesh")
    with pytest.raises(ValueError, match="units disagree"):
        tubi.getCurrentAlignedMesh()


def test_native_file_pattern_does_not_select_matlab_variable(plane, tmp_path):
    save_mesh(tmp_path / "chart_0004.npz", plane)
    tubi = TubULAR(time_points=[4], opts={
        "fullFileBase": {"spcutMeshSm": str(tmp_path / "chart_{time:04d}.npz")}})
    tubi.setTime(4)
    assert_allclose(tubi.getCurrentSPCutMeshSm().v, plane.v)
    assert tubi.getCurrentSPCutMeshSm().length_unit == plane.length_unit


def test_numpy_data_channels_and_mip(tmp_path):
    volume = np.arange(60).reshape(3, 4, 5)
    path = tmp_path / "volume.npy"
    np.save(path, volume * 2)
    tubi = TubULAR(data_sources={0: (volume, path)})
    original = volume.copy()
    volume[:] = 999
    tubi.setTime(0)
    channels = tubi.getCurrentData(False)
    assert_array_equal(channels[0], original)
    assert_array_equal(channels[1], original * 2)
    assert not channels[0].flags.writeable
    assert_array_equal(tubi.mip(0, 1, [1, 3], channel=1), (original * 2)[[0, 2]].max(axis=0))
    assert_array_equal(tubi.mip(0, 3), original.max(axis=2))
    np.save(path, original * 3)
    assert_array_equal(tubi.loadCurrentData()[1], original * 3)
    tubi.data.adjusthigh = 99
    with pytest.raises(NotImplementedError, match="intensity"):
        tubi.getCurrentData()
    assert_array_equal(tubi.getCurrentData(False)[0], original)
    with pytest.raises(ValueError, match="one-based"):
        tubi.mip(0, 1, [0])


def test_series_measurement_preserves_current_time(tetrahedron):
    twice = tetrahedron.with_vertices(2 * tetrahedron.v)
    tubi = TubULAR(time_points=[1, 4], opts={"timeInterval": 2},
                   mesh_sources={"alignedMesh": {1: tetrahedron, 4: twice}})
    tubi.setTime(1)
    cached = tubi.getCurrentAlignedMesh()
    result = tubi.measureSurfaceAreaVolume()
    assert_allclose(result["volume"], [1, 8])
    assert_allclose(result["area"][1] / result["area"][0], 4)
    assert_allclose(result["times"], [0, 6])
    assert result["length_unit"] == "um"
    assert tubi.currentTime == 1 and tubi.getCurrentAlignedMesh() is cached
    with pytest.raises(NotImplementedError, match="preview"):
        tubi.measureSurfaceAreaVolume({"preview": True})


def test_measurement_rejects_open_and_mixed_units(plane):
    tubi = TubULAR(mesh_sources={"alignedMesh": {0: plane}})
    with pytest.raises(ValueError, match="closed"):
        tubi.measureSurfaceAreaVolume()
    result = tubi.measureSurfaceAreaVolume({"allow_open": True})
    assert np.isnan(result["volume"]).all() and (result["area"] > 0).all()
    other = Mesh(plane.v, plane.f, plane.u, "mm")
    tubi = TubULAR(time_points=[0, 1], mesh_sources={"alignedMesh": {0: plane, 1: other}})
    with pytest.raises(ValueError, match="units differ"):
        tubi.measureSurfaceAreaVolume({"allow_open": True})


def test_class_pullback_and_forms(plane):
    z, y, x = np.indices((4, 5, 5))
    tubi = TubULAR(mesh_sources={"spcutMeshSm": {0: plane}}, data_sources={0: 100*z + 10*y + x})
    tubi.setTime(0)
    g, b = tubi.measure_current_geometry()
    assert_allclose(g, [np.array([[5., 2.], [2., 10.]])] * 2)
    assert_allclose(b, 0, atol=1e-14)
    u, v = np.linspace(0, 1, 4), np.linspace(0, 1, 7)
    result = tubi.sample_current_pullback(u, v)
    uu, vv = np.meshgrid(u, v)
    assert_allclose(result.image, 102*uu + 31*vv, atol=1e-12)
    assert result.valid.all()


def test_complete_method_catalog_and_explicit_pending_behavior(tmp_path):
    original = json.loads((Path(__file__).parent / "data/matlab_class_methods.json").read_text())
    expected = {m["name"] for m in original} - {"TubULAR", "empty"}
    catalog = TubULAR.method_status()
    assert set(catalog) == expected
    tubi = TubULAR(opts={"meshDir": tmp_path})
    tubi.setTime(0)
    mesh_cache = tubi.currentMesh
    for name, entry in catalog.items():
        assert callable(getattr(tubi, name)) and name in dir(tubi)
        if entry["status"] == "not_ported":
            with pytest.raises(MethodNotPortedError, match=name):
                getattr(tubi, name)()
    assert list(tmp_path.iterdir()) == []
    assert tubi.currentTime == 0 and tubi.currentMesh is mesh_cache
    with pytest.raises(MethodNotPortedError, match="doubleResolution"):
        TubULAR.doubleResolution(None)
    catalog["setTime"]["status"] = "tampered"
    assert TubULAR.method_status("setTime")["status"] == "adapted"
    assert "measureStrainRate" not in TubULAR.available_methods()


def test_unsupported_options_are_not_silently_ignored():
    with pytest.raises(NotImplementedError, match="useBioformats"):
        TubULAR(opts={"useBioformats": True})
    with pytest.raises(ValueError, match="both"):
        TubULAR({"fileMeta": {"timePoints": [0]}}, time_points=[0])
    with pytest.raises(ValueError):
        TubULAR(opts={"timeInterval": -1})
