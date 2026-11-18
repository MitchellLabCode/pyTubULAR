import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.io import loadmat

from tubular import Mesh, TubULAR
from tubular.alignment import align_mesh, landmark_frame
from tubular.geometry import face_normals, signed_volume, surface_area
from tubular.io import save_mesh
from tubular.mesh import topology


A = np.array([2., -3., 4.])
P = np.array([8., 1., 7.])
D = np.array([4., 2., 10.])


@pytest.mark.parametrize("ap,dv", [([1, 0, 0], [0, 0, 1]),
                                   ([-1, 0, 0], [0, 0, 1]),
                                   ([1, 0, 0], [0, 0, -1]),
                                   ([0, 0, 1], [1, 2, 0]),
                                   ([0, -1, 0], [3, 1, 2]),
                                   ([6, 4, 3], [2, 5, 6])])
def test_landmark_frame_axes_and_handedness(ap, dv):
    origin = np.array([2., -4., 1.])
    ap, dv = np.array(ap), np.array(dv)
    r, t = landmark_frame(origin, origin+ap, origin+dv)
    mapped = np.array([origin, origin+ap, origin+dv]) @ r.T+t
    assert_allclose(mapped[0], 0, atol=2e-15)
    assert_allclose(mapped[1], [np.linalg.norm(ap), 0, 0], atol=2e-15)
    assert abs(mapped[2, 1]) < 2e-15 and mapped[2, 2] > 0
    assert_allclose(r @ r.T, np.eye(3), atol=5e-16)
    assert_allclose(np.linalg.det(r), 1, atol=5e-16)


def test_frame_covariance_scale_and_explicit_origin():
    r, t = landmark_frame(A, P, D)
    q, _ = np.linalg.qr(np.array([[1., 2., 3.], [-2., 1., 1.], [1., 3., -1.]]))
    shift = np.array([7., 2., -6.])
    for scale in (1e-9, 1, 1e9):
        moved = np.array([A, P, D]) @ q.T*scale + shift*scale
        rr, tt = landmark_frame(*moved)
        assert_allclose(rr @ q, r, atol=1e-15)
        assert_allclose((moved @ rr.T + tt)/scale, np.array([A, P, D]) @ r.T+t, atol=6e-15)
    explicit = A+[0.1, 0.2, -0.4]
    actual = landmark_frame(A, P, D, origin=explicit)
    expected = landmark_frame(explicit, P, D)
    assert_allclose(actual[0], expected[0])
    assert_allclose(actual[1], expected[1])


@pytest.mark.parametrize("a,p,d", [([0, 0], P, D), (A, [np.nan, 1, 2], D),
                                  (A, A, D), (A, P, A), (A, P, A+2*(P-A)),
                                  ([0, 0, 0], [1, 0, 0], [2, 1e-13, 0]),
                                  (A.astype(complex), P, D)])
def test_invalid_landmarks(a, p, d):
    with pytest.raises(ValueError):
        landmark_frame(a, p, d)


@pytest.mark.parametrize("flip", [False, True])
@pytest.mark.parametrize("preserve", [False, True])
def test_mesh_alignment_orientation_units_and_correspondence(tetrahedron, flip, preserve):
    mesh = tetrahedron
    r, t = landmark_frame(A, P, D)
    result = align_mesh(mesh, r, t, resolution=0.3, flipy=flip,
                        preserve_orientation=preserve, length_unit="mm")
    reflect = np.array([1, -1, 1] if flip else [1, 1, 1])
    expected_sign = -1 if flip and not preserve else 1
    assert_allclose(result.v, (mesh.v @ r.T + t)*0.3*reflect)
    assert_array_equal(result.f, mesh.f[:, [0, 2, 1]] if flip and preserve else mesh.f)
    assert_allclose(face_normals(result), face_normals(mesh) @ r.T*reflect*expected_sign, atol=1e-14)
    assert_allclose(surface_area(result), surface_area(mesh)*0.3**2, rtol=2e-15)
    assert_allclose(signed_volume(result), signed_volume(mesh)*0.3**3*expected_sign, rtol=2e-15)
    assert topology(result) == topology(mesh)
    assert result.length_unit == "mm" and mesh.length_unit == "um"
    assert not result.v.flags.writeable
    assert not np.shares_memory(result.v, mesh.v)
    inward = Mesh(mesh.v, mesh.f[:, [0, 2, 1]], length_unit=mesh.length_unit)
    aligned_inward = align_mesh(inward, r, t, flipy=flip)
    assert signed_volume(aligned_inward) < 0  # Preservation does not infer outwardness.


def test_mesh_alignment_preserves_uv_and_interpolation(plane):
    from tubular.surface import locate_chart
    r, t = landmark_frame(A, P, D)
    result = align_mesh(plane, r, t, flipy=True)
    query = [[0.2, 0.3], [0.7, 0.6]]
    before = locate_chart(plane, query).interpolate(plane.v)
    after = locate_chart(result, query).interpolate(result.v)
    assert_allclose(after, (before @ r.T+t)*[1, -1, 1], atol=2e-15)
    assert_array_equal(plane.uv, result.uv)
    assert result.length_unit == plane.length_unit


@pytest.mark.parametrize("kw", [{"resolution": 0}, {"resolution": float("inf")},
                                {"flipy": 1}, {"preserve_orientation": "yes"},
                                {"length_unit": ""}])
def test_alignment_rejects_invalid_options(plane, kw):
    with pytest.raises((ValueError, TypeError)):
        align_mesh(plane, np.eye(3), [0, 0, 0], **kw)


def test_class_frame_atomic_validation_and_independent_arrays():
    tubi = TubULAR(opts={"resolution": 0.5, "flipy": True})
    returned = tubi.computeAPDVCoords(A, P, D)
    r, t = tubi.getRotTrans()
    returned[0][:] = -100
    returned[3][:] = 0
    returned[4][:] = 0
    assert_allclose(tubi.getRotTrans()[0], r)
    assert_allclose(tubi.getRotTrans()[1], t)
    assert tubi.APDV.resolution == 0.5 and tubi.flipy
    for kwargs in ({"origin": P}, {"resolution": -1}, {"flipy": "yes"}):
        with pytest.raises(ValueError):
            tubi.computeAPDVCoords(A, P, D, **kwargs)
        assert_allclose(tubi.getRotTrans()[0], r)
        assert_allclose(tubi.getRotTrans()[1], t)
    assert_allclose(A, [2, -3, 4])
    tubi.computeAPDVCoords(A, P, D, resolution=1, flipy=False)
    assert_allclose(tubi.xyz2APDV([A]), 0, atol=1e-15)


def test_series_preview_registration_and_file_sources(tmp_path, tetrahedron):
    path = tmp_path / "raw.npz"
    second = tetrahedron.with_vertices(tetrahedron.v*2)
    save_mesh(path, second)
    tubi = TubULAR(time_points=[3, 7], opts={"resolution": 0.5, "spaceUnits": "mm",
                    "timeInterval": 2, "meshDir": tmp_path, "flipy": True},
                    mesh_sources={"rawMesh": {3: tetrahedron, 7: path},
                                  "alignedMesh": {3: tetrahedron}})
    tubi.computeAPDVCoords(A, P, D)
    tubi.setTime(3)
    raw = tubi.getCurrentRawMesh()
    old = tubi.getCurrentAlignedMesh()
    result = tubi.alignMeshesAPDV()
    assert tubi.getCurrentAlignedMesh() is old and tubi.getCurrentRawMesh() is raw
    assert_array_equal(result["timepoints"], [3, 7])
    assert_array_equal(result["times"], [0, 8])
    assert_array_equal(result["face_corner_order"], [0, 2, 1])
    assert result["source_length_unit"] == "um" and result["length_unit"] == "mm"
    assert_allclose(result["meshes"][1].v, tubi.xyz2APDV(second.v))
    result = tubi.alignMeshesAPDV(register=True)
    assert tubi.currentTime == 3 and tubi.currentMesh.alignedMesh is None
    assert tubi.getCurrentRawMesh() is raw
    assert tubi.getCurrentAlignedMesh() is result["meshes"][0]
    assert tubi.loadCurrentAlignedMesh() is result["meshes"][0]
    subset = tubi.alignMeshesAPDV(time_points=[7], preserve_orientation=False)
    assert_array_equal(subset["face_corner_order"], [0, 1, 2])
    assert_array_equal(subset["meshes"][0].f, second.f)
    assert set(tmp_path.iterdir()) == {path}


def test_failed_series_does_not_partially_register(tetrahedron):
    tubi = TubULAR(time_points=[0, 2], mesh_sources={"rawMesh": {0: tetrahedron},
                    "alignedMesh": {0: tetrahedron, 2: tetrahedron}})
    tubi.computeAPDVCoords(A, P, D)
    tubi.setTime(0)
    cached = tubi.getCurrentAlignedMesh()
    with pytest.raises(FileNotFoundError):
        tubi.alignMeshesAPDV(register=True)
    assert tubi.getCurrentAlignedMesh() is cached
    tubi.register_mesh(2, Mesh(tetrahedron.v, tetrahedron.f, length_unit="mm"))
    with pytest.raises(ValueError, match="units"):
        tubi.alignMeshesAPDV(register=True)
    assert tubi.getCurrentAlignedMesh() is cached
    assert tubi.currentTime == 0
    tubi.setTime(2)
    assert tubi.getCurrentAlignedMesh() is tetrahedron


@pytest.mark.parametrize("times", [[], [0, 0], [2, 0], [1], [np.nan]])
def test_invalid_alignment_times(times):
    tubi = TubULAR(time_points=[0, 2])
    with pytest.raises(ValueError):
        tubi.alignMeshesAPDV(time_points=times)


def test_matlab_alignment_reference():
    root = Path(__file__).parent / "data"
    path = root / "matlab_alignment_reference.mat"
    provenance = json.loads((root / "alignment_provenance.json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance["fixture_sha256"]
    data = loadmat(path, simplify_cells=True)
    for i in range(len(data["anterior"])):
        r, t = landmark_frame(data["anterior"][i], data["posterior"][i], data["dorsal"][i])
        # MATLAB's class exposes only three outputs and saves R/t using
        # dlmwrite's five significant digits. Compare the same quantization.
        rounded_r = np.array([float(f"{x:.5g}") for x in r.ravel()]).reshape(3, 3)
        rounded_t = np.array([float(f"{x:.5g}") for x in t])
        assert_allclose(rounded_r, data["rotations"][:, :, i], rtol=0, atol=1e-14)
        assert_allclose(rounded_t, data["translations"][i], rtol=0, atol=1e-14)
        mesh = Mesh(data["vertices"], data["faces"].astype(int)-1)
        for flip, name in ((False, "aligned"), (True, "mirrored")):
            result = align_mesh(mesh, data["transformRotations"][:, :, i], data["translations"][i],
                                resolution=data["resolution"], flipy=flip,
                                preserve_orientation=False)
            assert_allclose(result.v, data[name][:, :, i], rtol=1e-12, atol=1e-13)
            assert_array_equal(result.f, mesh.f)
