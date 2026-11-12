import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.io import loadmat, savemat

from tubular import Mesh, TubULAR
from tubular.surface import locate_chart


def test_chart_affine_fields_and_shared_boundaries(plane):
    uv = np.array([[0.7, 0.2], [0.2, 0.8], [0.5, 0.5], [0, 0], [1, 1], [2, 0.5]])
    located = locate_chart(plane, uv)
    assert_array_equal(located.face_indices, [0, 1, 0, 0, 0, -1])
    xyz = located.interpolate(plane.v)
    u, v = uv[:5].T
    assert_allclose(xyz[:5], np.column_stack((2*u+v, 3*v, u)))
    assert np.isnan(xyz[-1]).all() and np.isnan(located.barycentric[-1]).all()
    assert_allclose(located.barycentric[:5].sum(axis=1), 1)
    assert_allclose(located.interpolate(plane.v[:, 0])[:5], 2*u+v)
    assert_array_equal(located.interpolate([7, 9], location="faces")[:5], [7, 9, 7, 7, 7])
    assert not located.face_indices.flags.writeable
    assert not located.barycentric.flags.writeable


@pytest.mark.parametrize("scale", [1e-8, 1, 1e8])
def test_chart_rescaling_reversed_winding_and_single_point(plane, scale):
    chart = plane.u*scale + [3*scale, -5*scale]
    reversed_mesh = Mesh(plane.v, plane.f[:, ::-1], chart, "um")
    query = np.array([[0.8, 0.2]])*scale + [3*scale, -5*scale]
    assert_allclose(locate_chart(reversed_mesh, query).interpolate(plane.v), [[1.8, 0.6, 0.8]])


def test_chart_does_not_fill_holes():
    uv = np.array([[-2, -2], [2, -2], [2, 2], [-2, 2],
                   [-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
    faces = np.array([[0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                      [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    mesh = Mesh(np.column_stack((uv, np.zeros(8))), faces, uv)
    located = locate_chart(mesh, [[0, 0], [1.5, 0], [3, 0]])
    assert_array_equal(located.valid, [False, True, False])
    assert_allclose(located.interpolate(mesh.v)[1], [1.5, 0, 0])


@pytest.mark.parametrize("height", [0, 2])
def test_chart_rejects_ambiguous_overlaps_even_when_xyz_agrees(height):
    uv = np.array([[0, 0], [1, 0], [0, 1]]*2)
    vertices = np.column_stack((uv, [0, 0, 0, height, height, height]))
    mesh = Mesh(vertices, np.array([[0, 1, 2], [3, 4, 5]]), uv)
    for query in ([[0.2, 0.2]], [[0, 0]]):
        with pytest.raises(ValueError, match="overlapping"):
            locate_chart(mesh, query)


def test_chart_rejects_wrong_fields_and_handles_empty_queries(plane):
    empty = locate_chart(plane, np.empty((0, 2)))
    assert empty.interpolate(plane.v).shape == (0, 3)
    for field in ([1, 2], [1, 2, np.nan, 4], [[1, 2, 3, 4]]):
        with pytest.raises(ValueError):
            locate_chart(plane, [[0.3, 0.2]]).interpolate(field)
    with pytest.raises(ValueError, match="UV"):
        locate_chart(Mesh(plane.v, plane.f), [[0, 0]])


def test_uv2apdv_uses_aligned_chart_without_mutation_or_second_transform(plane):
    mesh = Mesh(plane.v, plane.f, plane.u * [4, 1], "um")
    tubi = TubULAR(opts={"flipy": True, "rot": np.eye(3), "trans": [100, 200, 300]},
                   mesh_sources={"spcutMeshSmRS": {0: mesh}})
    tubi.setTime(0)
    uv = np.array([[1.6, 0.2], [0.4, 0.8], [3, 0]])
    original = uv.copy()
    xyz, faces, chart, weights = tubi.uv2APDV(uv, "SPSMRS", 2, return_details=True)
    assert_allclose(xyz[:2], [[1.8, 0.6, 0.8], [1.2, 2.4, 0.2]])
    assert np.isnan(xyz[2]).all() and faces[2] == -1
    assert_allclose(chart.u, plane.u * [2, 1])
    assert_allclose(mesh.u, plane.u * [4, 1])
    assert_array_equal(uv, original)
    assert_allclose(weights[:2].sum(axis=1), 1)
    with pytest.raises(NotImplementedError, match="sphi"):
        tubi.uv2APDV(uv, "sprs")
    with pytest.raises(NotImplementedError, match="vmax"):
        tubi.uv2APDV(uv, vmax=2)


def test_grid_interpolation_constant_and_method_selection():
    tubi = TubULAR(opts={"nU": 3, "nV": 3})
    xy = [[-30, -20], [3, 1], [7, 8], [30, 50]]
    opts = {"Lx": 9, "Ly": 12}
    assert_allclose(tubi.interpolateOntoPullbackXY(xy, np.full(9, 4.2), opts), 4.2)
    # Constant across hoops, piecewise linear along U: nodes x=1,5,9.
    field = np.array([[10, 10, 10], [20, 20, 20], [40, 40, 40]])
    assert_allclose(tubi.interpolateOntoPullbackXY([[2, 5], [8, 5]], field, opts), [12.5, 35])
    opts["iMethod"] = "nearest"
    assert_allclose(tubi.interpolateOntoPullbackXY([[2, 5], [8, 5]], field, opts), [10, 40])
    assert_allclose(tubi.interpolateOntoPullbackXY([[3, 5], [7, 5]], field, opts), [20, 40])
    assert_allclose(tubi.interpolateOntoPullbackXY([[2, 5]], field.ravel(order="F"), opts), [10])


@pytest.mark.parametrize("option", [{}, {"Lx": 9}, {"Lx": 1, "Ly": 5},
                                    {"Lx": 9, "Ly": 12, "imCoords": "other"},
                                    {"Lx": 9, "Ly": 12, "sfLocation": "faces"},
                                    {"Lx": 9, "Ly": 12, "iMethod": "cubic"},
                                    {"Lx": 9, "Ly": 12, "preview": True}])
def test_grid_options_fail_explicitly(option):
    with pytest.raises((ValueError, NotImplementedError)):
        TubULAR(opts={"nU": 3, "nV": 3}).interpolateOntoPullbackXY([[1, 1]], np.ones(9), option)


def test_grid_field_shape_and_option_precedence():
    tubi = TubULAR(opts={"nU": 2, "nV": 3})
    tubi.piv.imCoords = "unsupported"
    opts = {"Lx": 5, "Ly": 8, "imCoords": "sp_sme"}
    assert_allclose(tubi.interpolateOntoPullbackXY([[1, 1]], np.ones(6), opts), 1)
    for field in (np.ones(5), np.ones((3, 2)), np.ones((1, 6)), np.ones((2, 3, 1))):
        with pytest.raises(ValueError, match="scalar_field"):
            tubi.interpolateOntoPullbackXY([[1, 1]], field, opts)


def test_bounds_frame_padding_sources_and_selection(plane):
    tubi = TubULAR(time_points=[0, 3], opts={"rot": np.eye(3), "trans": [1, -2, 3],
                    "resolution": 0.5, "flipy": True, "normalShift": -2},
                   mesh_sources={"rawMesh": {0: plane, 3: plane.with_vertices(plane.v + [-4, 5, -2])}})
    tubi.setTime(0)
    cached = tubi.getCurrentRawMesh()
    raw, pixels, physical, buffered = tubi.measureXYZLims()
    assert_allclose(raw, [[-4, 3], [0, 8], [-2, 1]])
    assert_allclose(pixels, [[-3, 4], [-6, 2], [1, 4]])
    assert_allclose(physical, pixels*0.5)
    assert_allclose(buffered, physical + [-2, 2])
    assert tubi.currentTime == 0 and tubi.getCurrentRawMesh() is cached
    raw[:] = 999
    assert tubi.plotting.xyzlim_raw[0, 0] == -4
    tubi.register_mesh(3, plane)
    assert_allclose(tubi.getXYZLims()[0], [[0, 3], [0, 3], [0, 1]])
    tubi.set_apdv_frame(np.eye(3), [0, 0, 0], resolution=2, flipy=False)
    assert_allclose(tubi.getXYZLims()[2], [[0, 6], [0, 6], [0, 2]])


def test_bounds_validate_complete_series_without_partial_results(plane):
    tubi = TubULAR(time_points=[0, 1], opts={"rot": np.eye(3), "trans": [0, 0, 0]},
                   mesh_sources={"rawMesh": {0: plane}})
    with pytest.raises(FileNotFoundError):
        tubi.measureXYZLims()
    assert tubi.plotting.xyzlim_raw is None
    tubi.register_mesh(1, Mesh(plane.v, plane.f, plane.u, "mm"))
    with pytest.raises(ValueError, match="units differ"):
        tubi.getXYZLims()
    assert tubi.plotting.xyzlim_raw is None
    assert_allclose(tubi.measureXYZLims({"timePoints": [0]})[0], [[0, 3], [0, 3], [0, 1]])
    with pytest.raises(NotImplementedError, match="overwrite"):
        tubi.measureXYZLims({"overwrite": True})


def test_centerline_reload_reflection_and_independent_cache(tmp_path):
    path = tmp_path / "cline_0002.mat"
    data = {"mss": [0, 1, 2], "mcline": [[0, 1, 0], [1, 2, 0], [2, 3, 0]],
            "avgpts": [[0, 1, 0], [2, 3, 0]]}
    savemat(path, data)
    tubi = TubULAR(time_points=[2, 4], opts={"flipy": True,
                    "fullFileBase": {"clineDVhoop": str(tmp_path / "cline_%04d.mat")}})
    with pytest.raises(ValueError, match="setTime"):
        tubi.getCurrentClineDVhoop()
    tubi.setTime(2)
    mss, cline, avg = tubi.getCurrentClineDVhoop()
    assert_allclose(cline[:, 1], [-1, -2, -3])
    assert_allclose(avg[:, 1], [-1, -3])
    cline[:] = 99
    assert tubi.currentCline.mcline[0, 1] == -1
    assert_allclose(tubi.getCurrentClineDVhoop()[1][:, 1], [-1, -2, -3])  # No double reflection.
    previous = tubi.currentCline.mcline
    data["mss"] = [0, 0, 1]
    savemat(path, data)
    with pytest.raises(ValueError, match="increasing"):
        tubi.getCurrentClineDVhoop()
    assert tubi.currentCline.mcline is previous
    tubi.setTime(4)
    assert tubi.currentCline.mcline is None


def test_original_matlab_surface_references(tmp_path):
    root = Path(__file__).parent
    fixture = root / "data/matlab_surface_reference.mat"
    provenance = json.loads((root / "data/surface_provenance.json").read_text())
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == provenance["fixture_sha256"]
    generator = root.parent / "scripts/generate_surface_reference.m"
    assert hashlib.sha256(generator.read_bytes()).hexdigest() == provenance["generator_sha256"]
    data = loadmat(fixture, squeeze_me=True)
    mesh = Mesh(data["vertices"], data["faces"].astype(int)-1, data["chart"], "um")
    tubi = TubULAR(time_points=[0, 2], opts={"nU": int(data["nU"]), "nV": int(data["nV"]),
                    "rot": data["rot"], "trans": data["trans"], "resolution": data["resolution"],
                    "flipy": True, "normalShift": -2}, mesh_sources={"spcutMeshSmRS": {0: mesh},
                    "rawMesh": {0: mesh.with_vertices(data["rawVertices0"]),
                                2: mesh.with_vertices(data["rawVertices2"])}})
    tubi.setTime(0)
    for query, umax, xyz_key, face_key, bary_key in (
            ("queryUV", 1, "mappedXYZ", "mappedFaces", "barycentric"),
            ("scaledQuery", data["umax"], "scaledXYZ", "scaledFaces", "scaledBarycentric")):
        xyz, ids, chart, w = tubi.uv2APDV(data[query], umax=umax, return_details=True)
        valid = np.isfinite(data[face_key])
        assert_allclose(xyz, data[xyz_key], rtol=1e-12, atol=1e-12, equal_nan=True)
        assert_array_equal(ids[valid], data[face_key][valid].astype(int)-1)
        assert_array_equal(ids[~valid], -1)
        assert_allclose(w[valid], data[bary_key][valid], atol=1e-12)
    assert_allclose(tubi.interpolateOntoPullbackXY(data["queryXY"], data["field"],
                    {"Lx": data["Lx"], "Ly": data["Ly"]}), data["interpolatedField"], rtol=1e-11, atol=1e-12)
    assert_allclose(tubi.interpolateOntoPullbackXY(data["nearestQueries"], data["field"],
                    {"Lx": data["Lx"], "Ly": data["Ly"], "iMethod": "nearest"}), data["nearestField"], rtol=1e-11, atol=1e-12)
    for method, keys in ((tubi.measureXYZLims, ("boundsRaw", "boundsPixels", "boundsPhysical", "boundsBuffered")),
                         (tubi.getXYZLims, ("getRaw", "getPixels", "getPhysical", "getBuffered"))):
        for result, key in zip(method(), keys):
            assert_allclose(result, data[key], rtol=1e-11, atol=1e-12)
    path = tmp_path / "cline_0.mat"
    savemat(path, {k: data[k] for k in ("mss", "mcline", "avgpts")})
    tubi.fullFileBase.clineDVhoop = str(path)
    for result, key in zip(tubi.getCurrentClineDVhoop(), ("loadedMss", "loadedCline", "loadedAvgpts")):
        assert_allclose(result, data[key], rtol=1e-11, atol=1e-12)
