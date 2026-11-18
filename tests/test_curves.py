import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.io import loadmat, savemat

from tubular import TubULAR
from tubular.curves import curve_arclength, writhe_gauss, writhe_levitt


def helix(n=25):
    t = np.linspace(0, 2.3*np.pi, n)
    return np.column_stack((0.6*t, 1.2*np.sin(t)+0.04*t, 0.7*np.cos(t)+0.03*np.cos(3*t)))


def segment_quadrature(xyz, order=36):
    """Independent Gauss-Legendre integration over segment pairs; no solid angles."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes, weights = (nodes+1)/2, weights/2
    segments = np.diff(xyz, axis=0)
    total = 0.
    for i in range(2, len(segments)):
        for j in range(i-1):
            a = xyz[i]+nodes[:, None]*segments[i]
            b = xyz[j]+nodes[:, None]*segments[j]
            delta = a[:, None, :]-b[None, :, :]
            integrand = (delta @ np.cross(segments[i], segments[j])) / np.linalg.norm(delta, axis=2)**3
            total += weights @ integrand @ weights
    return total/(2*np.pi)


def test_levitt_against_independent_double_integral():
    curve = helix(13)
    total, density = writhe_levitt(curve)
    expected = segment_quadrature(curve)
    assert_allclose(total, expected, rtol=1e-11, atol=1e-12)
    assert_allclose(density @ np.diff(curve_arclength(curve)), total, atol=1e-14)


@pytest.mark.parametrize("fn", [writhe_levitt, writhe_gauss])
def test_writhe_rigid_motion_scaling_reflection_reversal(fn):
    curve = helix()
    original = curve.copy()
    total, density = fn(curve)
    q, _ = np.linalg.qr(np.array([[1., 2., 3.], [-2., 1., 1.], [1., 3., -1.]]))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    moved, moved_density = fn(curve @ q.T + [2, -4, 7])
    assert_allclose(moved, total, rtol=1e-10, atol=1e-12)
    assert_allclose(moved_density, density, rtol=1e-9, atol=1e-11)
    for scale in (1e-6, 3.4, 1e6):
        scaled, scaled_density = fn(curve*scale)
        assert_allclose(scaled, total, rtol=1e-10, atol=1e-12)
        assert_allclose(scaled_density*scale, density, rtol=1e-9, atol=1e-11)
    reflected, reflected_density = fn(curve * [1, -1, 1])
    assert_allclose(reflected, -total, atol=1e-12)
    assert_allclose(reflected_density, -density, atol=1e-12)
    reversed_total, reversed_density = fn(curve[::-1])
    assert_allclose(reversed_total, total, rtol=1e-10, atol=1e-12)
    assert_allclose(reversed_density, density[::-1], rtol=1e-9, atol=1e-11)
    assert_array_equal(curve, original)


def test_planar_and_straight_curves_have_zero_writhe():
    t = np.linspace(-1.3, 1.2, 17)
    for curve in (np.column_stack((t, t*t, 2*t+3*t*t)),
                  np.column_stack((t, 2*t, -t))):
        assert_allclose(writhe_levitt(curve)[0], 0, atol=1e-13)
        assert_allclose(writhe_gauss(curve)[0], 0, atol=1e-13)


def test_closed_curve_and_repeated_closure():
    theta = np.linspace(0, 2*np.pi, 17)[:-1]
    curve = np.column_stack((np.cos(theta), np.sin(theta), 0.3*np.cos(3*theta)))
    implicit, implicit_density = writhe_levitt(curve, closed=True)
    explicit, explicit_density = writhe_levitt(np.vstack((curve, curve[0])), closed=True)
    assert_allclose(implicit, explicit, atol=1e-14)
    assert_allclose(implicit_density, explicit_density, atol=1e-14)
    assert len(implicit_density) == len(curve)
    with pytest.raises(ValueError, match="intersect"):
        writhe_levitt(np.vstack((curve, curve[0])))


@pytest.mark.parametrize("curve", [
    [[0, 0, 0], [0, 0, 0], [1, 0, 0]],
    [[0, 0, 0], [1, 0, 0], [0.5, 0, 0]],
    [[0, 0, 0], [1, 1, 0], [0, 1, 0], [1, 0, 0]],
    [[0, 0, 0], [1, 0, 0], [1, 1e-10, 0], [0, -1e-10, 0]],
    [[0, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]],
    [[0, 0, 0], [1, np.nan, 0]],
    [[0, 0, 0]],
])
@pytest.mark.parametrize("fn", [writhe_levitt, writhe_gauss])
def test_invalid_and_intersecting_curves_fail(curve, fn):
    with pytest.raises(ValueError):
        fn(curve)


def test_gauss_normalization_density_and_refinement():
    curve = helix(21)
    ss = curve_arclength(curve)
    total, density = writhe_gauss(curve, ss)
    legacy, legacy_density = writhe_gauss(curve, ss, legacy=True)
    assert_allclose(legacy, total*3/(2*np.pi), atol=1e-14)
    assert_allclose(legacy_density, density*3/(2*np.pi), atol=1e-14)
    assert_allclose(density @ np.gradient(ss), total, atol=1e-14)
    # This is endpoint-weighted sampled quadrature, not exact polygon writhe.
    errors = []
    for n in (41, 81, 161):
        current = helix(n)
        errors.append(abs(writhe_gauss(current)[0]-writhe_levitt(current)[0]))
    assert errors[2] < errors[1] < errors[0]
    with pytest.raises(ValueError, match="strictly increasing"):
        writhe_gauss(curve, np.zeros(len(curve)))


def test_centerline_registration_owns_arrays_and_invalidates_current():
    curve = helix(17)
    tubi = TubULAR(opts={"flipy": True})
    tubi.setTime(0)
    tubi.register_centerline(0, curve)
    original = curve.copy()
    curve[:] = 999
    mss, cline, avg = tubi.getCurrentClineDVhoop()
    assert_allclose(cline, original*[1, -1, 1])
    assert_allclose(avg, cline)
    assert_allclose(mss, curve_arclength(original))
    tubi.register_centerline(0, original*2)
    assert tubi.currentCline.mcline is None
    assert_allclose(tubi.getCurrentClineDVhoop()[1], original*[2, -2, 2])


def test_length_physical_time_nonuniform_gaps_and_reference_offset():
    tubi = TubULAR(time_points=[2, 5, 9], opts={"timeInterval": 2, "t0": 5})
    for time, length in ((2, 4), (5, 10), (9, 18)):
        tubi.register_centerline(time, [[0, 0, 0], [length, 0, 0]], mss=[7, 7+length])
    tubi.setTime(5)
    tubi.getCurrentClineDVhoop()
    cached = tubi.currentCline
    result = tubi.measureLength()
    assert_allclose(result["lengths"], [4, 10, 18])
    assert_allclose(result["polygonal_length"], [4, 10, 18])
    assert_allclose(result["times"], [-6, 0, 8])
    assert_allclose(result["dl"], 1)
    assert tubi.currentTime == 5 and tubi.currentCline is cached
    assert np.isnan(tubi.measureLength({"timePoints": [5]})["dl"]).all()


def test_writhe_series_endpoint_counts_and_explicit_reflection(tmp_path):
    curve = helix(21)
    tubi = TubULAR(time_points=[5, 9], opts={"flipy": True, "timeInterval": 2, "meshDir": tmp_path})
    tubi.register_centerline(5, curve)
    tubi.register_centerline(9, curve*[1, -1, 1])
    tubi.setTime(5)
    tubi.getCurrentClineDVhoop()
    cached = tubi.currentCline
    wr, densities, derivative, length, curves = tubi.measureWrithe({"omit_endpts": [2, 3]})
    kept = curve[2:-3]
    expected = writhe_levitt(kept)[0]
    assert_allclose(wr["Levitt"], [expected, -expected], atol=1e-12)
    assert_allclose(derivative["Levitt"], -2*expected/8, atol=1e-12)
    assert_array_equal(curves[0]["kept_indices"], np.arange(2, 18))
    assert len(densities["Levitt"][0]) == len(kept)-1
    assert len(densities["Gauss"][0]) == len(kept)
    assert length["gauss_convention"] == "standard_1_over_4pi"
    reflected = tubi.measureWrithe({"omit_endpts": [2, 3], "flipy_centerline": True})[0]
    assert_allclose(reflected["Levitt"], -wr["Levitt"], atol=1e-12)
    assert tubi.currentTime == 5 and tubi.currentCline is cached
    assert list(tmp_path.iterdir()) == []
    assert "polar" not in wr


@pytest.mark.parametrize("options", [{"filter_curve": 7}, {"omit_endpts": -1},
                                     {"omit_endpts": True}, {"omit_endpts": [1, 2, 3]},
                                     {"omit_endpts": 2.5}, {"omit_endpts": 100},
                                     {"polar": True}, {"timePoints": []},
                                     {"flipy_centerline": 1}, {"gauss_legacy": 1}])
def test_writhe_rejects_unsupported_or_invalid_options(options):
    tubi = TubULAR()
    tubi.register_centerline(0, helix(17))
    with pytest.raises((ValueError, NotImplementedError)):
        tubi.measureWrithe(options)


def test_missing_series_does_not_change_selection():
    tubi = TubULAR(time_points=[3, 7])
    tubi.register_centerline(3, helix())
    tubi.setTime(3)
    tubi.getCurrentClineDVhoop()
    cached = tubi.currentCline
    with pytest.raises(FileNotFoundError):
        tubi.measureWrithe()
    assert tubi.currentTime == 3 and tubi.currentCline is cached


def test_matlab_writhe_kernels_and_series(tmp_path):
    root = Path(__file__).parent
    fixture = root / "data/matlab_writhe_reference.mat"
    provenance = json.loads((root / "data/writhe_provenance.json").read_text())
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == provenance["fixture_sha256"]
    generator = root.parent / "scripts/generate_writhe_reference.m"
    assert hashlib.sha256(generator.read_bytes()).hexdigest() == provenance["generator_sha256"]
    r = loadmat(fixture, squeeze_me=True, struct_as_record=False)
    for curve, total_name, density_name, closed in (
            ("xyz", "levitt", "levittDensity", False),
            ("reflected", "levittReflected", "densityReflected", False),
            ("closedXYZ", "levittClosed", "densityClosed", True)):
        total, density = writhe_levitt(r[curve], closed=closed)
        assert_allclose(total, r[total_name], rtol=1e-11, atol=1e-12)
        assert_allclose(density, r[density_name], rtol=1e-10, atol=1e-12)
    total, density = writhe_gauss(r["xyz"], r["ss"], legacy=True)
    assert_allclose(total, r["gaussLegacy"], rtol=1e-11, atol=1e-12)
    assert_allclose(density, r["gaussLegacyDensity"], rtol=1e-11, atol=1e-12)
    # Exercise file paths with actual acquisition coordinates, despite t0=9.
    tubi = TubULAR(time_points=r["times"], opts={"t0": 9, "timeInterval": 2,
                   "fullFileBase": {"clineDVhoop": str(tmp_path / "cline_%04d.mat")}})
    for time, entry in zip(r["times"], r["seriesCurves"]):
        savemat(tmp_path / f"cline_{int(time):04d}.mat", {"mss": entry.ssx,
                "mcline": entry.xyzp, "avgpts": entry.xyzp[::3]})
    wr, densities, _, lengths, curves = tubi.measureWrithe({"omit_endpts": [0, 1], "gauss_legacy": True})
    for name in ("Levitt", "Gauss"):
        assert_allclose(wr[name], getattr(r["seriesWr"], name), rtol=1e-11, atol=1e-12)
        for actual, expected in zip(densities[name], getattr(r["seriesDensity"], name)):
            assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
    assert_allclose(lengths["lengths"], r["seriesLength"].lengths, rtol=1e-12, atol=1e-12)
    for actual, expected in zip(curves, r["seriesCurves"]):
        assert_allclose(actual["xyzp"], expected.xyzp, atol=1e-12)
        assert_allclose(actual["ssx"], expected.ssx, atol=1e-12)
