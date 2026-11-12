from pathlib import Path
import hashlib
import json

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.io import loadmat

from tubular import TubULAR


def test_static_matlab_reference_and_provenance():
    root = Path(__file__).resolve().parents[1]
    data = root / "tests/data"
    provenance = json.loads((data / "class_provenance.json").read_text())
    for key, path in (("fixture_sha256", data / "matlab_class_reference.mat"),
                       ("inventory_sha256", data / "matlab_class_methods.json"),
                       ("generator_sha256", root / "scripts/generate_class_reference.m")):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance[key]
    r = loadmat(data / "matlab_class_reference.mat")
    size, xy, dxy = r["imageSize"].ravel(), r["XY"], r["dXY"]
    outputs = {
        "uvSingle": TubULAR.XY2uv(size, xy, False, 2.3, 3.7),
        "uvDouble": TubULAR.XY2uv(size, xy, True, 2.3, 3.7, legacy=True),
        "xySingle": TubULAR.uv2XY(size, r["uvSingle"], False, 2.3, 3.7),
        "xyDouble": TubULAR.uv2XY(size, r["uvDouble"], True, 2.3, 3.7),
        "duvSingle": TubULAR.dXY2duv(size, dxy, False, 2.3, 3.7),
        "duvDouble": TubULAR.dXY2duv(size, dxy, True, 2.3, 3.7, legacy=True),
        "coverEven": TubULAR.extendImage2DoubleCover(r["imEven"]),
        "coverOdd": TubULAR.extendImage2DoubleCover(r["imOdd"]),
        "singleDouble": TubULAR.singleToDoubleCover(xy, 17),
        "doubleSingle": TubULAR.doubleToSingleCover(xy, 34),
    }
    for name, actual in outputs.items():
        assert_allclose(actual, r[name], rtol=1e-12, atol=1e-12, err_msg=name)
    x, y = TubULAR.clipXY([-4, 1, 7, 20], [-1, 1, 17, 18], 13, 17, legacy=True)
    assert_array_equal(x, r["clipX"].ravel())
    assert_array_equal(y, r["clipY"].ravel())
    magnitude, theta = TubULAR.dvAverageNematic(r["magnitude"], r["theta"])
    assert_allclose(magnitude, r["magAverage"].ravel(), atol=1e-12)
    assert_allclose(theta, r["thetaAverage"].ravel(), atol=1e-12)
    for name, indices in zip(("dorsal", "ventral", "left", "right"), TubULAR.quarterIndicesDV(33)):
        assert_array_equal(indices, r[name].ravel() - 1)


@pytest.mark.parametrize("double", [False, True])
@pytest.mark.parametrize("vmax", [0.4, 1., 3.7])
def test_image_chart_inverse_and_differentials(double, vmax):
    image = np.zeros((17, 13))
    xy = np.array([[1., 1.], [13., 17.], [2.25, 11.5]])
    dx = np.array([[0., 0.], [1.3, -0.8], [-2.5, 7.1]])
    uv = TubULAR.XY2uv(image, xy, double, 2.3, vmax)
    assert_allclose(TubULAR.uv2XY(image, uv, double, 2.3, vmax), xy, atol=1e-12)
    duv = TubULAR.dXY2duv(image, dx, double, 2.3, vmax)
    assert_allclose(duv, TubULAR.XY2uv(image, xy + dx, double, 2.3, vmax) - uv, atol=1e-12)
    assert_array_equal(duv[0], [0, 0])
    assert_allclose(TubULAR.XY2uv([13, 17], xy, double, 2.3, vmax), uv)
    if double:
        assert_allclose(uv[[0, 1], 1], [-0.5*vmax, 1.5*vmax])


def test_legacy_arithmetic_is_explicit():
    assert_array_equal(TubULAR.dXY2duv([13, 17], [[0, 0]], legacy=True), [[0, -0.5]])
    assert_array_equal(TubULAR.dXY2duv([13, 17], [[0, 0]]), [[0, 0]])
    assert_array_equal(TubULAR.XY2uv([13, 17], [[1, 1]], vmax=4), [[0, -2]])


def test_double_cover_preserves_wide_intensity_and_inputs():
    im = (np.arange(20).reshape(5, 4) * 1000).astype(np.uint16)
    original = im.copy()
    result = TubULAR.extendImage2DoubleCover(im)
    assert result.dtype == np.uint16 and result.shape == (11, 4)
    assert_array_equal(result[:3], im[-3:])
    assert_array_equal(result[3:8], im)
    assert_array_equal(result[8:], im[:3])
    assert_array_equal(im, original)
    with pytest.raises(NotImplementedError):
        TubULAR.extendImage2DoubleCover(im, True)


def test_clip_wraps_multiple_periods():
    x = np.array([-1., 1., 20., 3., 3.])
    y = np.array([-31., 1., 17., 49., 50.])
    clipped_x, clipped_y = TubULAR.clipXY(x, y, 13, 17)
    assert_array_equal(clipped_x, [1, 1, 13, 3, 3])
    assert_array_equal(clipped_y, [1, 1, 17, 1, 2])
    assert_array_equal(y, [-31, 1, 17, 49, 50])


def test_nematic_equivalence_under_pi_rotation():
    magnitude = np.array([[2., 2., 2.], [1., 2., 3.]])
    theta = np.array([[0.2, 0.2 + np.pi, 0.2 - np.pi], [0.1, 0.1, 0.1]])
    mag, angle = TubULAR.dvAverageNematic(magnitude, theta)
    assert_allclose(mag, [2, 2], atol=1e-12)
    assert_allclose(angle, [0.2, 0.1], atol=1e-12)


@pytest.mark.parametrize("size", [[1, 3], [3, 1], [2.5, 4], [np.nan, 4]])
def test_invalid_image_size(size):
    with pytest.raises(ValueError):
        TubULAR.XY2uv(size, [[1, 1]])


def test_image_pair_order_and_small_image_disambiguation():
    assert_allclose(TubULAR.XY2uv(np.zeros((2, 2)), [[2, 2]], False), [[1, 1]])
    assert_allclose(TubULAR.XY2uv(np.zeros((5, 9)), [[9, 5]], False), [[1, 1]])
    with pytest.raises(ValueError):
        TubULAR.XY2uv([9, 5], [1, 2])


def test_inverse_rotation_rejects_scale_and_reflection():
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    assert_allclose(TubULAR.invertRotation(rotation) @ rotation, np.eye(3))
    for invalid in (np.zeros((3, 3)), np.diag([1, 2, 1]), np.diag([1, -1, 1])):
        with pytest.raises(ValueError, match="orthonormal"):
            TubULAR.invertRotation(invalid)
