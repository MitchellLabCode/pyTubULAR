import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from tubular import Mesh
from tubular.imaging import pullback_image, sample_volume


def test_anisotropic_sampling_axis_order_and_integer_interpolation():
    z, y, x = np.indices((5, 6, 7))
    volume = (100 * z + 10 * y + x).astype(np.uint16)
    spacing, origin = np.array([2., 3., 5.]), np.array([-4., 10., 2.])
    index = np.array([[1.5, 2.25, 3.5], [6, 5, 4], [0, 0, 0]])
    values = sample_volume(volume, origin + index * spacing, spacing_xyz=spacing, origin_xyz=origin)
    assert_allclose(values, [374., 456., 0.])
    assert values.dtype == np.float64
    outside = sample_volume(volume, [[-0.01, 0, 0], [7, 0, 0]])
    assert np.isnan(outside).all()


def test_pullback_affine_volume(plane):
    z, y, x = np.indices((4, 5, 5))
    volume = 100 * z + 10 * y + x
    u, v = np.linspace(0, 1, 7), np.linspace(0, 1, 9)
    progress = []
    result = pullback_image(plane, volume, u, v,
                            progress=lambda done, total: progress.append((done, total)))
    uu, vv = np.meshgrid(u, v)
    assert_allclose(result.image, 102 * uu + 31 * vv, atol=1e-12)
    assert result.image.shape == (9, 7)
    assert result.valid.all()
    assert progress[-1] == (2, 2)
    assert_allclose(result.xyz[..., 0], 2 * uu + vv)


def test_chart_holes_are_not_retriangulated():
    uv = np.array([[0., 0.], [1., 0.], [0., 1.], [2., 0.], [3., 0.], [3., 1.]])
    vertices = np.column_stack((uv, np.ones(6)))
    mesh = Mesh(vertices, [[0, 1, 2], [3, 4, 5]], uv)
    result = pullback_image(mesh, np.ones((3, 3, 5)), [0.2, 1.5, 2.8], [0.1])
    assert_allclose(result.image[0, [0, 2]], 1)
    assert np.isnan(result.image[0, 1])
    assert result.face_indices[0, 1] == -1


def test_overlapping_chart_rejected():
    uv = np.array([[0., 0.], [1., 0.], [0., 1.]] * 2)
    vertices = np.column_stack((uv, [0., 0., 0., 1., 1., 1.]))
    mesh = Mesh(vertices, [[0, 1, 2], [3, 4, 5]], uv)
    with pytest.raises(ValueError, match="overlapping"):
        pullback_image(mesh, np.ones((3, 3, 3)), [0.2], [0.2])


def test_reversed_chart_winding(plane):
    reverse = Mesh(plane.vertices, plane.faces[:, ::-1], plane.uv)
    z, y, x = np.indices((4, 5, 5))
    u = np.linspace(0, 1, 4)
    first = pullback_image(plane, x + 10 * y + 100 * z, u, u)
    second = pullback_image(reverse, x + 10 * y + 100 * z, u, u)
    assert_allclose(first.image, second.image, atol=1e-12)


def test_cancellation(plane):
    with pytest.raises(InterruptedError):
        pullback_image(plane, np.ones((5, 5, 5)), [0, 1], [0, 1], cancelled=lambda: True)


def test_volume_outside_marks_invalid(plane):
    result = pullback_image(plane, np.ones((1, 1, 1)), [0, 1], [0, 1])
    assert_array_equal(result.valid, [[True, False], [False, False]])
    assert (result.face_indices >= 0).all()


@pytest.mark.parametrize("spacing", [(0, 1, 1), (-1, 1, 1), (np.nan, 1, 1), (1, 1)])
def test_invalid_spacing(spacing):
    with pytest.raises(ValueError):
        sample_volume(np.ones((3, 3, 3)), [[0, 0, 0]], spacing_xyz=spacing)
