import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from tubular import Mesh
from tubular.geometry import surface_area
from tubular.grid import cylinder_mesh, glue_rectilinear_seam, rectilinear_faces
from tubular.mesh import topology


def test_exact_grid_order():
    assert_array_equal(rectilinear_faces(3, 3), [
        [0,1,4], [0,4,3], [3,4,7], [3,7,6],
        [1,2,5], [1,5,4], [4,5,8], [4,8,7]])


def test_glue_cylinder():
    cut = cylinder_mesh(7, 17, radius=2, length=5)
    glued, mapping = glue_rectilinear_seam(cut, 7, 17)
    assert_array_equal(glued.vertices[mapping], cut.vertices)
    assert_array_equal(glued.faces, mapping[cut.faces])
    assert glued.uv is None
    t = topology(glued)
    assert t.euler_characteristic == 0 and t.boundary_edges == 32
    assert t.manifold and t.oriented and not t.closed
    assert topology(cut).euler_characteristic == 1
    assert_allclose(surface_area(glued), surface_area(cut))
    assert_allclose(surface_area(cut), 16 * 2 * 2 * np.sin(np.pi / 16) * 5)
    errors = [abs(surface_area(cylinder_mesh(3, n + 1)) - 20 * np.pi) for n in (16, 32, 64)]
    assert errors[2] < errors[1] < errors[0]


def test_glue_does_not_silently_move_seam():
    cut = cylinder_mesh(3, 5)
    v = cut.vertices.copy()
    v[-1] += [0.01, 0, 0]
    with pytest.raises(ValueError, match="coincide"):
        glue_rectilinear_seam(Mesh(v, cut.faces, cut.uv), 3, 5)


@pytest.mark.parametrize("value", [1, 0, -1, 2.5, True])
def test_invalid_grid_sizes(value):
    with pytest.raises(ValueError):
        rectilinear_faces(value, 3)
