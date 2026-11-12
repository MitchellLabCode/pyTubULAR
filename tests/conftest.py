import numpy as np
import pytest

from tubular import Mesh


@pytest.fixture
def plane():
    uv = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    v = np.column_stack((2 * uv[:, 0] + uv[:, 1], 3 * uv[:, 1], uv[:, 0]))
    return Mesh(v, np.array([[0, 1, 2], [0, 2, 3]]), uv, "um")


@pytest.fixture
def tetrahedron():
    return Mesh(np.array([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]]),
                np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]), length_unit="um")
