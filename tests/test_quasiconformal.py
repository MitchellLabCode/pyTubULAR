import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.io import loadmat

from tubular import Mesh, TubULAR
from tubular.quasiconformal import beltrami_coefficient


def reference():
    uv = np.array([[0.,0], [1.3,0], [1.,1.2], [-.2,.9]])
    return Mesh(np.column_stack((uv, np.zeros(4))), np.array([[0,1,2],[0,2,3]]), uv, "um")


@pytest.mark.parametrize("dimension", [2,3])
def test_identity_and_anisotropic_stretch(dimension):
    mesh = reference()
    target = mesh.uv if dimension == 2 else mesh.v
    assert_allclose(beltrami_coefficient(mesh, target).mu, 0, atol=1e-15)
    stretched = target*np.array([2,1] if dimension==2 else [2,1,1])
    result = beltrami_coefficient(mesh, stretched)
    assert_allclose(result.mu, 1/3, atol=1e-15)
    assert_allclose(result.dilatation, 2, atol=1e-15)
    assert not result.mu.flags.writeable and not result.dilatation.flags.writeable


def test_shear_formula_and_metric_agree():
    mesh = reference()
    shear = .7
    target = mesh.uv @ np.array([[1.,0],[shear,1.]])
    result = beltrami_coefficient(mesh, target)
    expected = 1j*shear/(2-1j*shear)
    assert_allclose(result.mu, expected, atol=1e-15)
    embedded = np.column_stack((target, np.zeros(len(target))))
    assert_allclose(beltrami_coefficient(mesh, embedded).mu, result.mu, atol=1e-15)
    assert_allclose(result.dilatation, (1+abs(expected))/(1-abs(expected)), atol=1e-14)


def test_target_similarity_and_domain_rotation_law():
    mesh = reference()
    target = mesh.uv @ np.array([[2.,.3],[.5,1.4]])
    baseline = beltrami_coefficient(mesh, target)
    angle = .43
    r = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    for scale in (1e-6, 4, 1e6):
        moved = (target @ r.T+[2,-3])*scale
        assert_allclose(beltrami_coefficient(mesh, moved).mu, baseline.mu, atol=1e-15)
    changed_domain = Mesh(mesh.v, mesh.f, mesh.uv @ r.T)
    assert_allclose(beltrami_coefficient(changed_domain, target).mu,
                    baseline.mu*np.exp(2j*angle), atol=1e-15)
    embedded = np.column_stack((target, .2*target[:,0]-.3*target[:,1]))
    result = beltrami_coefficient(mesh, embedded)
    q, _ = np.linalg.qr(np.array([[1.,2,3],[-2.,1,1],[1.,3,-1]]))
    transformed = beltrami_coefficient(mesh, embedded @ q.T+[2,-3,5])
    assert_allclose(transformed.mu, result.mu, atol=1e-15)
    # A 3D metric cannot distinguish a mirror of the embedding.
    assert_allclose(beltrami_coefficient(mesh, embedded*[1,-1,1]).mu, result.mu, atol=1e-15)


def test_face_order_winding_and_local_values():
    mesh = reference()
    target = mesh.v.copy(); target[2,2] = .7
    result = beltrami_coefficient(mesh, target)
    reversed_mesh = Mesh(mesh.v, mesh.f[::-1, ::-1], mesh.uv)
    assert_allclose(beltrami_coefficient(reversed_mesh, target).mu, result.mu[::-1], atol=1e-15)
    assert_array_equal(mesh.v[:,2], 0)


@pytest.mark.parametrize("kind", ["reflection", "one_face_fold", "collapsed", "singular", "nan", "shape", "complex"])
def test_invalid_maps(kind):
    mesh = reference(); target = mesh.uv.copy()
    if kind == "reflection": target[:,1] *= -1
    elif kind == "one_face_fold": target[1] = [-1,1]
    elif kind == "collapsed": target[:,1] = 0
    elif kind == "singular": target[:,1] *= 1e-13
    elif kind == "nan": target[0,0] = np.nan
    elif kind == "shape": target = target[:-1]
    else: target = target.astype(complex)
    with pytest.raises(ValueError):
        beltrami_coefficient(mesh, target)
    with pytest.raises(ValueError, match="UV"):
        beltrami_coefficient(Mesh(mesh.v,mesh.f))


def test_class_series_requires_supplied_correspondence_preserves_state(tmp_path):
    mesh = reference()
    tubi = TubULAR(time_points=[2,5], opts={"timeInterval": 3, "meshDir": tmp_path},
                    mesh_sources={"rawMesh": {2:mesh}})
    tubi.setTime(2); cached = tubi.getCurrentRawMesh()
    positions = {2:mesh.v, 5:mesh.v*[2,1,1]}
    result = tubi.measureBeltramiCoefficient(mesh, positions)
    assert_array_equal(result["timepoints"], [2,5])
    assert_array_equal(result["times"], [0,9])
    assert_allclose(result["mu"], [[0,0],[1/3,1/3]], atol=1e-15)
    assert_allclose(result["reference_mu"], 0, atol=1e-15)
    charts = {t:v[:,:2] for t,v in positions.items()}
    planar = tubi.measureBeltramiCoefficientPullbackToPullback(mesh, charts)
    assert_allclose(planar["mu"], result["mu"], atol=1e-15)
    assert tubi.currentTime == 2 and tubi.getCurrentRawMesh() is cached
    assert not list(tmp_path.iterdir())
    with pytest.raises(KeyError, match="5"):
        tubi.measureBeltramiCoefficient(mesh, {2:mesh.v})
    with pytest.raises(TypeError, match="mapping"):
        tubi.measureBeltramiCoefficient(mesh, np.stack(list(positions.values())))
    with pytest.raises(ValueError, match="N, 2"):
        tubi.measureBeltramiCoefficientPullbackToPullback(mesh, positions)
    with pytest.raises(ValueError):
        tubi.measureBeltramiCoefficient(mesh, positions, time_points=[5,2])
    assert tubi.currentTime == 2 and tubi.getCurrentRawMesh() is cached


def test_nonconformal_reference_is_reported_not_silently_subtracted():
    mesh = reference()
    mesh = mesh.with_vertices(mesh.v*[2,1,1])
    result = TubULAR().measureBeltramiCoefficient(mesh, {0:mesh.v})
    assert_allclose(result["reference_mu"], 1/3, atol=1e-15)
    assert_allclose(result["mu"], 1/3, atol=1e-15)


def test_original_matlab_beltrami_reference():
    root = Path(__file__).parent / "data"
    path = root / "matlab_beltrami_reference.mat"
    provenance = json.loads((root / "beltrami_provenance.json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance["fixture_sha256"]
    data = loadmat(path,simplify_cells=True)
    mesh = Mesh(data["referenceXYZ"], data["faces"].astype(int)-1, data["uv"])
    for target, expected in ((data["target2D"],data["mu2D"]),(data["target3D"],data["mu3D"])):
        assert_allclose(beltrami_coefficient(mesh,target).mu, expected, atol=1e-12, rtol=1e-11)
    # The original pullback-to-pullback method uses a 3D metric on z=0 charts.
    assert_allclose(beltrami_coefficient(mesh,data["target2D"]).mu, data["muFlat3D"], atol=1e-12, rtol=1e-11)
