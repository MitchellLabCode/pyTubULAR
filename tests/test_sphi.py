import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.io import loadmat

from tubular import Mesh, TubULAR
from tubular.endcaps import remove_endcaps, shortest_seam
from tubular.grid import cylinder_mesh, glue_rectilinear_seam
from tubular.mesh import topology
from tubular.resampling import resample_annulus
from tubular.sphi import parameterize_sphi


ROOT = Path(__file__).resolve().parents[1]


def cylinder(nu=5,nv=9):
    m = cylinder_mesh(nu,nv,length=4,radius=2,length_unit="um")
    pairs = np.column_stack((np.arange(nu),np.arange((nv-1)*nu,nv*nu)))
    return m,pairs


def fixture():
    return loadmat(ROOT/'tests/data/matlab_sphi_reference.mat',simplify_cells=True)


def closed_cylinder():
    case = fixture()['caps']
    return Mesh(case['vertices'],case['faces'].astype(int)-1,length_unit='um')


@pytest.mark.parametrize('nu,nv',[(5,9),(8,17),(11,25)])
def test_cylinder_length_phase_exact_seam_and_fields(nu,nv):
    mesh,pairs = cylinder()
    result = parameterize_sphi(mesh,pairs,nu,nv)
    s,p = np.meshgrid(np.linspace(0,4,nu),np.linspace(0,1,nv))
    assert_allclose(result.sphi,np.column_stack((s.ravel(),p.ravel())),atol=2e-14)
    assert_allclose(result.mesh.v[:,0],s.ravel(),atol=2e-14)
    assert_allclose(result.hoop_centers[:,1:],0,atol=2e-14)
    assert_allclose(result.centerline_s,np.linspace(0,4,nu),atol=2e-14)
    assert_allclose(result.interpolate(mesh.v),result.mesh.v,atol=1e-14)
    assert_allclose(result.interpolate(np.ones(len(mesh.v))),1,atol=1e-14)
    assert_array_equal(result.mesh.v[:nu],result.mesh.v[-nu:])
    assert_allclose(result.phi_radians,2*np.pi*result.sphi[:,1])
    assert result.mesh.length_unit == 'um'
    glued = glue_rectilinear_seam(result.mesh,nu,nv)[0]
    assert topology(glued).euler_characteristic == 0
    for obj in (result,result.first_sampling,result.final_sampling):
        for value in vars(obj).values():
            if isinstance(value,np.ndarray):
                assert not value.flags.writeable


def test_known_phase_shift_and_winding_preservation():
    mesh,pairs = cylinder()
    plain = parameterize_sphi(mesh,pairs,5,9)
    shifted = parameterize_sphi(mesh,pairs,5,9,phi_offsets=np.full(5,.25))
    expected = np.roll(plain.mesh.v[:-5].reshape(8,5,3),-2,axis=0).reshape(-1,3)
    assert_allclose(shifted.mesh.v[:-5],expected,atol=1e-13)
    assert_array_equal(plain.mesh.f,shifted.mesh.f)
    reversed_result = parameterize_sphi(Mesh(mesh.v,mesh.f[:,[0,2,1]],mesh.uv),pairs[:,::-1],5,9)
    assert_array_equal(reversed_result.mesh.f,plain.mesh.f[:,[0,2,1]])
    assert_allclose(reversed_result.mesh.v,plain.mesh.v,atol=1e-13)


def test_rigid_motion_and_scale_invariance():
    mesh,pairs = cylinder()
    xyz = mesh.v.copy(); xyz[:,0] += .2*np.sin(mesh.uv[:,1])*np.sin(np.pi*mesh.uv[:,0]/4)
    xyz[-5:] = xyz[:5]
    source = Mesh(xyz,mesh.f,mesh.uv)
    rotation = np.array([[0,1,0],[0,0,1],[1,0,0.]])
    transformed = Mesh(3*xyz@rotation.T+[7,-2,9],mesh.f,mesh.uv)
    a,b = [parameterize_sphi(m,pairs,7,13) for m in (source,transformed)]
    assert_allclose(b.mesh.v,3*a.mesh.v@rotation.T+[7,-2,9],atol=1e-12)
    assert_allclose(b.sphi,a.sphi*[3,1],atol=1e-12)
    assert_allclose(b.hoop_radii,3*a.hoop_radii,atol=1e-12)
    assert_allclose(b.u_samples,a.u_samples,atol=1e-13)


def test_nonuniform_u_sampling_keeps_true_coordinates():
    mesh,pairs = cylinder()
    us = np.array([0,.03,.25,.7,1])
    grid = resample_annulus(mesh,pairs,5,9,u_samples=us)
    assert_allclose(grid.mesh.v[:,0],np.tile(4*us,9),atol=1e-14)
    assert_allclose(grid.mesh.uv[:,0],np.tile(us,9),atol=1e-14)
    for bad in ([0,.5,1],[0,.5,.5,.8,1],[.1,.3,.5,.8,1],[0,.3,.5,.8,1.1],np.ones(5)*np.nan):
        with pytest.raises(ValueError):
            resample_annulus(mesh,pairs,5,9,u_samples=bad)


@pytest.mark.parametrize('kw',[{'n_u':1},{'n_v':3},{'phi_offsets':[0,0]},
                            {'phi_offsets':np.full(7,np.inf)},{'seam_weight':'auto'}])
def test_invalid_sphi_options(kw):
    mesh,pairs = cylinder()
    options = dict(n_u=7,n_v=13); options.update(kw)
    with pytest.raises(ValueError):
        parameterize_sphi(mesh,pairs,**options)


def test_matlab_stage_references_and_provenance():
    provenance = json.loads((ROOT/'tests/data/sphi_provenance.json').read_text())
    for path,key in [('tests/data/matlab_sphi_reference.mat','fixture_sha256'),
                     ('scripts/generate_sphi_reference.m','generator_sha256')]:
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == provenance[key]
    differences = []
    for row in fixture()['cases']:
        row = [vars(c) for c in row]
        for c in row:
            mesh = Mesh(c['vertices'],c['faces'].astype(int)-1,c['chart'])
            result = parameterize_sphi(mesh,c['pathPairs'].astype(int)-1,int(c['nU']),int(c['nV']),
                                       phi_offsets=c['offsets'],seam_weight='unique' if c['mode']==1 else 'matlab')
            for actual,key in [(result.crude_ring_s,'crudeS'),(result.u_samples,'uSamples'),
                               (result.first_sampling.mesh.v,'firstXYZ'),(result.uniform_hoops,'hoops'),
                               (result.hoop_centers,'centers'),(result.centerline_s,'centerS'),
                               (result.hoop_radii,'radii'),(result.ring_s,'ringS'),
                               (result.sphi,'sphi'),(result.mesh.v,'finalXYZ')]:
                assert_allclose(actual,c[key],atol=2e-11,rtol=1e-11,err_msg=key)
        differences.append(row[1]['ringS'])
    assert np.max(abs(differences[0]-differences[1])) > 1e-5


def test_explicit_caps_match_matlab_and_preserve_source_correspondence():
    mesh = closed_cylinder()
    result = remove_endcaps(mesh,[40],[41])
    c = fixture()['caps']
    assert_array_equal(result.mesh.v,c['openXYZ'])
    assert_array_equal(result.mesh.f,c['openFaces']-1)
    assert_array_equal(result.vertex_sources,c['vertexSources']-1)
    assert_array_equal(result.source_to_open,c['sourceToOpen'].astype(int)-1)
    assert_array_equal(result.mesh.f,result.source_to_open[mesh.f[result.face_sources]])
    assert len(result.boundary_loops) == 2
    assert_array_equal(result.vertex_sources[result.boundary_loops[0]]%5,0)
    assert_array_equal(result.vertex_sources[result.boundary_loops[1]]%5,4)
    assert result.mesh.uv is None


def test_shortest_seam_is_interior_and_has_analytic_length():
    opened = remove_endcaps(closed_cylinder(),[40],[41])
    path = shortest_seam(opened.mesh,[0,4])
    assert_array_equal(path,np.arange(5))
    assert_allclose(np.linalg.norm(np.diff(opened.mesh.v[path],axis=0),axis=1).sum(),3)
    assert_array_equal(shortest_seam(opened.mesh,[4,0]),path[::-1])
    for ends in ([0,5],[0,1],[0,4,9],[-1,4],[0.,4.]):
        with pytest.raises(ValueError):
            shortest_seam(opened.mesh,ends)


@pytest.mark.parametrize('a,p',[([], [41]),([40],[40]),([40,40],[41]),
                              ([40,2],[41]),([40.],[41]),([40],[100])])
def test_bad_caps_rejected(a,p):
    with pytest.raises(ValueError):
        remove_endcaps(closed_cylinder(),a,p)


def test_class_segmented_closed_mesh_to_sphi_and_atomic_registration():
    tubi = TubULAR(time_points=[0],opts={'nU':7,'nV':13})
    tubi.setTime(0); tubi.register_mesh(0,closed_cylinder())
    opened = tubi.sliceMeshEndcaps([40],[41],register=True)
    cut = tubi.generateCurrentCutMesh(seam_endpoints=[0,4],register=True)
    result = tubi.generateCurrentSPCutMesh(cut.path_pairs,register=True)
    assert tubi.getCurrentSPCutMesh() is result.mesh
    assert_allclose(result.sphi[:,0].max(),3,atol=1e-12)
    source_field = closed_cylinder().v[:,0]
    cut_field = source_field[opened.vertex_sources][cut.cut.cut_to_uncut]
    assert_allclose(result.interpolate(cut_field),result.mesh.v[:,0],atol=1e-12)
    with pytest.raises(ValueError):
        tubi.generateCurrentSPCutMesh(cut.path_pairs,n_v=2,register=True)
    assert tubi.getCurrentSPCutMesh() is result.mesh
    with pytest.raises(ValueError):
        tubi.sliceMeshEndcaps([40],[40],register=True)
    assert tubi.getCurrentCylinderMeshClean() is opened.mesh
    with pytest.raises(ValueError):
        tubi.generateCurrentCutMesh([0,1,2,3,4],seam_endpoints=[0,4],register=True)
    assert tubi.getCurrentCutMesh() is cut.mesh
