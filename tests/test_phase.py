import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.io import loadmat

from tubular import Mesh, TubULAR
from tubular.grid import cylinder_mesh
from tubular.phase import fit_hoop_phase
from tubular.sphi import parameterize_sphi, fit_sphi_phase


ROOT = Path(__file__).resolve().parents[1]


def hoops(n=5,m=32,turns=1):
    theta = np.arange(m+1)*2*np.pi*turns/m
    result = np.stack([np.column_stack((np.full(m+1,i),np.cos(theta),np.sin(theta)))
                       for i in range(n)])
    result[:,-1] = result[:,0]
    return result


def shifted(h,offsets):
    """Generate known points on a regular polygon from its analytic edge formula."""
    m = h.shape[1]-1
    result = []
    for curve,offset in zip(h,offsets):
        x = np.arange(m)+offset*m
        k = np.floor(x).astype(int)
        f = x-k
        r = curve[k%m]*(1-f[:,None])+curve[(k+1)%m]*f[:,None]
        result.append(np.vstack((r,r[0])))
    return np.asarray(result)


@pytest.mark.parametrize('offset',[-.5,-.499999,-.23,0.,.125,.499999,.5])
def test_known_polygon_phase_sign_global_minimum_and_seam(offset):
    current = hoops()
    target = shifted(current,np.full(len(current),offset))
    fit = fit_hoop_phase(current,target)
    wrapped_error = (fit.raw_offsets-offset+.5)%1-.5
    assert_allclose(wrapped_error,0,atol=2e-14)
    assert_allclose(fit.aligned_hoops,target,atol=2e-14)
    assert_allclose(fit.rms_after,0,atol=2e-14)
    assert not fit.ambiguous.any()
    assert not fit.at_bounds.any()
    assert_array_equal(fit.aligned_hoops[:,0],fit.aligned_hoops[:,-1])
    for value in vars(fit).values():
        if isinstance(value,np.ndarray):
            assert not value.flags.writeable


def test_branch_crossing_unwraps_along_axis_without_a_full_turn_jump():
    current = hoops(7)
    offsets = np.linspace(.47,.53,7)
    target = shifted(current,offsets)
    fit = fit_hoop_phase(current,target)
    assert_allclose(fit.offsets,offsets,atol=2e-14)
    assert np.all(fit.raw_offsets >= -.5) and np.all(fit.raw_offsets <= .5)
    assert np.any(np.abs(np.diff(fit.raw_offsets)) > .9)
    assert_allclose(np.diff(fit.offsets),.01,atol=2e-14)
    assert_allclose(fit.rms_after,0,atol=2e-14)


def test_translation_rotation_and_scale_do_not_change_phase():
    current = hoops(3)
    target = shifted(current,[.173,-.111,.04])
    baseline = fit_hoop_phase(current,target)
    rotation = np.array([[0,1,0],[0,0,1],[1,0,0]])
    moved = fit_hoop_phase(7*current@rotation.T+[5,-8,3],7*target@rotation.T+[5,-8,3])
    assert_allclose(moved.offsets,baseline.offsets,atol=3e-14)
    # With unique periodic samples, translation between frames adds only a
    # constant to the objective; residuals still report the actual displacement.
    translated = fit_hoop_phase(current,target+[2,-3,4])
    assert_allclose(translated.offsets,baseline.offsets,atol=3e-14)
    assert_allclose(translated.rms_after,np.sqrt(29),atol=3e-14)


def test_duplicate_seam_weight_is_an_explicit_matlab_option():
    current = hoops(1,16)
    target = shifted(current,[.12]); target[:,0] += [0,.2,-.1]; target[:,-1]=target[:,0]
    unique = fit_hoop_phase(current,target)
    legacy = fit_hoop_phase(current,target,seam_weight='matlab')
    assert abs(unique.offsets[0]-legacy.offsets[0]) > 1e-4
    for fit,weights in [(unique,np.ones(16)),(legacy,np.r_[2.,np.ones(15)])]:
        expected = np.sqrt(np.average(fit.residuals**2,weights=weights,axis=1))
        assert_allclose(fit.rms_after,expected,atol=1e-14)


def test_multiple_minima_and_search_limited_fits_are_reported():
    doubled = hoops(2,32,turns=2)
    fit = fit_hoop_phase(doubled,doubled)
    assert fit.ambiguous.all()
    assert_allclose(fit.rms_after,0,atol=1e-14)
    current = hoops(2)
    target = shifted(current,[.2,-.2])
    limited = fit_hoop_phase(current,target,bounds=(-.1,.1))
    assert limited.at_bounds.all()
    assert_allclose(limited.raw_offsets,[.1,-.1],atol=1e-14)
    assert (limited.rms_after > .5).all()


def test_global_solution_beats_dense_independent_search_on_irregular_hoops():
    rng = np.random.default_rng(812)
    current = rng.normal(size=(2,12,3))
    current = np.concatenate((current,current[:,:1]),axis=1)
    target = rng.normal(size=(2,12,3))
    target = np.concatenate((target,target[:,:1]),axis=1)
    fit = fit_hoop_phase(current,target,bounds=(-.31,.47))
    grid = np.linspace(-.31,.47,10001)
    for i in range(2):
        phase = (np.arange(12)[None,:]/12+grid[:,None])%1
        # Independent scalar interpolation using NumPy, no fitting helper.
        candidate = np.stack([np.interp(phase,np.linspace(0,1,13),current[i,:,j])
                              for j in range(3)],axis=2)
        loss = np.mean(np.sum((candidate-target[i,:-1])**2,axis=2),axis=1)
        assert fit.rms_after[i]**2 <= loss.min()+1e-12
        assert abs(fit.rms_after[i]**2-loss.min()) < 1e-5


@pytest.mark.parametrize('bounds',[(-1.1,0),(0,1.1),(-.6,.6),(0,0),(.2,-.2),[0],(0,np.nan)])
def test_invalid_bounds(bounds):
    with pytest.raises(ValueError):
        fit_hoop_phase(hoops(),hoops(),bounds=bounds)


def test_invalid_hoop_arrays():
    good = hoops()
    unclosed = good.copy(); unclosed[0,-1,1] += .1
    repeated = good.copy(); repeated[0,1]=repeated[0,0]
    for bad in (good[0],good[:,:,:2],good[:,:3],unclosed,repeated,good+1j,
                np.full_like(good,np.nan),good[:0],np.ma.array(good)):
        with pytest.raises(ValueError):
            fit_hoop_phase(bad,good)
    with pytest.raises(ValueError):
        fit_hoop_phase(good,good[:2])
    with pytest.raises(ValueError):
        fit_hoop_phase(good,good,seam_weight='auto')


def mesh_pair(nu=5,nv=17,shift=.125):
    m = cylinder_mesh(nu,nv,length=4,radius=2,length_unit='um')
    angle = shift*2*np.pi
    rotation = np.array([[1,0,0],[0,np.cos(angle),np.sin(angle)],
                        [0,-np.sin(angle),np.cos(angle)]])
    current = Mesh(m.v@rotation.T,m.f,m.uv,m.length_unit)
    pairs = np.column_stack((np.arange(nu),np.arange((nv-1)*nu,nv*nu)))
    return m,current,pairs


def test_two_timepoint_class_alignment_preserves_fields_and_state():
    reference_mesh,current,pairs = mesh_pair()
    t = TubULAR(time_points=[0,7],opts={'nU':5,'nV':17})
    t.register_mesh(0,reference_mesh,kind='cutMesh');t.setTime(0)
    reference = t.generateCurrentSPCutMesh(pairs,register=True)
    t.register_mesh(7,current,kind='cutMesh');t.setTime(7)
    aligned = t.generateCurrentSPCutMesh(pairs,reference=reference,register=True)
    assert t.currentTime==7
    assert t.getCurrentSPCutMesh() is aligned.mesh
    assert_allclose(aligned.phi_offsets,-.125,atol=1e-13)
    assert_allclose(aligned.mesh.v,reference.mesh.v,atol=1e-13)
    assert_allclose(aligned.interpolate(current.v),aligned.mesh.v,atol=1e-13)
    fit = aligned.phase_alignment.fit
    assert_allclose(fit.rms_after,0,atol=1e-13)
    assert (fit.rms_before > 1).all()
    assert_array_equal(aligned.mesh.v[:5],aligned.mesh.v[-5:])
    assert_array_equal(aligned.mesh.f,reference.mesh.f)
    reference_map = aligned.phase_alignment.reference_sampling
    assert_allclose(reference_map.interpolate(reference.mesh.v),reference_map.mesh.v,atol=1e-14)
    with pytest.raises(ValueError,match='phase_bounds'):
        t.generateCurrentSPCutMesh(pairs,reference=reference,phase_bounds=(-.05,.05),register=True)
    assert t.getCurrentSPCutMesh() is aligned.mesh
    inspected = fit_sphi_phase(aligned,reference,bounds=(-.05,.05))
    assert inspected.fit.at_bounds.all()


def test_reference_resolution_axial_correspondence_units_and_invalid_options():
    mesh,current,pairs = mesh_pair()
    reference = parameterize_sphi(mesh,pairs,6,33)
    rs = np.array([0,.13,.4,.8,1])
    result = parameterize_sphi(current,pairs,5,17,reference=reference,reference_s=rs)
    assert_array_equal(result.phase_alignment.reference_s,rs)
    assert_allclose(result.phase_alignment.reference_sampling.mesh.v[:,0],np.tile(4*rs,17),atol=1e-13)
    assert_allclose(result.phi_offsets,-.125,atol=1e-13)
    assert result.phase_alignment.fit.rms_after[1] > .1 # axial mismatch is reported
    for opts in [{'reference_s':rs},{'phase_bounds':(-.1,.1)},
                 {'reference':reference,'phi_offsets':np.zeros(5)},
                 {'reference':reference,'reference_s':[0,.5,1]},
                 {'reference':reference,'reference_s':[0,.4,.3,.8,1]}]:
        with pytest.raises(ValueError):
            parameterize_sphi(current,pairs,5,17,**opts)
    with pytest.raises(TypeError):
        parameterize_sphi(current,pairs,5,17,reference=mesh)
    with pytest.raises(ValueError,match='units'):
        parameterize_sphi(Mesh(current.v,current.f,current.uv,'mm'),pairs,5,17,reference=reference)


def test_noncommensurate_circle_resampling_exposes_competing_minima():
    mesh,current,pairs = mesh_pair()
    reference = parameterize_sphi(mesh,pairs,6,25)
    provisional = parameterize_sphi(current,pairs,5,17)
    diagnostic = fit_sphi_phase(provisional,reference)
    # The 24-segment reference omits alternate vertices of the 16-segment
    # source polygon. Its altered shape admits two equally good small shifts
    # around the expected offset. Report this instead of selecting arbitrarily.
    assert diagnostic.fit.ambiguous.any()
    assert np.any(diagnostic.fit.rms_after > 0)
    with pytest.raises(ValueError,match='ambiguous'):
        parameterize_sphi(current,pairs,5,17,reference=reference)


def test_ambiguous_reference_alignment_does_not_register():
    mesh,_,pairs = mesh_pair(nv=33)
    uv = mesh.uv
    xyz = np.column_stack((mesh.v[:,0],2*np.cos(2*uv[:,1]),-2*np.sin(2*uv[:,1])))
    xyz[pairs[:,1]]=xyz[pairs[:,0]]
    doubled = Mesh(xyz,mesh.f,uv,'um')
    t = TubULAR(time_points=[0],opts={'nU':5,'nV':33})
    t.register_mesh(0,doubled,kind='cutMesh');t.setTime(0)
    reference = t.generateCurrentSPCutMesh(pairs,register=True)
    inspected = fit_sphi_phase(reference,reference)
    assert inspected.fit.ambiguous.all()
    with pytest.raises(ValueError,match='ambiguous'):
        t.generateCurrentSPCutMesh(pairs,reference=reference,register=True)
    assert t.getCurrentSPCutMesh() is reference.mesh


def test_original_matlab_phase_fit_and_provenance():
    provenance = json.loads((ROOT/'tests/data/phase_provenance.json').read_text())
    for name,key in [('tests/data/matlab_phase_reference.mat','fixture_sha256'),
                     ('scripts/generate_phase_reference.m','generator_sha256')]:
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==provenance[key]
    cases = loadmat(ROOT/'tests/data/matlab_phase_reference.mat',simplify_cells=True)['cases']
    for c in cases:
        result = fit_hoop_phase(c['currentHoops'],c['targetHoops'],bounds=c['bounds'],seam_weight='matlab')
        assert_allclose(result.raw_offsets,c['legacyOffsets'],atol=1e-8,rtol=0)
        assert_allclose(result.rms_after**2*17,c['losses'],atol=1e-12,rtol=1e-12)
        assert not result.ambiguous.any() and not result.at_bounds.any()
