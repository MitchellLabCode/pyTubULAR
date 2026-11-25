import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.ndimage import gaussian_filter, fourier_shift

from tubular import TubULAR
from tubular.texture_phase import fit_texture_phase, TexturePhaseFlag


def image(shape=(64,49)):
    return gaussian_filter(np.random.default_rng(429).normal(size=shape),(1.4,0),mode='wrap')


@pytest.mark.parametrize('shape',[(64,49),(63,18),(32,79)])
@pytest.mark.parametrize('shift',[-15,0,12])
def test_integer_shift_sign_rectangular_dimensions_gain_offsets_and_readonly(shape,shift):
    ref = image(shape)
    current = np.roll(ref,shift,axis=0)*17 + np.linspace(-2,3,shape[1])
    fit = fit_texture_phase(current,ref,strip_width=13,step=4)
    assert fit.valid.all()
    assert_allclose(fit.shift_pixels,shift,atol=2e-13)
    assert_allclose(fit.peak_correlation,1,atol=2e-15)
    assert_allclose(fit.phase_offsets([0,.25,1]),shift/shape[0],atol=1e-14)
    for v in vars(fit).values():
        if isinstance(v,np.ndarray): assert not v.flags.writeable


@pytest.mark.parametrize('shift',[-31.75,-.35,.2,31.8])
def test_fractional_fourier_shift_and_periodic_peak_neighbors(shift):
    ref = image()
    current = np.fft.ifftn(fourier_shift(np.fft.fftn(ref),(shift,0))).real
    fit = fit_texture_phase(current,ref,strip_width=17)
    assert fit.valid.all()
    errors = (fit.shift_pixels-shift+32)%64-32
    assert_allclose(errors,0,atol=.025,rtol=0)
    integer = fit_texture_phase(current,ref,strip_width=17,method='integer')
    assert_allclose((integer.shift_pixels-shift+32)%64-32,round(shift)-shift,atol=1e-12)


def test_variable_phase_unwraps_across_half_turn_without_smoothing():
    ref = image((64,30))
    shifts = np.arange(20,50)
    cur = np.column_stack([np.roll(ref[:,i],s) for i,s in enumerate(shifts)])
    fit = fit_texture_phase(cur,ref,strip_width=1,step=1)
    assert fit.valid.all()
    assert_allclose(fit.shift_pixels,shifts,atol=1e-12)
    assert fit.raw_offsets.min()<0 and fit.raw_offsets.max()>0


def test_direct_correlation_oracle():
    ref = image((16,7)); cur = np.roll(ref,3,axis=0)
    fit = fit_texture_phase(cur,ref,strip_width=7,step=3,method='integer')
    for i,x in enumerate([0,3,6]):
        c,r=cur[:,max(0,x-3):min(7,x+4)],ref[:,max(0,x-3):min(7,x+4)]
        c=c-c.mean(axis=0);r=r-r.mean(axis=0)
        direct=np.array([np.sum(np.roll(c,-k,axis=0)*r) for k in range(16)])/(np.linalg.norm(c)*np.linalg.norm(r))
        assert_allclose(fit.peak_correlation[i],direct.max(),atol=1e-14)
        assert_allclose(fit.raw_offsets[i]*16,direct.argmax(),atol=1e-14)


def test_rejected_strips_masks_repeated_patterns_flat_and_bounds():
    ref = image()
    for current,reference,opts,flag in [
        (ref,ref,{'mask':np.ones_like(ref,dtype=bool)},TexturePhaseFlag.MASKED),
        (ref,np.ones_like(ref),{},TexturePhaseFlag.LOW_TEXTURE),
        (np.tile(ref[:16],(4,1)),np.tile(ref[:16],(4,1)),{},TexturePhaseFlag.AMBIGUOUS_PEAK),
        (np.roll(ref,8,axis=0),ref,{'bounds':(-.125,.125)},TexturePhaseFlag.AT_BOUND)]:
        result = fit_texture_phase(current,reference,strip_width=17,**opts)
        assert_array_equal(result.flags,flag)
        assert np.isnan(result.offsets).all()
        with pytest.raises(ValueError,match='all texture strips'): result.phase_offsets([0,1])
    mask=np.zeros_like(ref,dtype=bool);mask[10,24]=True
    result=fit_texture_phase(ref,ref,strip_width=1,step=1,mask=mask)
    assert result.flags[24]==TexturePhaseFlag.MASKED and result.valid.sum()==48
    assert_allclose(result.offsets[result.valid],0,atol=1e-14)


@pytest.mark.parametrize('opts',[{'strip_width':0},{'strip_width':50},{'strip_width':True},
    {'step':0},{'step':1.5},{'method':'guess'},{'bounds':(-1,1)}, {'bounds':(.1,.1)},
    {'bounds':(.001,.002)}, {'bounds':(np.nan,.2)}, {'mask':np.zeros((64,49))},
    {'mask':np.zeros((2,2),dtype=bool)}, {'mask':np.ma.array(np.zeros((64,49),dtype=bool))}])
def test_invalid_options(opts):
    with pytest.raises(ValueError): fit_texture_phase(image(),image(),**opts)


def test_class_method_and_offset_interpolation_validation():
    tubi=TubULAR(time_points=[3,9]); tubi.setTime(9)
    fit=tubi.fitPhiOffsetsViaTexture(np.roll(image(),5,axis=0),image(),strip_width=13)
    assert tubi.method_status('fitPhiOffsetsViaTexture')['status']=='adapted'
    assert_allclose(fit.phase_offsets(np.linspace(0,1,11)),5/64,atol=1e-14)
    for bad in [[],[-.1],[1.1],[np.nan],[[0,1]],np.ma.array([0,1])]:
        with pytest.raises(ValueError): fit.phase_offsets(bad)
    for bad in [np.ma.array(image()),np.full((64,49),np.inf),image().astype(complex)]:
        with pytest.raises(ValueError): fit_texture_phase(bad,image())


def test_texture_fit_rebuilds_sphi_geometry_and_transfers_source_fields():
    from tubular import Mesh
    from tubular.grid import cylinder_mesh
    nu,nv=9,65
    base=cylinder_mesh(nu,nv,length=8,radius=2,length_unit='um')
    pairs=np.column_stack((np.arange(nu),np.arange((nv-1)*nu,nu*nv)))
    tubi=TubULAR(time_points=[0,7],opts={'nU':nu,'nV':nv})
    tubi.register_mesh(0,base,kind='cutMesh');tubi.setTime(0)
    reference=tubi.generateCurrentSPCutMesh(pairs)
    # Change the chart origin by -8 rows. Texture samples then shift +8 rows.
    angle=-2*np.pi*8/64
    rotation=np.array([[1,0,0],[0,np.cos(angle),np.sin(angle)],[0,-np.sin(angle),np.cos(angle)]])
    current=Mesh(base.v@rotation.T,base.f,base.uv,base.length_unit)
    tubi.register_mesh(7,current,kind='cutMesh');tubi.setTime(7)
    unaligned=tubi.generateCurrentSPCutMesh(pairs)
    texture=image((64,49))
    fit=tubi.fitPhiOffsetsViaTexture(np.roll(texture,8,axis=0),texture,strip_width=13)
    offsets=fit.phase_offsets(unaligned.ring_s/unaligned.ring_s[-1])
    aligned=tubi.generateCurrentSPCutMesh(pairs,phi_offsets=offsets,register=True)
    assert_allclose(aligned.mesh.v,reference.mesh.v,atol=4e-14)
    assert_array_equal(aligned.mesh.v[:nu],aligned.mesh.v[-nu:])
    assert_allclose(aligned.interpolate(current.v),aligned.mesh.v,atol=4e-14)
    assert_allclose(aligned.phi_offsets,8/64,atol=2e-14)


def test_original_matlab_integer_phase_contract_and_rectangular_bug():
    import hashlib
    import json
    from pathlib import Path
    from scipy.io import loadmat
    root=Path(__file__).resolve().parents[1]
    provenance=json.loads((root/'tests/data/texture_phase_provenance.json').read_text())
    for file,key in [('scripts/generate_texture_phase_reference.m','generator_sha256'),
                     ('tests/data/matlab_texture_phase_reference.mat','fixture_sha256')]:
        assert hashlib.sha256((root/file).read_bytes()).hexdigest()==provenance[key]
    cases=loadmat(root/'tests/data/matlab_texture_phase_reference.mat',simplify_cells=True)['cases']
    matches=defects=0
    for row in cases:
        for item in row:
            c=vars(item)
            fit=fit_texture_phase(c['current'],c['reference'],strip_width=13,method='integer')
            assert fit.valid.all()
            assert_allclose(fit.shift_pixels,c['expected'],atol=1e-12)
            assert c['original_dx']==0
            if c['original_dy']==c['expected']:
                matches+=1
                assert_allclose(fit.shift_pixels,c['original_dy'],atol=1e-12)
            else:
                defects+=1
                assert c['current'].shape==(32,79)
                assert c['expected']==-9 and c['original_dy']==23
    assert (matches,defects)==(11,1)
