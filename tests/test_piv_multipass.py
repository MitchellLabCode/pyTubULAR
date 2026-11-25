import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.ndimage import gaussian_filter, map_coordinates
from tubular import TubULAR
from tubular.piv import piv_pair, piv_multipass, PIVFlag, PIVResult, _predictor


def image(size=192):
    return gaussian_filter(np.random.default_rng(741).random((size,size)),1,mode='wrap')


@pytest.mark.parametrize('correlation',['circular','linear'])
@pytest.mark.parametrize('shift',[(12,-8),(-11,9),(0,0)])
def test_large_shift_recovered_by_small_windows_and_dt(shift,correlation):
    a=image();b=np.roll(a,(shift[1],shift[0]),axis=(0,1))
    result=piv_multipass(a,b,window_sizes=(64,32,16),dt=.2,correlation=correlation)
    assert len(result.passes)==3
    final=result.final
    assert final.valid.sum()>150
    assert_allclose(final.dx[final.valid],shift[0],atol=1e-12)
    assert_allclose(final.dy[final.valid],shift[1],atol=1e-12)
    assert_allclose(final.u[final.valid],shift[0]/.2,atol=1e-11)
    first=piv_pair(a,b,window_size=64,dt=.2,correlation=correlation)
    assert_allclose(first.dx,result.passes[0].dx,equal_nan=True)
    if shift!=(0,0):
        single=piv_pair(a,b,window_size=16)
        assert np.median(np.abs(single.dx[single.valid]-shift[0]))>4
    assert np.all(np.isnan(final.dx[~final.valid]))
    assert np.any(final.flags==PIVFlag.PREDICTOR_UNAVAILABLE)
    for part in result.passes:
        for v in vars(part).values():
            if isinstance(v,np.ndarray): assert not v.flags.writeable


def test_fractional_affine_shear_against_known_particle_motion():
    a=image(); yy,xx=np.indices(a.shape)
    # b(x+8.35+.025*y,y-3.2)=a(x,y), using inverse analytic affine map.
    source_y=yy+3.2
    b=map_coordinates(a,(source_y,xx-8.35-.025*source_y),order=3,mode='constant')
    result=piv_multipass(a,b,window_sizes=(64,32)).final
    assert result.valid.sum()==81
    assert_allclose(result.dx[result.valid],8.35+.025*result.y[result.valid],atol=.15,rtol=0)
    assert_allclose(result.dy[result.valid],-3.2,atol=.15,rtol=0)


def test_no_filling_of_invalid_predictors_and_masked_source_or_destination():
    a=image(128);b=np.roll(a,10,axis=1)
    mask=np.zeros_like(a,dtype=bool);mask[64,65]=True
    out=piv_multipass(a,b,window_sizes=(32,16),mask=mask).final
    assert out.valid.any()
    assert np.any(out.flags==PIVFlag.MASKED)
    assert np.any(out.flags==PIVFlag.PREDICTOR_UNAVAILABLE)
    for i,j in np.argwhere(out.valid):
        r,c=int(out.y[i,j]-7.5),int(out.x[i,j]-7.5)
        assert not mask[r:r+16,c:c+16].any()
        sx=int(round(out.dx[i,j]));sy=int(round(out.dy[i,j]))
        assert not mask[r+sy:r+sy+16,c+sx:c+sx+16].any()
    flat=piv_multipass(np.ones((64,64)),np.ones((64,64)),window_sizes=(32,16))
    assert np.all(flat.passes[0].flags==PIVFlag.LOW_TEXTURE)
    assert np.all(flat.final.flags==PIVFlag.PREDICTOR_UNAVAILABLE)


def test_outside_shifted_image_is_rejected():
    a=image(192);b=np.roll(a,25,axis=1)
    out=piv_multipass(a,b,window_sizes=(64,32)).final
    assert np.any(out.flags==PIVFlag.OUTSIDE_IMAGE)
    assert np.isnan(out.dx[out.flags==PIVFlag.OUTSIDE_IMAGE]).all()


def test_bilinear_predictor_exact_affine_field_and_invalid_corner_policy():
    x,y=np.meshgrid([1.,3.,7.],[2.,4.])
    dx=2*x+3*y;dy=5*x-y;flags=np.zeros(x.shape,dtype=np.uint8)
    prev=PIVResult(x,y,dx,dy,flags,1,4,2,'circular')
    assert_allclose(_predictor(prev,4,3),(17,17))
    assert _predictor(prev,0,3) is None
    flags[0,0]=PIVFlag.LOW_TEXTURE;dx[0,0]=dy[0,0]=np.nan
    assert _predictor(prev,2,3) is None
    assert_allclose(_predictor(prev,3,2),(12,13))  # Zero-weight NaNs do not contaminate.
    assert_allclose(_predictor(prev,4,3),(17,17))


def test_class_dispatch_repeated_sizes_and_single_pass_unchanged():
    a=image(96);tubi=TubULAR()
    out=tubi.measurePIV2d(a,a,window_size=(32,32),step=(16,16))
    assert_allclose(out.final.dx,0,atol=1e-12)
    assert isinstance(tubi.measurePIV2d(a,a),PIVResult)
    assert_array_equal(out.passes[0].x,out.final.x)


@pytest.mark.parametrize('opts',[{'window_sizes':32},{'window_sizes':(32,)},
    {'window_sizes':(16,32)}, {'window_sizes':(33,16)}, {'window_sizes':(64,True)},
    {'window_sizes':(128,16)}, {'window_sizes':(32,16.5)}, {'steps':(16,)},
    {'steps':16}, {'steps':(16,0)}, {'dt':0}, {'correlation':'bad'},
    {'mask':np.zeros((64,64))}, {'mask':np.ma.array(np.zeros((64,64),dtype=bool))}])
def test_invalid_options(opts):
    args={'window_sizes':(32,16)};args.update(opts)
    with pytest.raises(ValueError): piv_multipass(image(64),image(64),**args)
