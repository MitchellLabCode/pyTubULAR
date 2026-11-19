import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.ndimage import gaussian_filter
from scipy.io import loadmat

from tubular import TubULAR
from tubular.piv import PIVFlag, correlate_windows, piv_pair


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('shift',[(3,-2),(-4,2),(0,0)])
def test_periodic_integer_translation_signs_centers_and_time_units(shift):
    a = gaussian_filter(np.random.default_rng(321).random((32,32)),1,mode='wrap')
    b = np.roll(a,(shift[1],shift[0]),axis=(0,1))
    result = piv_pair(a,b,window_size=32,dt=.25)
    assert result.valid.all()
    assert_allclose([result.dx.item(),result.dy.item()],shift,atol=1e-12)
    assert_allclose([result.u.item(),result.v.item()],np.array(shift)/.25,atol=1e-12)
    assert_array_equal(result.x,[[15.5]]); assert_array_equal(result.y,[[15.5]])
    inverse = piv_pair(b,a,window_size=32)
    assert_allclose(inverse.dx,-result.dx,atol=1e-12)
    assert_allclose(inverse.dy,-result.dy,atol=1e-12)
    for value in vars(result).values():
        if isinstance(value,np.ndarray):
            assert not value.flags.writeable


def test_known_fractional_particle_translation_and_matlab_kernels():
    provenance = json.loads((ROOT/'tests/data/piv_provenance.json').read_text())
    for path,key in [('tests/data/matlab_piv_reference.mat','fixture_sha256'),
                     ('scripts/generate_piv_reference.m','generator_sha256')]:
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == provenance[key]
    cases = loadmat(ROOT/'tests/data/matlab_piv_reference.mat',simplify_cells=True)['cases']
    for row in cases:
        for c in row:
            c = vars(c)
            mode = 'circular' if c['mode']==1 else 'linear'
            corr = correlate_windows(c['a'],c['b'],correlation=mode)
            assert_allclose(corr,c['correlation'],rtol=1e-12,atol=1e-10)
            result = piv_pair(c['a'],c['b'],correlation=mode)
            actual = np.array([result.dx.item(),result.dy.item()])
            assert_allclose(actual,c['displacement'],rtol=1e-12,atol=1e-12)
            if mode == 'circular':
                # Independent true displacement, not just agreement between ports.
                assert_allclose(actual,c['shift'],atol=.025,rtol=0)


def test_linear_correlation_matches_direct_sums():
    a,b = np.random.default_rng(52).normal(size=(2,8,8))
    a -= a.mean(); b -= b.mean()
    expected = np.empty((8,8))
    for row,dy in enumerate(range(-4,4)):
        for col,dx in enumerate(range(-4,4)):
            total = 0.
            for y in range(8):
                for x in range(8):
                    if 0 <= x+dx < 8 and 0 <= y+dy < 8:
                        total += a[y,x]*b[y+dy,x+dx]
            expected[row,col] = total
    assert_allclose(correlate_windows(a,b,correlation='linear'),expected,atol=1e-14)


def test_grid_mask_flat_windows_and_ambiguous_peaks_are_not_filled():
    a = np.random.default_rng(4).random((48,48))
    mask = np.zeros_like(a,dtype=bool); mask[0,0] = True
    result = piv_pair(a,a,window_size=16,step=16,mask=mask)
    assert result.dx.shape == (3,3)
    assert_array_equal(result.x[0],[7.5,23.5,39.5])
    assert result.flags[0,0] == PIVFlag.MASKED
    assert np.isnan(result.dx[0,0]) and result.valid.sum()==8
    assert_allclose(result.dx[result.valid],0,atol=1e-12)
    flat = piv_pair(np.ones((32,32)),np.ones((32,32)))
    assert flat.flags.item() == PIVFlag.LOW_TEXTURE
    assert np.isnan(flat.u).all()
    stripe = np.tile([0.,1.],(32,16))
    repeated = piv_pair(stripe,stripe)
    assert repeated.flags.item() == PIVFlag.AMBIGUOUS_PEAK
    assert np.isnan(repeated.dx).all()
    impulse = np.zeros((32,32)); impulse[8,8] = 1
    isolated = piv_pair(impulse,impulse)
    assert isolated.flags.item() == PIVFlag.INVALID_GAUSSIAN
    assert np.isnan(isolated.dx).all()


def test_border_peak_is_invalid_and_intensity_scaling_invariant():
    a = gaussian_filter(np.random.default_rng(44).random((32,32)),1,mode='wrap')
    result = piv_pair(a,np.roll(a,15,axis=1))
    assert result.flags.item() == PIVFlag.BORDER_PEAK
    b = np.roll(a,2,axis=0)
    plain = piv_pair(a,b)
    scaled = piv_pair(a*1e100,b*1e90)
    assert_allclose(scaled.dx,plain.dx,atol=1e-12)
    assert_allclose(scaled.dy,plain.dy,atol=1e-12)
    tubi = TubULAR(time_points=[0])
    assert_allclose(tubi.measurePIV2d(a,b).dy,plain.dy)
    assert tubi.method_status('measurePIV2d')['status']=='adapted'


@pytest.mark.parametrize('options',[{'window_size':31},{'window_size':64},{'step':0},
    {'dt':0},{'dt':True},{'dt':np.nan},{'correlation':'auto'},
    {'mask':np.zeros((32,32))},{'mask':np.zeros((2,2),dtype=bool)}])
def test_bad_settings_rejected(options):
    with pytest.raises(ValueError):
        piv_pair(np.ones((32,32)),np.ones((32,32)),**options)


def test_bad_images_and_unsupported_multipass_rejected():
    for a,b in [(np.ones((32,32)),np.ones((31,32))),
                (np.ones((32,32))*1j,np.ones((32,32))),
                (np.full((32,32),np.nan),np.ones((32,32))),
                (np.ma.array(np.ones((32,32))),np.ones((32,32)))]:
        with pytest.raises(ValueError):
            piv_pair(a,b)
    with pytest.raises(TypeError):
        TubULAR().measurePIV2d(np.ones((32,32)),np.ones((32,32)),numPasses=4)
