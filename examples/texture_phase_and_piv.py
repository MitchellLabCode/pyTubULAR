"""Texture-align a cylinder chart, then demonstrate large-displacement PIV."""
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from tubular import Mesh, TubULAR
from tubular.grid import cylinder_mesh


def main():
    nu,nv=9,65
    base=cylinder_mesh(nu,nv,length=8,radius=2,length_unit='um')
    pairs=np.column_stack((np.arange(nu),np.arange((nv-1)*nu,nu*nv)))
    tubi=TubULAR(time_points=[0,7],opts={'nU':nu,'nV':nv})
    tubi.register_mesh(0,base,kind='cutMesh');tubi.setTime(0)
    reference=tubi.generateCurrentSPCutMesh(pairs)
    angle=-np.pi/4
    rotation=np.array([[1,0,0],[0,np.cos(angle),np.sin(angle)],[0,-np.sin(angle),np.cos(angle)]])
    current=Mesh(base.v@rotation.T,base.f,base.uv,base.length_unit)
    tubi.register_mesh(7,current,kind='cutMesh');tubi.setTime(7)
    initial=tubi.generateCurrentSPCutMesh(pairs)
    texture=gaussian_filter(np.random.default_rng(429).normal(size=(64,49)),(1.4,0),mode='wrap')
    fit=tubi.fitPhiOffsetsViaTexture(np.roll(texture,8,axis=0),texture,strip_width=13)
    offsets=fit.phase_offsets(initial.ring_s/initial.ring_s[-1])
    aligned=tubi.generateCurrentSPCutMesh(pairs,phi_offsets=offsets,register=True)
    np.testing.assert_allclose(aligned.mesh.v,reference.mesh.v,atol=4e-14)
    np.testing.assert_allclose(aligned.interpolate(current.v),aligned.mesh.v,atol=4e-14)
    assert np.array_equal(aligned.mesh.v[:nu],aligned.mesh.v[-nu:])
    print(f'Texture alignment: {fit.shift_pixels.mean():.6g} rows; surface residual '
          f'{np.max(np.linalg.norm(aligned.mesh.v-reference.mesh.v,axis=1)):.3g} um')
    a=gaussian_filter(np.random.default_rng(741).random((192,192)),1,mode='wrap')
    b=np.roll(a,(-8,12),axis=(0,1))
    result=tubi.measurePIV2d(a,b,window_size=(64,32,16),dt=.5).final
    np.testing.assert_allclose(result.dx[result.valid],12,atol=1e-12)
    np.testing.assert_allclose(result.dy[result.valid],-8,atol=1e-12)
    print(f'PIV: {result.valid.sum()} valid vectors; displacement (12,-8) pixels, '
          'velocity (24,-16) pixels per supplied time unit')
    # Report the sampling limit on a prescribed shear, without claiming that
    # ever smaller interrogation windows improve accuracy.
    y,x=np.indices(a.shape);sy=y+3.2
    sheared=map_coordinates(a,(sy,x-8.35-.025*sy),order=3,mode='constant')
    for sizes in [(64,32),(64,32,16)]:
        r=tubi.measurePIV2d(a,sheared,window_size=sizes).final
        ex=np.max(abs(r.dx[r.valid]-8.35-.025*r.y[r.valid]))
        ey=np.max(abs(r.dy[r.valid]+3.2))
        print(f'Shear windows {sizes}: max component errors ({ex:.3f},{ey:.3f}) pixels')


if __name__ == '__main__':
    main()
