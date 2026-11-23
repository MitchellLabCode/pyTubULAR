"""Align three explicit timepoints to the previous geometric s-phi chart."""

import numpy as np

from tubular import Mesh, TubULAR
from tubular.grid import cylinder_mesh


def main():
    nu,nv = 9,33
    base = cylinder_mesh(nu,nv,length=8,radius=2,length_unit="um")
    pairs = np.column_stack((np.arange(nu),np.arange((nv-1)*nu,nu*nv)))
    tubi = TubULAR(time_points=[0,3,9],opts={"nU":nu,"nV":nv})
    reference = None
    for time,shift in [(0,0),(3,.125),(9,.25)]:
        # Same circular surface, with its chart origin rotated around the axis.
        angle = 2*np.pi*shift
        rotation = np.array([[1,0,0],[0,np.cos(angle),np.sin(angle)],
                            [0,-np.sin(angle),np.cos(angle)]])
        mesh = Mesh(base.v@rotation.T,base.f,base.uv,base.length_unit)
        tubi.register_mesh(time,mesh,kind="cutMesh"); tubi.setTime(time)
        result = tubi.generateCurrentSPCutMesh(pairs,reference=reference,register=True)
        assert np.array_equal(result.mesh.v[:nu],result.mesh.v[-nu:])
        if reference is not None:
            fit = result.phase_alignment.fit
            print(f"t={time}: phase offset {result.phi_offsets.mean():.6g} turns; "
                  f"mean residual {fit.rms_before.mean():.6g} -> {fit.rms_after.mean():.3g} um")
            assert np.max(abs(result.mesh.v-reference.mesh.v)) < 1e-12
        reference = result
    print("Geometric chart alignment does not distinguish physical rotation from a shifted chart origin.")


if __name__ == "__main__":
    main()
