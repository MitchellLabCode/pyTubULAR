import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.io import loadmat

from tubular import Mesh, TubULAR
from tubular.geometry import face_normals
from tubular.grid import cylinder_mesh, glue_rectilinear_seam, rectilinear_faces
from tubular.mesh import topology
from tubular.resampling import resample_annulus, tile_annular_mesh


def cylinder(n_u=5,n_v=9):
    mesh = cylinder_mesh(n_u,n_v,radius=2,length=4,length_unit="um")
    pairs = np.column_stack((np.arange((n_v-1)*n_u,n_v*n_u),np.arange(n_u)))
    return mesh,pairs


@pytest.mark.parametrize("counts", [(0,0),(1,1),(2,0),(0,2)])
@pytest.mark.parametrize("reverse", [False,True])
def test_tiling_connectivity_correspondence_and_period(counts,reverse):
    mesh,pairs = cylinder()
    if reverse:
        pairs = pairs[:,::-1]
    tiled = tile_annular_mesh(mesh,pairs,counts)
    count = 1+sum(counts)
    assert len(tiled.mesh.f) == count*len(mesh.f)
    assert len(tiled.mesh.v) == count*len(mesh.v)-(count-1)*len(pairs)
    topo = topology(tiled.mesh)
    assert topo.manifold and topo.oriented and topo.components == 1
    assert topo.euler_characteristic == 1
    assert_array_equal(tiled.mesh.v,mesh.v[tiled.vertex_sources])
    expected = mesh.u[tiled.vertex_sources].copy()
    expected[:,1] += tiled.vertex_v_shifts
    assert_allclose(tiled.mesh.u,expected,atol=1e-13)
    canonical = np.arange(len(mesh.v)); canonical[pairs[:,1]] = pairs[:,0]
    assert_array_equal(canonical[tiled.vertex_sources[tiled.mesh.f]],
                       canonical[mesh.f[tiled.face_sources]])
    assert tiled.mesh.length_unit == "um"
    for value in vars(tiled).values():
        if isinstance(value,np.ndarray):
            assert not value.flags.writeable


@pytest.mark.parametrize("method", ["mesh","matlab_scattered"])
def test_polygonal_cylinder_sampling_fields_normals_and_gluing(method):
    mesh,pairs = cylinder()
    normals = np.column_stack((np.zeros(len(mesh.v)),mesh.v[:,1:]/2))
    result = resample_annulus(mesh,pairs,11,17,normals=normals,interpolation=method)
    u,v = np.meshgrid(np.linspace(0,1,11),np.linspace(0,1,17))
    assert_allclose(result.mesh.u,np.column_stack((u.ravel(),v.ravel())),atol=1e-14)
    # Piecewise-linear polygonal circumference, not a fitted smooth cylinder.
    t = 8*v.ravel(); k = np.minimum(np.floor(t).astype(int),7); t -= k
    theta = np.arange(9)*np.pi/4
    y = 2*((1-t)*np.cos(theta[k])+t*np.cos(theta[k+1]))
    z = -2*((1-t)*np.sin(theta[k])+t*np.sin(theta[k+1]))
    expected = np.column_stack((4*u.ravel(),y,z))
    assert_allclose(result.mesh.v,expected,atol=1e-12)
    assert_array_equal(result.mesh.v[:11],result.mesh.v[-11:])
    assert_array_equal(result.normals[:11],result.normals[-11:])
    assert_allclose(np.linalg.norm(result.normals,axis=1),1,atol=1e-14)
    analytic_normals = np.column_stack((np.zeros(len(y)),y,z))
    analytic_normals /= np.linalg.norm(analytic_normals,axis=1)[:,None]
    assert_allclose(result.normals,analytic_normals,atol=1e-12)
    assert_allclose(result.interpolate(mesh.v),result.mesh.v,atol=1e-14)
    assert_allclose(result.interpolate(np.ones(len(mesh.v))),1,atol=1e-14)
    assert_array_equal(result.path_pairs,np.column_stack((np.arange(11),np.arange(176,187))))
    glued,_ = glue_rectilinear_seam(result.mesh,11,17,tolerance=0)
    assert topology(glued).euler_characteristic == 0
    assert topology(glued).boundary_edges == 32
    corner_uv = mesh.u[result.source_vertices].copy()
    corner_uv[:,:,1] += result.corner_v_shifts
    assert_allclose(np.sum(result.barycentric[...,None]*corner_uv,axis=1),
                    result.source_query_uv,atol=1e-12)
    assert_allclose(result.barycentric.sum(axis=1),1,atol=1e-14)
    if method == "mesh":
        assert_array_equal(result.source_vertices,mesh.f[result.source_faces])
    else:
        assert result.source_faces is None
    for value in vars(result).values():
        if isinstance(value,np.ndarray):
            assert not value.flags.writeable


def test_curved_seam_multiple_periods_and_negative_uv_orientation():
    mesh,pairs = cylinder(9,17)
    chart = mesh.u.copy()
    chart[:,1] += 2.8*2*np.pi*np.sin(np.pi*chart[:,0]/4)
    curved = Mesh(mesh.v,mesh.f,chart,mesh.length_unit)
    result = resample_annulus(curved,pairs,13,25)
    # At least one sample must use a tile more than one period from the base.
    assert np.max(np.abs(result.corner_v_shifts)) > 2*np.pi+1e-10
    assert_array_equal(result.mesh.v[:13],result.mesh.v[-13:])
    corner_uv = curved.u[result.source_vertices].copy()
    corner_uv[:,:,1] += result.corner_v_shifts
    assert_allclose(np.sum(result.barycentric[...,None]*corner_uv,axis=1),
                    result.source_query_uv,atol=1e-12)
    reversed_mesh = Mesh(curved.v,curved.f[:,[0,2,1]],curved.u,"um")
    preserved = resample_annulus(reversed_mesh,pairs[:,::-1],13,25)
    legacy = resample_annulus(reversed_mesh,pairs[:,::-1],13,25,preserve_orientation=False)
    assert_allclose(preserved.mesh.v,result.mesh.v,atol=1e-12)
    assert_array_equal(preserved.mesh.f,rectilinear_faces(13,25)[:,[0,2,1]])
    assert_array_equal(legacy.mesh.f,rectilinear_faces(13,25))
    assert_allclose(face_normals(preserved.mesh),-face_normals(legacy.mesh),atol=1e-12)


def test_end_to_end_generated_chart_class_registration_and_failures(tmp_path):
    mesh,_ = cylinder(6,14)
    annulus,_ = glue_rectilinear_seam(mesh,6,14)
    tubi = TubULAR(time_points=[0,2],opts={"nU":11,"nV":19,"meshDir":tmp_path,"flipy":True},
                   mesh_sources={"cylinderMeshClean":{0:annulus}})
    with pytest.raises(ValueError,match="setTime"):
        tubi.generateCurrentUVCutMesh(np.array([[0,1],[2,3]]))
    tubi.setTime(0)
    cut = tubi.generateCurrentCutMesh(np.arange(6)*7,register=True)
    cached = tubi.getCurrentCutMesh()
    preview = tubi.generateCurrentUVCutMesh(cut.path_pairs)
    assert preview.mesh.uv.shape == (209,2)
    with pytest.raises(FileNotFoundError):
        tubi.getCurrentUVCutMesh()
    result = tubi.generateCurrentUVCutMesh(cut.path_pairs,register=True)
    assert tubi.getCurrentUVCutMesh() is result.mesh
    assert tubi.getCurrentCutMesh() is cached and tubi.currentTime == 0
    assert_array_equal(result.mesh.v,preview.mesh.v)
    for kwargs in ({"n_v":3},{"interpolation":"other"},{"normals":np.zeros_like(cut.mesh.v)}):
        with pytest.raises(ValueError):
            tubi.generateCurrentUVCutMesh(cut.path_pairs,register=True,**kwargs)
        assert tubi.getCurrentUVCutMesh() is result.mesh
    tubi.setTime(2)
    with pytest.raises(FileNotFoundError):
        tubi.generateCurrentUVCutMesh(cut.path_pairs)
    tubi.setTime(0)
    assert tubi.getCurrentUVCutMesh() is result.mesh
    assert not list(tmp_path.iterdir())
    assert TubULAR.method_status("generateCurrentUVCutMesh")["status"] == "adapted"


def test_rigid_motion_and_source_vertex_face_renumbering():
    mesh,pairs = cylinder()
    reference = resample_annulus(mesh,pairs,11,17)
    q,_ = np.linalg.qr([[1.,2,3],[-2.,1,1],[1.,3,-1]])
    moved = resample_annulus(mesh.with_vertices(mesh.v@q.T+[2,-3,4]),pairs,11,17)
    assert_allclose(moved.mesh.v,reference.mesh.v@q.T+[2,-3,4],atol=1e-12)
    rng = np.random.default_rng(28)
    permutation = rng.permutation(len(mesh.v)); inverse = np.argsort(permutation)
    renamed = Mesh(mesh.v[permutation],inverse[mesh.f[rng.permutation(len(mesh.f))]],
                   mesh.u[permutation])
    sampled = resample_annulus(renamed,inverse[pairs],11,17)
    assert_allclose(sampled.mesh.v,reference.mesh.v,atol=1e-12)


def test_large_integer_phase_offset_and_no_legacy_extrapolation():
    mesh,pairs = cylinder()
    mesh = Mesh(mesh.v,mesh.f,mesh.u/[4,2*np.pi])
    shifted = Mesh(mesh.v,mesh.f,mesh.u+[0,10000])
    original = resample_annulus(mesh,pairs,11,17)
    result = resample_annulus(shifted,pairs,11,17)
    assert_allclose(result.mesh.v,original.mesh.v,atol=1e-12)
    assert np.all(np.abs(result.corner_v_shifts+10000) <= 2)
    with pytest.raises(ValueError,match="no extrapolation"):
        resample_annulus(shifted,pairs,11,17,interpolation="matlab_scattered")


@pytest.mark.parametrize("counts", [(-1,1),(1.5,1),(True,1),(1,),None,"1,1"])
def test_invalid_tile_counts_fail(counts):
    mesh,pairs = cylinder()
    with pytest.raises(ValueError):
        tile_annular_mesh(mesh,pairs,counts)


@pytest.mark.parametrize("kwargs", [{"n_u":1},{"n_v":3},{"n_u":3.5},{"n_v":True},
                                     {"interpolation":"cubic"},{"preserve_orientation":1}])
def test_invalid_grid_options_fail(kwargs):
    mesh,pairs = cylinder()
    options = {"n_u":5,"n_v":9}; options.update(kwargs)
    with pytest.raises(ValueError):
        resample_annulus(mesh,pairs,**options)


def test_inconsistent_period_geometry_and_nonvertical_end_rings_fail():
    mesh,pairs = cylinder()
    with pytest.raises(ValueError,match="UV"):
        resample_annulus(Mesh(mesh.v,mesh.f),pairs,7,13)
    chart = mesh.u.copy(); chart[-3,1] += .01
    with pytest.raises(ValueError,match="uniform"):
        tile_annular_mesh(Mesh(mesh.v,mesh.f,chart),pairs)
    xyz = mesh.v.copy(); xyz[-3,0] += .01
    with pytest.raises(ValueError,match="identical"):
        tile_annular_mesh(Mesh(xyz,mesh.f,mesh.u),pairs)
    chart = mesh.u.copy(); chart[:,0] += .05*np.sin(chart[:,1]); chart[-5:] = chart[:5]+[0,2*np.pi]
    with pytest.raises(ValueError,match="boundary rings"):
        resample_annulus(Mesh(mesh.v,mesh.f,chart),pairs,7,13)
    with pytest.raises(ValueError,match="boundary rings"):
        resample_annulus(Mesh(mesh.v,mesh.f,mesh.u-[.5,0]),pairs,7,13)


def test_discontinuous_or_invalid_vertex_fields_and_normals_fail():
    mesh,pairs = cylinder()
    result = resample_annulus(mesh,pairs,7,13)
    discontinuous = np.ones(len(mesh.v)); discontinuous[-1] = 2
    for field in (discontinuous,np.ones(3),np.full(len(mesh.v),np.nan),
                  np.ones((len(mesh.v),0)),np.ones((len(mesh.v),2,2)),np.ones(len(mesh.v))*1j):
        with pytest.raises(ValueError):
            result.interpolate(field)
    for normals in (np.zeros_like(mesh.v),np.ones((len(mesh.v),2)),np.ones(len(mesh.v))):
        with pytest.raises(ValueError):
            resample_annulus(mesh,pairs,7,13,normals=normals)
    normals = np.tile([1.,0,0],(len(mesh.v),1)); normals[5:10] = [-1,0,0]
    with pytest.raises(ValueError,match="cancel"):
        resample_annulus(mesh,pairs,7,17,normals=normals)


def test_matlab_tiling_full_class_and_triangle_interpolation_reference():
    root = Path(__file__).resolve().parents[1]
    fixture = root/"tests/data/matlab_uvgrid_reference.mat"
    provenance = json.loads((root/"tests/data/uvgrid_provenance.json").read_text())
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == provenance["fixture_sha256"]
    assert hashlib.sha256((root/"scripts/generate_uvgrid_reference.m").read_bytes()).hexdigest() == provenance["generator_sha256"]
    differences = []
    for case_id,case in enumerate(loadmat(fixture,simplify_cells=True)["cases"]):
        mesh = Mesh(case["vertices"],case["faces"].astype(int)-1,case["chart"])
        pairs = case["pathPairs"].astype(int)-1
        tiled = tile_annular_mesh(mesh,pairs)
        assert_array_equal(tiled.mesh.f,case["tiledFaces"]-1)
        assert_array_equal(tiled.mesh.v,case["tiledXYZ"])
        assert_allclose(tiled.mesh.u,case["tiledUV"],atol=1e-14)
        assert_array_equal(case["normals"][tiled.vertex_sources],case["tiledNormals"])
        for method,prefix in (("mesh","mesh"),("matlab_scattered","legacy")):
            result = resample_annulus(mesh,pairs,int(case["nU"]),int(case["nV"]),
                                      normals=case["normals"],interpolation=method,
                                      preserve_orientation=False)
            assert_allclose(result.mesh.v,case[prefix+"XYZ"],rtol=1e-11,atol=1e-12)
            assert_allclose(result.normals,case[prefix+"Normals"],rtol=1e-11,atol=1e-12)
            actual_field = result.interpolate(case["field"])
            if method == "matlab_scattered" and case_id == 0:
                # Co-circular grid cells have two equally valid Delaunay
                # diagonals. Non-affine fields distinguish them. Both MATLAB
                # and Python must choose one of the two analytic interpolants;
                # their tie choices need not be the same across libraries.
                options = rectangular_interpolants(mesh,case["field"],case["query"])
                for field in (actual_field,case["legacyField"]):
                    errors = np.max(np.abs(options-field[:,None,:]),axis=2)
                    assert np.all(np.min(errors,axis=1) < 1e-12)
            else:
                assert_allclose(actual_field,case[prefix+"Field"],rtol=1e-11,atol=1e-12)
            assert_allclose(result.mesh.u,case["legacyUV"],atol=1e-14)
            assert_array_equal(result.mesh.f,case["legacyFaces"]-1)
            assert_array_equal(result.path_pairs,case["legacyPairs"]-1)
        differences.append(np.max(np.abs(case["meshXYZ"]-case["legacyXYZ"])))
    # The curved cases actually exercise the documented mathematical choice:
    # Delaunay interpolation must not masquerade as original-triangle sampling.
    assert max(differences) > .01


def rectangular_interpolants(mesh,values,query):
    """Independent piecewise-linear values for the two diagonals of a cell."""
    us,vs = np.unique(mesh.u[:,0]),np.unique(mesh.u[:,1])
    i = np.clip(np.searchsorted(us,query[:,0],side="right")-1,0,len(us)-2)
    j = np.clip(np.searchsorted(vs,query[:,1],side="right")-1,0,len(vs)-2)
    a = ((query[:,0]-us[i])/(us[i+1]-us[i]))[:,None]
    b = ((query[:,1]-vs[j])/(vs[j+1]-vs[j]))[:,None]
    grid = values.reshape(len(vs),len(us),-1)
    f00,f10,f01,f11 = grid[j,i],grid[j,i+1],grid[j+1,i],grid[j+1,i+1]
    first = np.where(b>=a,(1-b)*f00+a*f11+(b-a)*f01,
                     (1-a)*f00+(a-b)*f10+b*f11)
    second = np.where(a+b<=1,(1-a-b)*f00+a*f10+b*f01,
                      (1-b)*f10+(a+b-1)*f11+(1-a)*f01)
    return np.stack((first,second),axis=1)
