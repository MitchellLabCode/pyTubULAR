function generate_matlab_reference(sourceRoot, outputFile)
% Run the ORIGINAL MATLAB routines, not a second implementation of the port.
% Example: generate_matlab_reference('/path/to/tubular', '/tmp/reference.mat')
oldPath = path;
cleanup = onCleanup(@() path(oldPath));
addpath(fullfile(sourceRoot, 'utility', 'mesh_handling'));
addpath(fullfile(sourceRoot, 'utility', 'mesh_handling', 'coordinate_transformations'));
addpath(fullfile(sourceRoot, 'utility', 'mesh_handling', 'rectilinearMesh'));
addpath(fullfile(sourceRoot, 'utility', 'geometry', 'metric'));
nU = 5; nV = 7;
[uu, vv] = ndgrid(linspace(-0.3, 1.4, nU), linspace(-0.4, 1.1, nV));
u = [uu(:), vv(:)];
f = defineFacesRectilinearGrid(u, nU, nV);
v = [1.3*u(:,1)+0.2*u(:,2), 0.1*u(:,1)+0.9*u(:,2), ...
    0.23*u(:,1).^2 + 0.17*u(:,1).*u(:,2) + 0.11*u(:,2).^2];
J23 = cellstack(jacobian2Dto3DMesh(u, v, f));
J32 = cellstack(jacobian3Dto2DMesh(u, v, f));
metric = cellstack(inducedMetric(f, v, u));
[g, b] = constructFundamentalForms(f, v, u);
g = cellstack(g); b = cellstack(b);
[V2F, F2V] = meshAveragingOperators(f, v);
V2F = full(V2F); F2V = full(F2V);
normals = faceNormal(triangulation(f,v));
legacyNormals = faceNormals(f,v);
area = meshArea(v,f);
% A closed tetrahedron supplies the signed-volume reference.
tetV = [0 0 0; 1 0 0; 0 2 0; 0 0 3] + [5 -7 2];
tetF = [1 3 2; 1 2 4; 1 4 3; 2 3 4];
[volume, tetArea] = meshVolumeArea(tetV,tetF);
matlabVersion = version;
save(outputFile, 'u','v','f','nU','nV','J23','J32','metric','g','b', ...
    'V2F','F2V','normals','legacyNormals','area','tetV','tetF','volume', ...
    'tetArea','matlabVersion','-v7');
end

function output = cellstack(values)
output = permute(cat(3, values{:}), [3 1 2]);
end
