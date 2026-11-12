function generate_surface_reference(sourceRoot, outputFile)
% Run unchanged original class methods with synthetic inputs in temporary files.
oldPath = path;
cleanupPath = onCleanup(@() path(oldPath));
addpath(sourceRoot);
addpath(genpath(fullfile(sourceRoot, 'utility')));
scratch = tempname; mkdir(scratch);
cleanupScratch = onCleanup(@() rmdir(scratch, 's'));
xp = struct('fileMeta', struct('timePoints', [0 2], 'stackResolution', [0.5 0.5 0.5], ...
    'filenameFormat', 'sample_%06d.tif', 'dataDir', scratch), 'expMeta', struct(), ...
    'detectOptions', struct('ssfactor', 1, 'ofn_smoothply', 'mesh_%06d.ply'));
% lambda_err is missing from the legacy default smoothing struct.
% makeDirs=false and no t0 option prevent initialization output writes.
opts = struct('flipy', true, 'meshDir', scratch, 'timeUnits', 'min', 'spaceUnits', 'um', ...
    'nU', 3, 'nV', 5, 'makeDirs', false, 'lambda_err', 0);
tubi = TubULAR(xp, opts);
tubi.setTime(0);

chart = [0 0; 2 0; 2 1; 0 1];
vertices = [0 0 0; 4 0 2; 5 3 2; 1 3 0];
faces = [1 2 3; 1 3 4];
mesh = struct('v', vertices, 'f', faces, 'u', chart);
tubi.currentMesh.spcutMeshSmRS = mesh;
queryUV = [0.7 0.2; 0.2 0.8; 1.2 0.4; -0.1 0.4];
[mappedXYZ, mappedFaces, ~, barycentric] = tubi.uv2APDV(queryUV, 'spsmrs');
umax = 2.3;
scaledQuery = queryUV .* [umax 1];
[scaledXYZ, scaledFaces, ~, scaledBarycentric] = tubi.uv2APDV(scaledQuery, 'spsmrs', umax, 1);

nU = 3; nV = 5; Lx = 13; Ly = 18;
field = reshape(sin(1:nU*nV), nU, nV) + (1:nU)' * (1:nV);
queryXY = [1 1; 6.25 9.3; 12.2 17.5; -2 8.3; 9.7 35; 14.1 -6];
tubi.piv.imCoords = 'sp_sme';
interpolatedField = tubi.interpolateOntoPullbackXY(queryXY, field(:), struct('Lx', Lx, 'Ly', Ly));
% Original class ignores iMethod; compare the newly honored nearest option
% directly to MATLAB's griddedInterpolant, including exact X midpoint ties.
xgrid = linspace(1, Lx, nU);
ygrid = linspace(1, 1.5*Ly, 3*nV-2) - 0.25*Ly;
tiledField = [field(:, 1:end-1), field, field(:, 2:end)];
nearestQueries = [4, ygrid(5); 10, ygrid(8); -20, ygrid(4); 15, ygrid(6)];
nearestInterpolant = griddedInterpolant({xgrid, ygrid}, tiledField, 'nearest', 'nearest');
nearestField = nearestInterpolant(nearestQueries(:, 1), nearestQueries(:, 2));

rot = [0 -1 0; 1 0 0; 0 0 1]; trans = [3 -4 5]; resolution = 0.5;
tubi.APDV.rot = rot; tubi.APDV.trans = trans; tubi.APDV.resolution = resolution;
tubi.normalShift = -2;
rawVertices0 = vertices;
rawVertices2 = vertices * 1.3 + [-2 3 1];
tubi.fullFileBase.mesh = fullfile(scratch, 'raw_%06d.ply');
write_fixture_ply(sprintf(tubi.fullFileBase.mesh, 0), rawVertices0, faces);
write_fixture_ply(sprintf(tubi.fullFileBase.mesh, 2), rawVertices2, faces);
[boundsRaw, boundsPixels, boundsPhysical, boundsBuffered] = tubi.measureXYZLims();
[getRaw, getPixels, getPhysical, getBuffered] = tubi.getXYZLims();

mss = [0; 1.5; 3];
mcline = [1 2 3; 2 3 4; 3 2 5];
avgpts = [1 2 3; 3 2 5];
tubi.fullFileBase.clineDVhoop = fullfile(scratch, 'cline_%06d.mat');
save(sprintf(tubi.fullFileBase.clineDVhoop, 0), 'mss', 'mcline', 'avgpts');
[loadedMss, loadedCline, loadedAvgpts] = tubi.getCurrentClineDVhoop();
matlabVersion = version;
save(outputFile, 'chart', 'vertices', 'faces', 'queryUV', 'mappedXYZ', 'mappedFaces', ...
    'barycentric', 'umax', 'scaledQuery', 'scaledXYZ', 'scaledFaces', 'scaledBarycentric', ...
    'nU', 'nV', 'Lx', 'Ly', 'field', 'queryXY', 'interpolatedField', 'nearestQueries', 'nearestField', ...
    'rot', 'trans', 'resolution', 'rawVertices0', 'rawVertices2', ...
    'boundsRaw', 'boundsPixels', 'boundsPhysical', 'boundsBuffered', ...
    'getRaw', 'getPixels', 'getPhysical', 'getBuffered', ...
    'mss', 'mcline', 'avgpts', 'loadedMss', 'loadedCline', 'loadedAvgpts', 'matlabVersion', '-v7');
end

function write_fixture_ply(filename, vertices, faces)
fid = fopen(filename, 'w');
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, 'ply\nformat ascii 1.0\nelement vertex %d\n', size(vertices, 1));
fprintf(fid, 'property double x\nproperty double y\nproperty double z\n');
fprintf(fid, 'element face %d\nproperty list uchar int vertex_indices\nend_header\n', size(faces, 1));
fprintf(fid, '%.17g %.17g %.17g\n', vertices');
fprintf(fid, '3 %d %d %d\n', (faces-1)');
end
