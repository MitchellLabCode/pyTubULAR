function generate_alignment_reference(sourceRoot, outputFile)
% Run the unchanged computeAPDVCoords on supplied landmark files. The small
% HDF5 volume only satisfies its initial read; no landmark segmentation runs.
% Plotting is unavoidable in that method, but all outputs stay in scratch.
oldPath = path;
cleanupPath = onCleanup(@() path(oldPath));
addpath(sourceRoot);
addpath(genpath(fullfile(sourceRoot, 'utility')));
oldVisible = get(groot, 'defaultFigureVisible');
set(groot, 'defaultFigureVisible', 'off');
cleanupVisible = onCleanup(@() set(groot, 'defaultFigureVisible', oldVisible));
scratch = tempname; mkdir(scratch);
cleanupScratch = onCleanup(@() rmdir(scratch, 's'));
anterior = [2 -3 4; -1 2 .4; .5 1 -2; 0 0 0];
posterior = [8 1 7; 3 2 .4; 2 -3 1; 0 0 4];
dorsal = [4 2 10; 0 2 -3; -1 5 3; 3 1 2];
vertices = [0 0 0; 1 0 0; 0 2 0; 0 0 3];
faces = [1 3 2; 1 2 4; 1 4 3; 2 3 4];
resolution = .37;
rotations = zeros(3, 3, size(anterior, 1));
transformRotations = rotations;
translations = zeros(size(anterior));
aligned = zeros(size(vertices, 1), 3, size(anterior, 1));
mirrored = aligned;
for i = 1:size(anterior, 1)
    caseDir = fullfile(scratch, sprintf('case%d', i)); mkdir(caseDir);
    xp = struct('fileMeta', struct('timePoints', 0, 'stackResolution', [1 1 1], ...
        'filenameFormat', 'sample_%06d.tif', 'dataDir', caseDir), 'expMeta', struct(), ...
        'detectOptions', struct('ssfactor', 1, 'ofn_smoothply', 'mesh_%06d.ply'));
    opts = struct('flipy', false, 'meshDir', caseDir, 'timeUnits', 'min', ...
        'spaceUnits', 'um', 'nU', 3, 'nV', 5, 'makeDirs', false, 'lambda_err', 0);
    tubi = TubULAR(xp, opts);
    tubi.fileName.rot = fullfile(caseDir, 'rot.txt');
    tubi.fileName.trans = fullfile(caseDir, 'trans.txt');
    tubi.fileName.dpt = fullfile(caseDir, 'dpt.txt');
    writematrix(anterior(i, :), fullfile(caseDir, 'apt_for_rot.txt'), 'Delimiter', ',');
    writematrix(posterior(i, :), fullfile(caseDir, 'ppt_for_rot.txt'), 'Delimiter', ',');
    writematrix(dorsal(i, :), tubi.fileName.dpt, 'Delimiter', ',');
    probFile = fullfile(caseDir, 'labels.h5');
    h5create(probFile, '/exported_data', [2 3 4]);
    h5write(probFile, '/exported_data', zeros(2, 3, 4));
    tubi.fullFileBase.apdProb = probFile;
    tubi.data.ilastikOutputAxisOrder = 'xyz';
    tubi.fullFileBase.mesh = fullfile(caseDir, 'raw_%06d.ply');
    write_fixture_ply(sprintf(tubi.fullFileBase.mesh, 0), vertices, faces);
    % The class declaration exposes only three outputs, despite the external
    % method's five-output signature. Read its saved frame (5 significant digits).
    tubi.computeAPDVCoords(struct('use_iLastik', true, ...
        'tref', 0, 'preview', false, 'overwrite', false));
    rot = readmatrix(tubi.fileName.rot);
    trans = readmatrix(tubi.fileName.trans);
    rotations(:, :, i) = rot;
    translations(i, :) = trans;
    % Test the vertex transform separately with a full-precision proper input
    % rotation, rather than treating rounded frame files as exactly orthogonal.
    az = .37*i; ay = .21*i;
    rz = [cos(az) -sin(az) 0; sin(az) cos(az) 0; 0 0 1];
    ry = [cos(ay) 0 sin(ay); 0 1 0; -sin(ay) 0 cos(ay)];
    transformRotations(:, :, i) = rz*ry;
    tubi.APDV.rot = rz*ry; tubi.APDV.trans = trans; tubi.APDV.resolution = resolution;
    % Original xyz2APDV uses the same vertex transform as alignMeshesAPDV.
    aligned(:, :, i) = tubi.xyz2APDV(vertices);
    tubi.flipy = true;
    mirrored(:, :, i) = tubi.xyz2APDV(vertices);
end
matlabVersion = version;
save(outputFile, 'anterior', 'posterior', 'dorsal', 'vertices', 'faces', ...
    'resolution', 'rotations', 'transformRotations', 'translations', ...
    'aligned', 'mirrored', 'matlabVersion', '-v7');
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
