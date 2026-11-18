function generate_beltrami_reference(sourceRoot, outputFile)
% Exercise unchanged bc_metric with 2D and 3D target maps.
oldPath = path;
cleanupPath = onCleanup(@() path(oldPath));
addpath(fullfile(sourceRoot, 'utility', 'geometry', 'metric'));
uv = [0 0; 1.3 .1; 1.1 1.2; -.2 .9; .5 .45];
faces = [1 2 5; 2 3 5; 3 4 5; 4 1 5];
referenceXYZ = [uv, .1*uv(:,1).^2-.2*uv(:,2).^2];
target2D = [1.4*uv(:,1)+.2*uv(:,2).^2, .8*uv(:,2)+.1*uv(:,1).^2];
target3D = [target2D, .3*uv(:,1).^2+.2*uv(:,1).*uv(:,2)-.15*uv(:,2).^2];
mu2D = bc_metric(faces, uv, target2D, 2);
mu3D = bc_metric(faces, uv, target3D, 3);
muFlat3D = bc_metric(faces, uv, [target2D, zeros(size(uv,1),1)], 3);
matlabVersion = version;
save(outputFile, 'uv', 'faces', 'referenceXYZ', 'target2D', 'target3D', ...
    'mu2D', 'mu3D', 'muFlat3D', 'matlabVersion', '-v7');
end
