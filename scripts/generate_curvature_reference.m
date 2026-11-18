function generate_curvature_reference(dependencyRoot, outputFile)
% Run curvature routines used by TubULAR.measureCurvatures. All calculations
% are unchanged; fix one obsolete vector-valued loop bound in a TEMPORARY copy.
oldPath = path;
cleanupPath = onCleanup(@() path(oldPath));
addpath(dependencyRoot);
scratch = tempname; mkdir(scratch);
cleanupScratch = onCleanup(@() rmdir(scratch, 's'));
principalSource = fileread(fullfile(dependencyRoot, 'getPrincipalCurvatures.m'));
pattern = 'for i=1:size\(FV.vertices\)(\r?\n)%This';
assert(numel(regexp(principalSource, pattern)) == 1);
principalSource = regexprep(principalSource, pattern, 'for i=1:size(FV.vertices,1)$1%This');
fid = fopen(fullfile(scratch, 'getPrincipalCurvatures.m'), 'w');
fwrite(fid, principalSource); fclose(fid);
addpath(scratch, '-begin');
principalReferencePatch = 'Temporary copy only: active for i=1:size(FV.vertices) becomes for i=1:size(FV.vertices,1).';
[x, y] = ndgrid([-1 -.35 .2 1.3], [-.8 -.1 .7 1.4]);
v = [x(:), y(:), .3*x(:).^2+.1*x(:).*y(:)-.2*y(:).^2];
f = zeros(18, 3); count = 1;
for j = 0:2
    for i = 1:3
        a = j*4+i;
        f(count:count+1, :) = [a a+1 a+5; a a+5 a+4];
        count = count+2;
    end
end
cases = cell(1, 3);
cases{1} = evaluate(v, f);
% A nonuniform closed ellipsoid; orient each hull triangle outward.
ids = (0:63)'; z = 1-2*(ids+.5)/64;
phi = ids*pi*(3-sqrt(5));
v = [sqrt(1-z.^2).*cos(phi), sqrt(1-z.^2).*sin(phi), z];
f = convhull(v);
nf = cross(v(f(:,2),:)-v(f(:,1),:), v(f(:,3),:)-v(f(:,1),:));
bad = sum(nf.*v(f(:,1),:), 2)<0;
f(bad,:) = f(bad,[1 3 2]);
v = v.*[1.6 1 .7]+[3 -2 4];
cases{2} = evaluate(v, f);
cases{3} = evaluate(v, f(:,[1 3 2]));
matlabVersion = version;
save(outputFile, 'cases', 'matlabVersion', 'principalReferencePatch', '-v7');
end

function result = evaluate(vertices, faces)
fv = struct('vertices', vertices, 'faces', faces);
[k, d1, d2] = GetCurvatures(fv, 0);
nf = CalcFaceNormals(fv);
[normals, vertexAreas, cornerAreas, up, vp] = CalcVertexNormals(fv, nf);
[~, tensor] = CalcCurvature(fv, normals, nf, vertexAreas, cornerAreas, up, vp);
shapeOperator = zeros(3,3,size(vertices,1));
for i = 1:size(vertices,1)
    basis = [up(i,:); vp(i,:)];
    shapeOperator(:,:,i) = basis'*tensor{i}*basis;
end
result = struct('vertices', vertices, 'faces', faces, 'principal', k', ...
    'd1', d1, 'd2', d2, 'mean', -mean(k,1)', 'gaussian', prod(k,1)', ...
    'normals', normals, 'vertexAreas', vertexAreas, 'cornerAreas', cornerAreas, ...
    'shapeOperator', shapeOperator);
end
