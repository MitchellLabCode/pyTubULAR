function generate_parameterization_reference(sourceRoot, outputFile)
% Execute the unchanged custom-path cutter and Dirichlet flattenAnnulus.
% No class initialization, plotting, file workflows, path search or remeshing.
oldPath = path;
cleanupPath = onCleanup(@() path(oldPath));
addpath(fullfile(sourceRoot, 'utility', 'axisymmetric_pullbacks'));
addpath(fullfile(sourceRoot, 'external', 'gptoolbox', 'mesh'));
nu = 6; nv = 13;
[x, theta] = ndgrid(linspace(0, 4, nu), (0:nv-1)*2*pi/nv);
faces = zeros(2*(nu-1)*nv, 3); count = 1;
for j = 0:nv-1
    for i = 1:nu-1
        a = j*nu+i; b = mod(j+1,nv)*nu+i;
        faces(count:count+1,:) = [a a+1 b+1; a b+1 b];
        count = count+2;
    end
end
cases = cell(1, 4);
v = [x(:), cos(theta(:)), -sin(theta(:))];
cases{1} = evaluate(v, faces, (1:nu)');
% Nonuniform curved tube, a tapered elliptical cross-section, and a seam
% following diagonal edges. This exercises negative assembled cot weights.
radius = 1+.17*sin(1.1*x)+.08*cos(3*theta+.4*x);
angle = theta+.13*x+.04*sin(2*theta).*sin(x);
v = [x(:)+.07*sin(theta(:)).*sin(x(:)), ...
    radius(:).*cos(angle(:))+.12*x(:).^2, ...
    -.75*radius(:).*sin(angle(:))+.08*x(:)];
cases{2} = evaluate(v, faces, (1:nu)');
seam = (1:nu)' + (0:nu-1)'*nu;
cases{3} = evaluate(v, faces, seam);
cases{4} = evaluate(v, faces(:,[1 3 2]), flipud(seam));
matlabVersion = version;
save(outputFile, 'cases', 'matlabVersion', '-v7');
end

function result = evaluate(vertices, faces, seam)
options = struct('method', 'custom', 'path', seam);
cut = cylinderCutMesh(faces, vertices, zeros(size(vertices)), seam(1), seam(end), options);
mapped = flattenAnnulus(cut, 'Dirichlet');
L = cotmatrix(cut.v, cut.f);
clampedEdges = nnz(tril(L,-1)<0);
result = struct('vertices', vertices, 'faces', faces, 'seam', seam, ...
    'cutVertices', cut.v, 'cutFaces', cut.f, 'pathPairs', cut.pathPairs, ...
    'cutToUncut', cut.cutIndsToUncutInds, 'uv', mapped.u, ...
    'clampedEdges', clampedEdges);
end
