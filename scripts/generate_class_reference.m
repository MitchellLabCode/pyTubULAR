function generate_class_reference(sourceRoot, outputDir)
% Inventory and reference outputs from the unchanged original TubULAR class.
oldPath = path;
cleanup = onCleanup(@() path(oldPath));
addpath(sourceRoot);
meta = ?TubULAR;
entries = struct('name', {}, 'static', {}, 'hidden', {});
for i = 1:numel(meta.MethodList)
    m = meta.MethodList(i);
    if strcmp(m.DefiningClass.Name, 'TubULAR') && strcmp(m.Access, 'public')
        % InputNames would parse every external method, including legacy files
        % with syntax errors. Names/flags can be inspected without executing them.
        entries(end+1) = struct('name', m.Name, 'static', m.Static, 'hidden', m.Hidden);
    end
end
fid = fopen(fullfile(outputDir, 'matlab_class_methods.json'), 'w');
fprintf(fid, '%s\n', jsonencode(entries, PrettyPrint=true));
fclose(fid);

imageSize = [13, 17];
XY = [1 1; 13 17; 4.25 9.5; -2.5 23];
dXY = [0 0; 1.5 -2.25; -3 7];
uvSingle = TubULAR.XY2uv(imageSize, XY, false, 2.3, 3.7);
uvDouble = TubULAR.XY2uv(imageSize, XY, true, 2.3, 3.7);
xySingle = TubULAR.uv2XY(imageSize, uvSingle, false, 2.3, 3.7);
xyDouble = TubULAR.uv2XY(imageSize, uvDouble, true, 2.3, 3.7);
duvSingle = TubULAR.dXY2duv(imageSize, dXY, false, 2.3, 3.7);
duvDouble = TubULAR.dXY2duv(imageSize, dXY, true, 2.3, 3.7);
imEven = uint8(reshape(1:24, [6 4]));
imOdd = uint8(reshape(1:20, [5 4]));
coverEven = TubULAR.extendImage2DoubleCover(imEven);
coverOdd = TubULAR.extendImage2DoubleCover(imOdd);
singleDouble = TubULAR.singleToDoubleCover(XY, 17);
doubleSingle = TubULAR.doubleToSingleCover(XY, 34);
[clipX, clipY] = TubULAR.clipXY([-4 1 7 20], [-1 1 17 18], 13, 17);
magnitude = [1 2 3; 4 2 1];
theta = [0 pi/4 pi/2; -pi/3 0 pi/5];
[magAverage, thetaAverage] = TubULAR.dvAverageNematic(magnitude, theta);
[dorsal, ventral, left, right] = TubULAR.quarterIndicesDV(33);
matlabVersion = version;
save(fullfile(outputDir, 'matlab_class_reference.mat'), 'imageSize', 'XY', 'dXY', ...
    'uvSingle', 'uvDouble', 'xySingle', 'xyDouble', 'duvSingle', 'duvDouble', ...
    'imEven', 'imOdd', 'coverEven', 'coverOdd', 'singleDouble', 'doubleSingle', ...
    'clipX', 'clipY', 'magnitude', 'theta', 'magAverage', 'thetaAverage', ...
    'dorsal', 'ventral', 'left', 'right', 'matlabVersion', '-v7');
end
