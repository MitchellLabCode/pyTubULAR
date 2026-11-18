function generate_writhe_reference(sourceRoot, dependencyRoot, outputFile)
% Run unchanged TubULAR curve kernels and its measureWrithe numerical helper.
% dependencyRoot supplies the original linesegments.m (not in TubULAR itself).
oldPath = path;
cleanupPath = onCleanup(@() path(oldPath));
addpath(dependencyRoot);
addpath(genpath(fullfile(sourceRoot, 'utility')));
scratch = tempname; mkdir(scratch);
cleanupScratch = onCleanup(@() rmdir(scratch, 's'));
t = 2.3*pi*linspace(0, 1, 37)'.^1.15;
xyz = [0.6*t, 1.2*sin(t)+0.04*t, 0.7*cos(t)+0.03*cos(3*t)];
ss = ss_from_xyz(xyz);
[levitt, levittDensity] = writheLevitt(xyz, false);
[gaussLegacy, gaussLegacyDensity] = writheGaussIntegral(xyz, ss);
reflected = xyz .* [1 -1 1];
[levittReflected, densityReflected] = writheLevitt(reflected, false);
angle = linspace(0, 2*pi, 17)'; angle = angle(1:end-1);
closedXYZ = [cos(angle)+0.1*sin(2*angle), 1.3*sin(angle)+0.04*cos(3*angle), ...
    0.4*sin(2*angle)+0.3*cos(3*angle)];
[levittClosed, densityClosed] = writheLevitt(closedXYZ, true);
times = [5 9];
pattern = fullfile(scratch, 'cline_%04d.mat');
for k = 1:2
    mcline = xyz*(1+0.2*(k-1)) + (k-1)*[3 -2 4];
    if k == 2
        mcline(:, 2) = -mcline(:, 2);
    end
    mss = ss_from_xyz(mcline);
    avgpts = mcline(1:3:end, :);
    save(sprintf(pattern, times(k)), 'mss', 'mcline', 'avgpts');
end
% omit_endpts=1 in the original retains indices 1:N-1 (not symmetric).
[seriesWr, seriesDensity, seriesDerivative, seriesLength, seriesCurves] = ...
    aux_compute_writhe(pattern, times, 0, 1, false, false);
matlabVersion = version;
save(outputFile, 'xyz', 'ss', 'levitt', 'levittDensity', 'gaussLegacy', ...
    'gaussLegacyDensity', 'reflected', 'levittReflected', 'densityReflected', ...
    'closedXYZ', 'levittClosed', 'densityClosed', 'times', 'seriesWr', ...
    'seriesDensity', 'seriesDerivative', 'seriesLength', 'seriesCurves', 'matlabVersion', '-v7');
end
