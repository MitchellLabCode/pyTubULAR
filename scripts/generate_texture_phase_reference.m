function generate_texture_phase_reference(matlabRoot, outputFile)
% Run the unchanged original integer phase-correlation helper, read-only.
% This checks the integer shift/sign contract, NOT whitening or full workflow
% equivalence. Python uses 1D normalized strip cross-correlation instead.
oldPath = path; restore = onCleanup(@() path(oldPath));
addpath(fullfile(matlabRoot,'ExtPhaseCorrelation'));
rng(403,'twister'); shapes = [64 64;63 63;64 49;32 79];
shifts = [-9 0 7]; cases = cell(4,3);
for j = 1:4
    reference = rand(shapes(j,:));
    for k = 1:3
        current = circshift(reference,[shifts(k),0]);
        % Disable the default smoothing. Use no bounds: the original secondary
        % peak search has a one-based indexing defect, deliberately not ported.
        [dx,dy] = PhaseCorrelationPotentialBound(current,reference,'none',2, ...
            'SigmaFilter',0);
        % Preserve the original rectangular-image wrap defect as evidence.
        expectedOriginal = shifts(k);
        if j==4 && k==1, expectedOriginal = 23; end
        assert(dx==0 && dy==expectedOriginal);
        cases{j,k} = struct('current',current,'reference',reference, ...
            'expected',shifts(k),'original_dx',dx,'original_dy',dy);
    end
end
matlabVersion = version;
save(outputFile,'cases','matlabVersion','-v7');
end
