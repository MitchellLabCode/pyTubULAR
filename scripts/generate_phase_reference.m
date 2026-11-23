function generate_phase_reference(sourceRoot, outputFile)
% Execute the unchanged non-resampled phiOffsetsFromPrevMesh branch.
% Request one output: its non-resampled residual branch assigns nV values
% into nV-1 slots. Straight centerlines make its hoop matching independently
% reproducible by linear interpolation, without modifying any source files.
oldPath = path; cleanup = onCleanup(@() path(oldPath));
addpath(genpath(fullfile(sourceRoot,'utility')));
nu = 11; nv = 17;
us = linspace(.03,.97,nu)'; prevu = linspace(0,1,13)';
vs = linspace(0,1,nv)';
[u,v] = ndgrid(us,vs); [pu,pv] = ndgrid(prevu,vs);
faces = defineFacesRectilinearGrid([u(:),v(:)],nu,nv);
prevFaces = defineFacesRectilinearGrid([pu(:),pv(:)],length(prevu),nv);
pairs = [(1:nu)',((nv-1)*nu+1:nv*nu)'];
cases = cell(1,3);
for c = 1:3
    if c == 1, shifts = -.125*ones(nu,1);
    elseif c == 2, shifts = .073*ones(nu,1);
    else, shifts = -.1+.2*us; end
    theta = 2*pi*(v+shifts);
    prevTheta = 2*pi*pv;
    xyz = [u(:), (1+.3*u(:)).*cos(theta(:))+.1*u(:), ...
           (1+.2*u(:)).*sin(theta(:))+.05*u(:)];
    prevXYZ = [pu(:), (1+.3*pu(:)).*cos(prevTheta(:))+.1*pu(:), ...
               (1+.2*pu(:)).*sin(prevTheta(:))+.05*pu(:)];
    xyz(end-nu+1:end,:) = xyz(1:nu,:);
    prevXYZ(end-length(prevu)+1:end,:) = prevXYZ(1:length(prevu),:);
    currentHoops = reshape(xyz,[nu nv 3]);
    previousHoops = reshape(prevXYZ,[length(prevu) nv 3]);
    % Same target curves as original distance2curve + straight-centerline
    % interpolation; computed with the independent unchanged mesh interpolator.
    target = interpolate2Dpts_3Dmesh(prevFaces,[pu(:),pv(:)],prevXYZ,[u(:),v(:)]);
    targetHoops = reshape(target,[nu nv 3]);
    targetHoops(:,end,:) = targetHoops(:,1,:);
    mesh = struct('v',xyz,'f',faces,'u',[u(:),v(:)],'pathPairs',pairs);
    [tf,tuv,tv] = tileAnnularCutMesh(mesh,[1 1]);
    opts = struct('preview',false,'resample_hoops',false, ...
        'avgpts',[us,.1*us,.05*us],'prev_avgpts',[prevu,.1*prevu,.05*prevu], ...
        'prev_avgpts_ss',prevu,'optimization',optimset('TolX',1e-12));
    bounds = [-.35,.35];
    legacyOffsets = phiOffsetsFromPrevMesh(tf,tuv,tv,us,vs,previousHoops,bounds(1),bounds(2),opts);
    aligned = zeros(nu,nv,3); losses = zeros(nu,1);
    for i = 1:nu
        points = interpolate2Dpts_3Dmesh(tf,tuv,tv,[us(i)*ones(nv,1),mod(vs+legacyOffsets(i),1)]);
        aligned(i,:,:) = points;
        losses(i) = sum(vecnorm(points-squeeze(targetHoops(i,:,:)),2,2).^2);
    end
    cases{c} = struct('currentHoops',currentHoops,'previousHoops',previousHoops, ...
        'targetHoops',targetHoops,'us',us,'bounds',bounds,'legacyOffsets',legacyOffsets, ...
        'aligned',aligned,'losses',losses,'phaseShift',shifts);
end
matlabVersion = version;
save(outputFile,'cases','matlabVersion','-v7');
end
