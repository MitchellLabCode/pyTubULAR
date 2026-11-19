function generate_uvgrid_reference(sourceRoot, outputFile)
% Execute unchanged tiling/interpolation helpers AND generateCurrentUVCutMesh.
% All class outputs live in a temporary directory. Source repositories stay read-only.
oldPath = path;
cleanupPath = onCleanup(@() path(oldPath));
addpath(sourceRoot);
addpath(genpath(fullfile(sourceRoot,'utility')));
scratch = tempname; mkdir(scratch);
cleanupScratch = onCleanup(@() rmdir(scratch,'s'));
xp = struct('fileMeta',struct('timePoints',0,'stackResolution',[1 1 1], ...
    'filenameFormat','sample_%06d.tif','dataDir',scratch),'expMeta',struct(), ...
    'detectOptions',struct('ssfactor',1,'ofn_smoothply','mesh_%06d.ply'));
opts = struct('meshDir',scratch,'nU',11,'nV',19,'makeDirs',false,'lambda_err',0, ...
    'flipy',false,'timeUnits','min','spaceUnits','um');
tubi = TubULAR(xp,opts); tubi.setTime(0);
tubi.APDV.rot = eye(3); tubi.APDV.trans = [0 0 0]; tubi.APDV.resolution = 1;
tubi.flipy = false;
tubi.plotting.preview = false;
tubi.plotting.xyzlim_raw = [-5 5;-5 5;-5 5];
tubi.plotting.xyzlim_pix = tubi.plotting.xyzlim_raw;
tubi.plotting.xyzlim_um = tubi.plotting.xyzlim_raw;
tubi.plotting.xyzlim_um_buff = tubi.plotting.xyzlim_raw;
tubi.fullFileBase.uvcutMesh = fullfile(scratch,'uvcut_%06d.mat');
nu = 5; nv = 8;
[u,v] = ndgrid([0 .18 .43 .71 1],[0 .11 .25 .41 .56 .72 .87 1]);
baseUV = [u(:),v(:)]; theta = 2*pi*v(:);
faces = defineFacesRectilinearGrid(baseUV,nu,nv);
pairs = [((nv-1)*nu+1:nv*nu)',(1:nu)'];
normals = [zeros(numel(u),1),cos(theta),-sin(theta)];
normals(pairs(:,1),:) = normals(pairs(:,2),:);
field = [u(:).*cos(theta), u(:).^2+sin(theta), cos(2*theta)];
field(pairs(:,1),:) = field(pairs(:,2),:);
cases = cell(1,3);
for c = 1:3
    chart = baseUV;
    xyz = [4*u(:),2*cos(theta),-2*sin(theta)];
    ff = faces; pp = pairs;
    width = 1;
    if c > 1
        chart(:,1) = u(:)+.015*sin(theta).*sin(pi*u(:));
        chart(:,2) = v(:)+.06*sin(2*pi*u(:))+.007*sin(3*pi*u(:)).*sin(theta);
        chart(pairs(:,1),:) = chart(pairs(:,2),:)+[0 1];
        width = 1.7;
        chart(:,1) = width*chart(:,1);
        xyz = [4*u(:)+.3*sin(theta).*sin(pi*u(:)), ...
            (1+.2*sin(pi*u(:))).*cos(theta)+.4*u(:).^2, ...
            -(1+.12*sin(2*pi*u(:))).*sin(theta)+.1*u(:)];
    end
    if c == 3
        ff = ff(:,[1 3 2]); pp = pp(:,[2 1]);
    end
    xyz(pairs(:,1),:) = xyz(pairs(:,2),:);
    cutMesh = struct('v',xyz,'f',ff,'u',chart,'vn',normals,'pathPairs',pp,'umax',width);
    tubi.currentMesh.cutMesh = cutMesh;
    legacy = tubi.generateCurrentUVCutMesh(struct('overwrite',true));
    [tf,tuv,txyz,tnormals] = tileAnnularCutMesh(cutMesh,[1 1]);
    % Carry a separate vertex field through the same unchanged tiler.
    fieldMesh = cutMesh; fieldMesh.v = field;
    [~,~,tfield] = tileAnnularCutMesh(fieldMesh,[1 1]);
    [qu,qv] = ndgrid(linspace(0,width,tubi.nU),linspace(0,1,tubi.nV));
    query = [qu(:),qv(:)];
    [meshXYZ,~,~,~] = interpolate2Dpts_3Dmesh(tf,tuv,txyz,query);
    meshNormals = interpolate2Dpts_3Dmesh(tf,tuv,tnormals,query);
    meshNormals = meshNormals./vecnorm(meshNormals,2,2);
    meshField = interpolate2Dpts_3Dmesh(tf,tuv,tfield,query);
    scatteredField = zeros(size(meshField));
    for k = 1:3
        interp = scatteredInterpolant(tuv(:,1),tuv(:,2),tfield(:,k),'linear','nearest');
        scatteredField(:,k) = interp(query(:,1),query(:,2));
    end
    assert(all(isfinite(meshXYZ),'all'));
    cases{c} = struct('vertices',xyz,'faces',ff,'chart',chart,'pathPairs',pp, ...
        'normals',normals,'field',field,'nU',tubi.nU,'nV',tubi.nV,'width',width, ...
        'tiledFaces',tf,'tiledUV',tuv,'tiledXYZ',txyz,'tiledNormals',tnormals, ...
        'query',query,'meshXYZ',meshXYZ,'meshNormals',meshNormals,'meshField',meshField, ...
        'legacyXYZ',legacy.v,'legacyNormals',legacy.vn,'legacyUV',legacy.uv, ...
        'legacyFaces',legacy.f,'legacyPairs',legacy.pathPairs,'legacyField',scatteredField);
end
matlabVersion = version;
save(outputFile,'cases','matlabVersion','-v7');
end
