function generate_sphi_reference(sourceRoot, uvFixture, outputFile)
% Stage-level references using unchanged MATLAB kernels, not the full class.
% ringpathsGridSampling's second output has a dimension/indexing defect;
% request ONLY its first output and sample XYZ with the unchanged interpolator.
oldPath = path; cleanup = onCleanup(@() path(oldPath));
addpath(genpath(fullfile(sourceRoot,'utility')));
input = load(uvFixture); cases = cell(2,3);
nu = 9; nv = 17;
for c = 1:3
    src = input.cases{c};
    m = struct('v',src.vertices,'f',src.faces,'u',src.chart,'pathPairs',src.pathPairs);
    [tf,tuv,tv] = tileAnnularCutMesh(m,[1 1]);
    us = linspace(0,src.width,nu)'; vs = linspace(0,1,nv)';
    [uq,vq] = ndgrid(us,vs);
    crude = reshape(interpolate2Dpts_3Dmesh(tf,tuv,tv,[uq(:),vq(:)]),[nu nv 3]);
    legacyCrudeS = ringpathsGridSampling(us,vs,tf,tuv,tv);
    for mode = 1:2
        cols = 1:nv; if mode == 1, cols = 1:nv-1; end
        ds = vecnorm(diff(crude,1,1),2,3);
        crudeS = cumsum([0;mean(ds(:,cols),2)]);
        if mode == 2, assert(max(abs(crudeS-legacyCrudeS)) < 1e-12); end
        [uspace,~] = equidistantSampling1D(linspace(0,1,nu)',crudeS,nu,'linear');
        [uq,vq] = ndgrid(src.width*uspace,vs);
        query = [uq(:),vq(:)];
        first = interpolate2Dpts_3Dmesh(tf,tuv,tv,query);
        first(end-nu+1:end,:) = first(1:nu,:); % exact declared seam
        curves = reshape(first,[nu nv 3]); hoops = zeros(size(curves));
        for k = 1:nu
            hoops(k,:,:) = resampleCurvReplaceNaNs(squeeze(curves(k,:,:)),nv,true);
        end
        [~,~,radii,centerS,centers] = srFromDVCurves(hoops);
        ds = vecnorm(diff(hoops,1,1),2,3);
        ringS = cumsum([0;mean(ds(:,cols),2)]);
        offsets = zeros(nu,1);
        if c > 1, offsets = .08*sin(linspace(0,pi,nu))'; end
        [sq,pq] = ndgrid(ringS,vs); pq = pq-offsets;
        chart = [sq(:),pq(:)];
        faces = defineFacesRectilinearGrid(query,nu,nv);
        ab = src.chart(src.faces(1,2),:)-src.chart(src.faces(1,1),:);
        ac = src.chart(src.faces(1,3),:)-src.chart(src.faces(1,1),:);
        if det([ab;ac]) < 0, faces = faces(:,[1 3 2]); end
        pairs = [(1:nu)',(nu*(nv-1)+1:nu*nv)'];
        intermediate = struct('v',first,'f',faces,'u',chart,'pathPairs',pairs);
        [ff,uv,vv] = tileAnnularCutMesh(intermediate,[2 2]);
        [sq,pq] = ndgrid(linspace(0,ringS(end),nu),vs);
        sphi = [sq(:),pq(:)];
        final = interpolate2Dpts_3Dmesh(ff,uv,vv,sphi);
        assert(all(isfinite(final),'all'));
        cases{mode,c} = struct('vertices',src.vertices,'faces',src.faces, ...
            'chart',src.chart,'pathPairs',src.pathPairs,'nU',nu,'nV',nv, ...
            'mode',mode,'crudeS',crudeS,'uSamples',uspace,'firstXYZ',first, ...
            'hoops',hoops,'centers',centers,'centerS',centerS,'radii',radii, ...
            'ringS',ringS,'offsets',offsets,'sphi',sphi,'finalXYZ',final);
    end
end
% Original custom endcap deletion helper on a closed polygonal cylinder.
[u,v] = ndgrid(linspace(0,3,5),linspace(0,2*pi,9));
xyz = [u(:),cos(v(:)),-sin(v(:))];
faces = defineFacesRectilinearGrid([u(:),v(:)],5,9);
faces(faces > 40) = faces(faces > 40)-40; xyz = xyz(1:40,:);
xyz = [xyz;0 0 0;3 0 0];
for k = 0:7
    next = mod(k+1,8);
    faces = [faces;41,1+5*k,1+5*next;42,5+5*next,5+5*k]; %#ok<AGROW>
end
[openFaces,openXYZ,vertexSources,sourceToOpen] = remove_vertex_from_mesh(faces,xyz,[41 42]);
caps = struct('vertices',xyz,'faces',faces,'openFaces',openFaces,'openXYZ',openXYZ, ...
    'vertexSources',vertexSources,'sourceToOpen',sourceToOpen);
matlabVersion = version;
save(outputFile,'cases','caps','matlabVersion','-v7');
end
