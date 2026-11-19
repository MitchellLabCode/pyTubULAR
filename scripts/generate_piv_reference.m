function generate_piv_reference(pivlabRoot, outputFile)
% Run unchanged local kernels extracted from the pinned PIVlab source.
% The supplied checkout is read only; extracted helpers live in a temp folder.
oldPath = path; restore = onCleanup(@() path(oldPath));
scratch = tempname; mkdir(scratch); cleanup = onCleanup(@() rmdir(scratch,'s'));
source = fileread(fullfile(pivlabRoot,'+piv','piv_FFTmulti.m'));
starts = regexp(source,'(?m)^function ');
names = {'meanzeropad','do_correlations','rescale_array','SUBPIXGAUSS'};
for k = 1:numel(names)
    found = false;
    for j = 1:numel(starts)
        stop = length(source); if j < numel(starts), stop = starts(j+1)-1; end
        block = source(starts(j):stop);
        header = regexp(block,'^[^\n]*','match','once');
        if contains(header,[names{k} '('])
            fid = fopen(fullfile(scratch,[names{k} '.m']),'w');
            fwrite(fid,block); fclose(fid); found = true; break
        end
    end
    assert(found,['Missing kernel: ' names{k}]);
end
addpath(scratch);
rng(781,'twister'); w = 32; cases = cell(2,3);
[x,y] = meshgrid(0:w-1,0:w-1);
a = zeros(w);
for k = 1:55
    cx = w*rand; cy = w*rand;
    ddx = mod(x-cx+w/2,w)-w/2; ddy = mod(y-cy+w/2,w)-w/2;
    a = a+(0.5+rand)*exp(-(ddx.^2+ddy.^2)/(2*1.1^2));
end
shifts = [3 -2;-.65 1.35;0 0];
freq = [0:w/2-1,-w/2:-1]/w; [fx,fy] = meshgrid(freq,freq);
for k = 1:3
    b = real(ifft2(fft2(a).*exp(-2*pi*1i*(fx*shifts(k,1)+fy*shifts(k,2)))));
    for mode = 1:2
        corr = do_correlations(a,b,mode==2,w);
        scaled = rescale_array(corr);
        [~,index] = max(scaled(:)); [py,px] = ind2sub(size(scaled),index);
        vector = SUBPIXGAUSS(scaled,w/2+1,px,py,1);
        assert(all(isfinite(vector),'all'));
        cases{mode,k} = struct('a',a,'b',b,'mode',mode,'shift',shifts(k,:), ...
            'correlation',corr,'displacement',vector);
    end
end
matlabVersion = version;
save(outputFile,'cases','matlabVersion','-v7');
end
