function I = sampleFromTensorProbabilityTable(x,n)
simult = true;
if isa(x,'double')
    x = FullTensor(x);
end

switch class(x)
    case 'FullTensor'
        w = x.data(:);
        if any(w<0)
            warning('negative probabilities')
            w = max(w,0);
        end
        
        i = randsample((1:prod(x.sz))',n,true,w);
        I = cell(1,x.order);
        [I{:}] = ind2sub(x.sz,i);
        I = [I{:}];


    case 'TreeBasedTensor'
        d = x.order;  
        sz = x.sz ; 
        p = cell(1,d);
        p{d} = x;
        for k=d-1:-1:1
            p{k} = squeeze(sum(p{k+1} , k+1 ) , k+1);
        end
        I = zeros(n,d);
        w = double(p{1});
        if any(w<0)
             warning('negative probabilities')
             w = max(w,0);
        end
        I(:,1) = randsample((1:sz(1))',n,true,w);  
        p{1} = double(evalAtIndices(p{1} , I(:,1)));
        for k=2:d
            if simult
                try
                w = double(p{k}.evalAtIndices(I(:,1:k-1),1:k-1));
                catch
                    simult = false;
                end
            end
            if ~simult
                fprintf('slow sequential sampling...')
                w = zeros(size(I,1),sz(k));
                for l=1:size(I,1)
                    w(l,:) = double(p{k}.evalAtIndices(I(l,1:k-1),1:k-1));
                end
                fprintf('done\n')
            end
            
            for i=1:n
                wi = w(i,:)'./p{k-1}(i);
                if any(wi<0)
                    warning('negative probabilities')
                    wi = max(wi,0);
                end
                I(i,k) = randsample((1:sz(k))',1,true,wi);  
            end
            p{k} = evalAtIndices(p{k} , I(:,1:k));
        end


    otherwise
        error('not implemented')

end


