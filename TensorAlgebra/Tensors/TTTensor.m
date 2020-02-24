% Class TTTensor: algebraic tensors in tensor train format

% Copyright (c) 2020, Anthony Nouy, Erwan Grelier, Loic Giraldi
% 
% This file is part of ApproximationToolbox.
% 
% ApproximationToolbox is free software: you can redistribute it and/or modify
% it under the terms of the GNU Lesser General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% ApproximationToolbox is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU Lesser General Public License for more details.
% 
% You should have received a copy of the GNU Lesser General Public License
% along with ApproximationToolbox.  If not, see <https://www.gnu.org/licenses/>.

classdef TTTensor < AlgebraicTensor
    
    properties
        cores    % cell array containing the cores of the representation
        order    % order of the tensor
        sz       % size of the tensor
        ranks    % TT-rank
        isOrth   % The flag is true if the representation of the tensor is orthogonal
    end
    
    methods
        function x = TTTensor(y)
            % Constructor for the class TTTensor
            %
            % x = TTTensor(c)
            % Returns a tensor x of order d in tensor train format
            % c: 1-by-d cell array containing the d cores, or TuckerLikeTensor
            
            if isa(y,'cell')
                x.cores = y;
                d = numel(y);
                x.order = d;
                x.ranks = zeros(1,d-1);
                x.sz = zeros(1,d);
                for mu = 1:d-1
                    x.ranks(mu) = y{mu}.sz(3);
                    x.sz(mu) = y{mu}.sz(2);
                end
                x.sz(d) = y{d}.sz(2);
                x.isOrth = false;
            elseif isa(y,'TTTensor')
                x.cores = y.cores;
                x.order = y.order;
                x.ranks = y.ranks;
                x.sz = y.sz;
                x.ranks = y.ranks;
                x.isOrth = y.isOrth;
            elseif isa(y, 'TuckerLikeTensor')
                xc = TTTensor(y.core);
                x = timesMatrix(xc, y.space.spaces);
            end
        end
        
        function x = treeBasedTensor(x)
            % Converts the TTTensor into a TreeBasedTensor
            
            tree = DimensionTree.linear(x.order);
            cores = x.cores;
            cores{1} = squeeze(cores{1},1);
            cores{end} = squeeze(cores{end},3);
            tensors = cell(1,tree.nbNodes);
            nod = tree.dim2ind(1);
            for mu=1:x.order
                tensors{nod} = cores{mu};
                nod = tree.parent(nod);
            end
            x = TreeBasedTensor(tensors,tree);
        end
        
        function n =  storage(x)
            n = sum(cellfun(@numel,x.cores));
        end
        
        function n =  sparseStorage(x)
            n = sum(cellfun(@nnz,x.cores));
        end
        
        function z = plus(x,y)
            z = x;
            d = z.order;
            z.ranks = x.ranks + y.ranks;
            z.sz = x.sz;
            z.cores{1} = cat(x.cores{1},y.cores{1},3);
            z.cores{d} = cat(x.cores{d},y.cores{d},1);
            for mu = 2:d-1
                z.cores{mu} = cat(x.cores{mu},y.cores{mu},[1 3]);
            end
            z.isOrth=false;
            %if x.isOrth ~= y.isOrth
            %    x.isOrth=false;
            %end
        end
        
        function z = uminus(z)
            z.cores{1} = -z.cores{1};
        end
        
        function z = minus(x,y)
            z = x + (-y);
        end
        
        function x = mtimes(x,a)
            if isa(x,'double')
                b = x;
                x = a;
                x = x*b;
            elseif isa(a,'double')
                x.cores{1} = x.cores{1}*a;
            else
                error('Wrong type of input arguments.');
            end
        end
        
        function x = subTensor(x,varargin)
            assert(nargin == 1+x.order, ...
                'Wrong number of input arguments.');
            for k = 1:x.order
                if ~strcmp(':',varargin{k})
                    x.cores{k} =  x.cores{k}.subTensor(':',varargin{k},':');
                end
            end
            x = updateProperties(x);
        end
        
        function x = times(x,y)
            for mu = 1:x.order
                szmu0 = x.cores{mu}.sz;
                szmu = x.cores{mu}.sz + y.cores{mu}.sz;
                szmu(2) = x.cores{mu}.sz(2);
                xmu = FullTensor.zeros(szmu);
                xmu(1:szmu0(1),:,1:szmu0(3)) = x.cores{mu};
                x.cores{mu} = xmu;
            end
        end
        
        function z = dot(x,y)
            z = timesTensor(x.cores{1},y.cores{1},[1 2],[1 2]);
            for mu = 2:x.order
                z = timesTensor(z,x.cores{mu},1,1);
                z = timesTensor(z,y.cores{mu},[1 2],[1 2]);
            end
            z = z.data;
        end
        
        function n = norm(x)
            if ~x.isOrth
                x = orth(x);
            end
            n = norm(x.cores{x.isOrth});
        end
        
        function xf = full(x)
            xf = x.cores{1};
            for mu = 2:x.order
                xf = timesTensor(xf,x.cores{mu},xf.order,1);
            end
            xf = squeeze(xf,[1 xf.order]);
        end
        
        function [x,dims] = squeeze(x,varargin)
            d = x.order;
            if nargin == 1
                dims = 1:d;
                dims(x.sz ~= 1) = [];
            else
                dims = varargin{1};
            end
            dims = sort(dims);
            remainingDims = 1:d;
            remainingDims(dims) = [];
            if isempty(remainingDims)
                xc = x.cores;
                x = reshape(xc{1}.data,[xc{1}.sz(1) xc{1}.sz(3)]);
                for mu = 2:d
                    x = x*reshape(xc{mu}.data,[xc{mu}.sz(1) xc{mu}.sz(3)]);
                end
            else
                xc = x.cores;
                mu = 1;
                while (mu <= numel(dims)) && (mu == dims(mu))
                    xmu = squeeze(xc{mu},2);
                    xmu = xmu.data;
                    xc{mu+1} = timesMatrix(xc{mu+1},xmu,1);
                    mu = mu + 1;
                end
                dims(1:mu-1) = [];
                mu = d;
                k = numel(dims);
                while (k>0) &&(mu == dims(k))
                    xmu = squeeze(xc{mu},2);
                    xmu = xmu.data;
                    xc{mu-1} = timesMatrix(xc{mu-1},xmu',3);
                    mu = mu - 1;
                    k = k-1;
                end
                dims(numel(dims):-1:k+1) = [];
                if ~isempty(dims)
                    mu0 = dims(1)-1;
                    for mu = dims(1):dims(end)
                        if any(remainingDims == mu)
                            mu0 = mu;
                        else
                            xmu = squeeze(x.cores{mu},2);
                            xmu = xmu.data;
                            xc{mu0} = timesMatrix(xc{mu0},xmu',3);
                        end
                    end
                end
                xc = xc(remainingDims);
                x = TTTensor(xc);
            end
        end
        
        function x = insertDims(x,dims)
            % Inserts new dimensions
            d = x.order+numel(dims);
            cdims = setdiff(1:d,dims);
            xc = cell(d,1);
            xc(cdims) = x.cores;
            for mu = dims
                switch mu
                    case {1,d}
                        xc{mu} = FullTensor(1,3,[1 1 1]);
                    otherwise
                        id = FullTensor(eye(xc{mu-1}.sz(3)));
                        id = reshape(id,[size(id,1),1,size(id,1)]);
                        xc{mu} = id;
                end
            end
            x = TTTensor(xc);
        end
        
        function x = timesVector(x,V,varargin)
            if isa(V,'double')
                V = {V};
            end
            V = cellfun(@(v) v',V,'uniformoutput',false);
            x = timesMatrix(x,V,varargin{:});
            x = squeeze(x,varargin{:});
        end
        
        function [x] = timesDiagMatrix(x,M,varargin)
            if nargin == 2
                dims = 1:x.order;
            else
                dims = varargin{1};
            end
            M = mat2cell(M,size(M,1),ones(1,size(M,2)));
            M = cellfun(@(d) diag(d) , M ,'UniformOutput',0);
            for k = 1:numel(dims)
                mu = dims(k);
                x.cores{mu} = x.cores{mu}.timesMatrix(M(k),2);
                x.sz(mu) = x.cores{mu}.sz(2);
            end
            x.isOrth = 0;
        end
        
        function [x] = timesMatrix(x,M,varargin)
            if nargin == 2
                dims = 1:x.order;
            else
                dims = varargin{1};
            end
            for k = 1:numel(dims)
                mu = dims(k);
                x.cores{mu} = timesMatrix(x.cores{mu},M(k),2);
                x.sz(mu) = x.cores{mu}.sz(2);
            end
            x.isOrth = 0;
        end
        
        function x = timesTensor(x,y,xd,yd)
            % Contraction of two tensors along given dimensions
            %
            % z = timesTensor(x,y,xDims,yDims)
            % See also FullTensor/timesTensor
            
            x = timesTensor(full(x),full(y),xd,yd);
        end
        
        function z = orth(x,lambda)
            % Orthogonalization of the representation
            %
            % z = orth(x,lambda)
            % right orthogonalization of the cores 1 to lambda-1
            % left orthogonalization of the cores lambda+1 to x.order
            
            z = x;
            if nargin == 1
                lambda = 1;
            end
            d = x.order;
            U = x.cores;
            % Left to right orth
            for mu = 1:lambda-1
                [U{mu},r] = orth(U{mu},3);
                U{mu+1} = timesMatrix(U{mu+1},{r},1);
            end
            % Right to left orth
            for mu = d:-1:lambda+1
                [U{mu},r] = orth(U{mu},1);
                U{mu-1} = timesMatrix(U{mu-1},{r},3);
            end
            z.cores = U;
            z.isOrth = lambda;
            for mu = 1:d-1
                z.ranks(mu) = z.cores{mu}.sz(3);
            end
        end
        
        function [x] = cat(x,y)
            d = x.order;
            xc = cell(d,1);
            for mu = 1:d
                xc{mu} = cat(x.cores{mu},y.cores{mu});
            end
            xc{1}.data = sum(xc{1}.data,1);
            xc{1}.sz(1) = 1;
            xc{d}.data = sum(xc{d}.data,3);
            xc{d}.sz(3) = 1;
            x.cores = xc;
            x.ranks = x.ranks+y.ranks;
            x.sz = x.sz+y.sz;
            x.isOrth = false;
        end
        
        function [x] = kron(x,y)
            for mu = 1:x.order
                x.cores{mu} = kron(x.cores{mu},y.cores{mu});
            end
            x.sz = x.sz.*y.sz;
            x.ranks = x.ranks.*y.ranks;
            x.isOrth = false;
        end
        
        function z = dotWithRankOneMetric(x,y,M)
            yM = M;
            for mu = 1:x.order
                yM{mu} = y.cores{mu}.timesMatrix(M(mu),2);
            end
            z = timesTensor(yM{1},x.cores{1},[1 2],[1 2]);
            for mu = 2:x.order
                yM{mu} = timesMatrix(yM{mu},{z.data'},1);
                z = timesTensor(yM{mu},x.cores{mu},[1 2],[1 2]);
            end
            z = z.data;
        end
        
        function z = timesTensorTimesMatrixExceptDim(x,y,M,dim)
            Z = reduceDotWithRankOneMetricAtCore(x,y,M,dim);
            z = y.cores{dim};
            if dim ~= 1
                z = timesMatrix(z,{Z{1}'},1);
            end
            % right to left contraction
            if dim ~= x.order
                z = timesMatrix(z,{Z{3}'},3);
            end
            % final contraction
            z = timesTensor(z,x.cores{dim},[1 3],[1 3]);
            z = z.data';
        end
        
        function c = reduceDotWithRankOneMetricAtCore(x,y,M,dim)
            yM = y.cores;
            d = x.order;
            dims = 1:d;
            dims(dim) = [];
            for mu = dims
                yM{mu} = timesMatrix(y.cores{mu},M(mu),2);
            end
            % left to right contraction
            if dim ~= 1
                cl = timesTensor(yM{1},x.cores{1},[1 2],[1 2]);
                cl = reshape(cl,[yM{1}.sz(3),x.cores{1}.sz(3)]);
                cl = cl.data;
                for mu = 2:dim-1
                    yM{mu} = timesMatrix(yM{mu},{cl'},1);
                    cl = timesTensor(yM{mu},x.cores{mu},[1 2],[1 2]);
                    cl = reshape(cl,[yM{mu}.sz(3),x.cores{mu}.sz(3)]);
                    cl = cl.data;
                end
            else
                cl = [];
            end
            % right to left contraction
            if dim ~= d
                cr = timesTensor(yM{d},x.cores{d},[2 3],[2 3]);
                cr = cr.data;
%                 cr = reshape(cr,size(yM{1}.a,1),size(x.cores{1}.a,1));
                for mu = d-1:-1:dim+1
                    yM{mu} = timesMatrix(yM{mu},{cr'},3);
                    cr = timesTensor(yM{mu},x.cores{mu},[2 3],[2 3]);
                    cr = cr.data;
%                     cr = reshape(cr,size(yM{mu}.a,3),size(x.cores{mu}.a,3));
                end
            else
                cr = [];
            end
            c = {cl,M{dim}',cr};
        end
        
        function c = reduceDotWithRankOneMetricAt2Cores(x,y,M,dim)
            yM = y.a;
            d = x.order;
            dims = 1:d;
            dims([dim,dim+1]) = [];
            for mu = dims
                yM{mu} = timesMatrix(y.a{mu},M(mu),2);
            end
            % left to right contraction
            if dim ~= 1
                cl = timesTensor(yM{1}.a,x.cores{1}.a,[1 2],[1 2]);
                cl = reshape(cl,size(yM{1}.a,3),size(x.cores{1}.a,3));
                for mu = 2:dim-1
                    yM{mu} = timesMatrix(yM{mu},{cl'},1);
                    cl = timesTensor(yM{mu}.a,x.cores{mu}.a,[1 2],[1 2]);
                    cl = reshape(cl,size(yM{mu}.a,3),size(x.cores{mu}.a,3));
                end
            else
                cl = [];
            end
            % right to left contraction
            if dim ~= d-1
                cr = timesTensor(yM{d}.a,x.cores{d}.a,[2 3],[2 3]);
                %cr = reshape(cr,size(yM{1}.a,1),size(x.cores{1}.a,1));
                for mu = d-1:-1:dim+2
                    yM{mu} = timesMatrix(yM{mu},{cr'},3);
                    cr = timesTensor(yM{mu}.a,x.cores{mu}.a,[2 3],[2 3]);
                    %cr = reshape(cr,size(yM{mu}.a,3),size(x.cores{mu}.a,3));
                end
            else
                cr = [];
            end
            c = {cl,M{dim}',M{dim+1}',cr};
        end
        
        function s = evalDiag(x,dims)
            if nargin==1 || numel(dims)==x.order
                n = x.cores{1}.sz(2);
                s=ones(1,n);
                for mu=1:x.order
                    s = repmat(s,[1 1 size(x.cores{mu},3)]);
                    s = sum(s.*x.cores{mu}.data,1);
                    s = permute(s,[3,2,1]);
                end
                s=s(:);
            else
                dims = sort(dims);
                if ~all(dims(2:end)-dims(1:end-1))
                    error('Requires adjacent dimensions.')
                end
                
                n = x.cores{dims(1)}.sz(2);
%                 s = ones(x.cores{dims(1)}.sz(1),n,1,1);
                s = x.cores{dims(1)}.data;
                for mu=dims(2:end)
                    s = repmat(s,[1 1 1 x.cores{mu}.sz(3)]);
                    s2 = x.cores{mu}.data;
                    s2 = repmat(s2,[1 1 1 size(s,1)]);
                    s2 = permute(s2,[4,2,1,3]);
                    s = sum(s.*s2,3);
                    s = permute(s,[1,2,4,3]);
                end
                leftdims = 1:dims(1)-1;
                rightdims = dims(end)+1:x.order;
                siz = [size(x.cores{dims(1)},1),n,size(x.cores{dims(end)},3)];
                x.cores{dims(1)}=FullTensor(s,3,siz);
                x.cores = x.cores([leftdims,dims(1),rightdims]);
                x.isOrth=false;
                s = updateProperties(x);
            end
        end
        
        function alpha = evalDiagOnLeft(x,k)
            n = size(x.cores{1},2);
%             alpha=ones(n,1);
%             for mu=1:k-1
%                 alpha = repmat(alpha,[1 1 size(x.cores{mu}.data,3)]);
%                 alpha = sum(alpha.*permute(x.cores{mu}.data,[2 1 3]),2);
%                 alpha = alpha(:,:);
%
%             end
            alpha = FullTensor.ones([n,1]);
            for mu=1:k-1
                alpha = timesTensorEvalDiag(alpha,x.cores{mu},2,1,1,2);
            end
        end
        
        function beta = evalDiagOnRight(x,k)
            d=length(x.cores);
            n = size(x.cores{d},2);
            beta = FullTensor.ones([n,1]);
            for mu=d:-1:k+1
                beta = timesTensorEvalDiag(beta,x.cores{mu},2,3,1,2);
            end
%             beta=ones(n,1);
%             for mu=d:-1:k+1
%                 beta = repmat(beta,[1 1 size(x.cores{mu}.data,1)]);
%                 beta = sum(beta.*permute(x.cores{mu}.data,[2 3 1]),2);
%                 beta = beta(:,:);
%             end
        end
        
        function [g,ind] = parameterGradientEvalDiag(f,mu,H)
            % Returns the diagonal of the gradient of the tensor with respect to a given parameter
            %
            % [g,ind] = parameterGradientEvalDiag(x,mu)
            % x: TTTensor
            % mu: index of the parameter (integer from 1 to x.order)
            
            alpha = evalDiagOnLeft(f,mu);
            beta = evalDiagOnRight(f,mu);
            
            ind = [mu ; 3];
            
            if mu == 1
                alpha = FullTensor.ones([beta.sz(1),1]);
            elseif mu == f.order
                beta = FullTensor.ones([alpha.sz(1),1]);
            end
            
            if nargin == 3
                g = outerProductEvalDiag(alpha,FullTensor(H{mu}),1,1);
            else
                g = outerProductEvalDiag(alpha,FullTensor(eye(f.sz(mu))),[],[],true);
            end
            g = outerProductEvalDiag(g,beta,1,1);
        end
        
        
        function [x,pelem] = permute(x,dims,tol)
            % Permutes dimensions by applying succesive elementary permutations between consecutive dimensions
            %
            % function [x,pelem] = permute(x,dims,tol)
            % pelem: sequence of elementary permutations
            % tol: relative precision (1e-15 by default)
            
            pelem = decomposePermutation(dims);
            if nargin<3 || isempty(tol)
                tol = 1e-15;
            end
            tol = max(1e-15,tol/size(pelem,1));
            
            for k=1:size(pelem,1)
                x = permuteConsecutiveDims(x,pelem(k,1),tol);
            end
        end
        
        function x = permuteConsecutiveDims(x,mu,tol)
            % Permutes two consecutive dimensions (using SVD)
            %
            % function x = permuteConsecutiveDims(x,mu,tol)
            % permute dimensions mu and mu+1 of the tensor
            % tol: relative precision (1e-14 by default)
            
            if nargin<3 || isempty(tol)
                tol=1e-14;
            end
            tr=Truncator('tolerance',tol);
            x = orth(x,mu);
            w = timesTensor(x.cores{mu},x.cores{mu+1},3,1);
            w2 = permute(w,[1,3,2,4]);
            w2 = reshape(w2,[w.sz(1)*w.sz(3),w.sz(2)*w.sz(4)]);
            w2=tr.truncate(w2);
            r2 = w2.core.sz(1);
            x.cores{mu} = reshape(FullTensor(w2.space.spaces{1}),[w.sz(1),w.sz(3),r2]);
            x.cores{mu+1} = reshape(FullTensor(w2.space.spaces{2}),[w.sz(2),w.sz(4),r2]);
            x.cores{mu+1} = permute(x.cores{mu+1},[3,1,2]);
            x.cores{mu+1} = timesMatrix(x.cores{mu+1},diag(w2.core.data),1);
            x.ranks(mu)=r2;
            x.sz(mu:mu+1)=x.sz(mu+1:-1:mu);
        end
        
        function x=updateProperties(x)
            % Update properties order, sz and ranks from the property cores
            
            x.order = numel(x.cores);
            x.sz = cellfun(@(xs)size(xs,2),x.cores)';
            x.ranks = cellfun(@(xs)size(xs,3),x.cores(1:end-1))';
        end
        
        function [x,p,pelem] = optimizePermutationGlobal(x,tol,N)
            % Global optimization over permutations for minimizing the representation rank
            %
            % [x,p]=optimizePermutationGlobal(x,tol,N)
            % x: TTTensor
            % tol: tolerance
            % N: number of iterations
            
            d = x.order;
            p = 1:d;
            s = storage(x);
            pelem=zeros(0,2);
            type = 2;
            if type==1
                mu1=1;
            end
            for i=1:N
                switch type
                    case 1
                        c = randi(3);
                        if c==1
                            mu2=mu1+1;
                        elseif c==2
                            mu1 = randi(x.order-1);
                            mu2=mu1+1;
                        elseif c==3
                            mu1 = randi(x.order-1);
                            mu2 = setdiff(1:x.order,mu1);
                            pr = 1./abs(mu1-mu2).^2;
                            mu2 = DiscreteRandomVariable(mu2,pr/sum(pr));
                            mu2 = random(mu2,1);
                        end
                    case 2
                        pr = 1./(1:d-1).^2;
                        di = random(DiscreteRandomVariable(1:d-1,pr/sum(pr)));
                        mu1 = random(DiscreteRandomVariable((1:d-di).'));
                        mu2 = mu1 + di;
                end
                
                dims = 1:x.order;
                dims(mu1) = mu2;
                dims(mu2) = mu1;
                [xnew,~] = permute(x,dims,tol);
                snew = storage(xnew);
                if snew < s
                    x = xnew;
                    p([mu1,mu2]) =  p([mu2,mu1]);
                    pelem = [pelem;mu1,mu2];
                    s = snew;
                end
            end
        end
        
        function [x,p,pelem]=optimizePermutation(x,tol,N,tolselect)
            % Optimization over permutations for minimizing the representation rank
            %
            % [x,p,pelem]=optimizePermutation(x,tol,N)
            % find a permutation of 1:d that minimizes the representation
            % rank of x
            % p: permutation of 1:d
            % pelem: sequence of elementary permutations (permuting two consecutive dimensions)
            
            if nargin==1 || isempty(tol)
                tol=1e-14;
            end
            if nargin<3 || isempty(N)
                N=10;
            end
            if nargin<4
                tolselect=tol;
            end
            if tolselect<tol
                error('Tolerance for truncation must be higher than tolerance for selection.')
            end
            tr=Truncator('tolerance',tol);
            p = 1:x.order;
            pelem=zeros(0,2);
            for i=1:N
                isperm = false;
                for mu=1:x.order-1
                    x = orth(x,mu);
                    w = timesTensor(x.cores{mu},x.cores{mu+1},3,1);
                    w1 = reshape(w,[w.sz(1)*w.sz(2),w.sz(3)*w.sz(4)]);
                    tr.tolerance=tolselect;
                    tr.maxRank=x.ranks(mu);
                    w1select=tr.truncate(w1);
                    r1 = w1select.core.sz(1);
                    w2 = permute(w,[1,3,2,4]);
                    w2 = reshape(w2,[w.sz(1)*w.sz(3),w.sz(2)*w.sz(4)]);
                    w2select=tr.truncate(w2);
                    r2 = w2select.core.sz(1);
                    c1 = w.sz(1)*w.sz(2)*r1 + r1*w.sz(3)*w.sz(4);
                    c2 = w.sz(1)*w.sz(3)*r2 + r2*w.sz(2)*w.sz(4);
                    c0 = storage(x);
                    if c2<c1
                        tr.tolerance=tol;
                        tr.maxRank=x.ranks(mu);
                        w2=tr.truncate(w2);
                        r2 = w2.core.sz(1);
                        x.cores{mu} = reshape(FullTensor(w2.space.spaces{1}),[w.sz(1),w.sz(3),r2]);
                        x.cores{mu+1} = reshape(FullTensor(w2.space.spaces{2}),[w.sz(2),w.sz(4),r2]);
                        x.cores{mu+1} = permute(x.cores{mu+1},[3,1,2]);
                        x.cores{mu+1} = timesMatrix(x.cores{mu+1},diag(w2.core.data),1);
                        x.ranks(mu)=r2;
                        p(mu:mu+1)=p(mu+1:-1:mu);
                        x.sz(mu:mu+1)=x.sz(mu+1:-1:mu);
                        pelem = [pelem;mu,mu+1];
                        isperm=true;
                    end
                end
                if ~isperm || storage(x)==c0
                    break
                end
            end
        end
        
        function sv = singularValues(x)
            % Returns sets of singular values
            % sv = singularValues(x)
            % sv is a cell of length x.order-1, such that sv{mu} is the set of singular values of the {1,...,mu} matricization of x
            
            if x.order == 2
                x = full(x);
                sv = {svd(x.data)};
            else
                d = x.order;
                sv = cell(d,1);
                for mu = 1:d-1
                    x = orth(x,mu);
                    sz = x.cores{mu}.sz;
                    xmu = reshape(x.cores{mu},[sz(1)*sz(2),sz(3)]);
                    sv{mu} = svd(xmu.data);
                end
            end
        end
        
        function r = representationRank(x)
            % Returns the representation TT-rank
            
            d = x.order;
            r = zeros(1,d-1);
            for mu = 1:d-1
                r(mu) = size(x.cores{mu},3);
            end
        end
        
        function r  = rank(x)
            % Returns the TT-rank of the tensor
            
            sv = singularValues(x);
            %             r = cellfun(@nnz,sv);
            r = cellfun(@(x) nnz(x / max(x) > eps),sv);
        end
    end
    
    methods (Static)
        function x = create(generator,sz,ranks)
            % Creates a tensor of size sz and TT-rank ranks with entries generated using a given generator function x = create(generator,sz,ranks)
            
            d = numel(sz);
            if length(ranks)==1
                ranks = repmat(ranks,d-1,1);
            end
            r = ranks(:);
            r = [1;r];
            rc = circshift(r,-1);
            x = cell(d,1);
            for mu = 1:d
                x{mu} = FullTensor.create(generator,[r(mu),sz(mu),rc(mu)]);
            end
            x = TTTensor(x);
        end
        
        function x = randn(sz,ranks)
            % Creates a tensor of size sz and TT-rank ranks with i.i.d. entries drawn according to the standard gaussian distribution
            % x = randn(sz,ranks)
            
            if nargin==1
                ranks = randi(10,numel(sz)-1,1);
            end
            x = TTTensor.create(@randn,sz,ranks);
        end
        
        function x = rand(sz,ranks)
            % Creates a tensor of size sz and TT-rank ranks with i.i.d. entries drawn according the uniform distribution on (0,1)
            % x = rand(sz,ranks)
            
            if nargin==1
                ranks = randi(10,numel(sz)-1,1);
            end
            x = TTTensor.create(@rand,sz,ranks);
        end
        
        function x = zeros(sz,ranks)
            % Creates a tensor of size sz and TT-rank ranks with zero entries
            % x = zeros(sz,ranks)
            
            if nargin==1
                ranks=1;
            end
            x = TTTensor.create(@zeros,sz,ranks);
        end
        
        function x = ones(sz,ranks)
            % Creates a tensor of size sz and TT-rank ranks with entries equal to one
            % x = ones(sz,ranks)
            
            if nargin==1
                ranks=1;
            end
            x = TTTensor.create(@ones,sz,ranks);
        end
    end
end