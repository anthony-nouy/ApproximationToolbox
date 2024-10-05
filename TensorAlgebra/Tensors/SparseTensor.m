% Class SparseTensor: sparse algebraic tensors

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

classdef SparseTensor < AlgebraicTensor
    
    properties
        order      % order of the tensor
        sz         % size of the tensor
        isOrth = 1 % flag not useful for this class
        indices    % object of class MultiIndices
        data       % contains the value of the tensor at entries indices
    end
    
    methods
        
        function x = SparseTensor(varargin)
            % x = SparseTensor(data)
            % data: full or sparse array
            %
            % x = SparseTensor(data,indices,sz)
            % indices: MultiIndices with n elements
            % data: full array of size nx1
            % sz: size of the tensor
            
            switch nargin
                case 0
                    x.data = [];
                    x.order = 0;
                    x.indices = [];
                    x.sz = [];
                case 1
                    y = varargin{1};
                    
                    if isa(y,'double')
                        x.order = ndims(y);
                        x.sz = size(y);
                        
                        rep = find(y);
                        x.data = full(y(rep));
                        x.indices = MultiIndices.ind2sub(x.sz,rep);
                        
                    elseif isa(y,'SparseTensor')
                        x.data = y.data;
                        x.order = y.order;
                        x.sz = y.sz;
                        x.indices = y.indices;
                    else
                        error('Wrong type of input argument.');
                    end
                    
                case 3
                    if ~isa(varargin{2},'MultiIndices')
                        error('Second argument must be a MultiIndices')
                    end
                    x.indices = varargin{2};
                    x.order = ndims(x.indices);
                    x.sz = varargin{3}(:)'; % make sure it is a row
                    x.data = varargin{1}(:);
                    if length(x.data)~=cardinal(x.indices)
                        error('data and indices must have the same number of elements.')
                    end
                otherwise
                    error('Wrong number of input arguments.');
            end
        end
        
        function y = storage(x)
            y = numel(x);
        end
        
        function y = sparseStorage(x)
            y = nnz(x);
        end
        
        function n = nnz(x)
            % NNZ - Number of nonzero coefficients of the tensor.
            %
            % SEE ALSO sparseStorage
            
            n = cardinal(x.indices);
        end
        
        function n = ndims(x)
            n = length(x.sz);
        end
        
        function y = full(x)
            y = zeros(x.sz);
            y = FullTensor(y);
            I = sub2ind(x.indices, x.sz);
            y.data(I) = x.data;
        end
        
        function y = double(x)
            if ~all(x.sz(3:end)==1)
                error('nd sparse arrays are not allowed for d>2.')
            end
            y = sparse(x.sz(1),x.sz(2));
            rep = sub2ind(x.sz,x.indices);
            y(rep) = x.data;
        end
        
        function s = evalAtIndices(x,I,varargin)
            % EVALATINDICES - evaluation of tensor components
            % s = evalAtIndices(x,I)
            % s(k) = x(I(k,1),I(k,2),...,I(k,d)), 1\le k \le N
            % I is an array of N-by-x.order or a MultiIndices
            %
            % s = evalAtIndices(x,I,dims)
            % partial evaluation
            % I is an array of size N-by-M, with M=numel(dims), or a
            % MultiIndices
            % s is a tensor of size N-by-n_1-...n_{d'} with d'=x.order-M
            % Up to a permutation (placing dimensions dims on the left)
            % s(k,i_1,...,i_d') = x(I(k,1),I(k,2),...,I(k,M),i_1,...,i_d'), 1\le k \le N
            
            if isa(I,'MultiIndices')
                I = I.array;
            end
            if nargin==2
                J = x.indices.array;
                [locI,locJ] = ismember(I,J,'rows');
                s = sparse(numel(locI),1);
                s(locI) = x.data(locJ);
            else
                J = x.indices.array;
                %[locI,locJ] = ismember(I,J(:,dims),'rows');
                %s = sparse(...);
                error('Method not implemented.')
            end
            
            
        end
        
        function x = squeeze(x,varargin)
            % SQUEEZE - remove dimensions of a tensor
            % s = squeeze(x)
            % remove dimensions with size 1
            % s = squeeze(x,dims)
            % remove dimensions dims
            
            error('Method not implemented.')
        end
        
        
        function x = plus(x,y)
            I = addIndices(x.indices,y.indices);
            [~,repx]= intersectIndices(I,x.indices);
            [~,repy]= intersectIndices(I,y.indices);
            data = zeros(numel(I),1);
            data(repx) = data(repx) + x.data;
            data(repy) = data(repy) + y.data;
            x = SparseTensor(data,I,x.sz);
        end
        
        function x = minus(x,y)
            error('Method not implemented.')
        end
        
        function x = uminus(x)
            error('Method not implemented.')
        end
        
        function x = mtimes(x,s)
            error('Method not implemented.')
        end
        
        function x = timesVector(x,V,varargin)
            
            switch nargin
                case 2
                    assert(isa(V,'cell'),'M should be a cell.');
                    assert(numel(V) == x.order,'Wrong number of cells.');
                    dims = 1:x.order;
                case 3
                    dims = varargin{1};
                    if isa(V,'double')
                        V = {V};
                    end
                    assert(numel(V) == numel(dims),'Wrong number of cells.');
                otherwise
                    error('Wrong number of input arguments.');
            end
            
            V = cellfun(@(x) reshape(x,length(x),1), V, 'UniformOutput', false);
            
            x.sz(dims) = [];
            
            for i = 1:length(dims)
                a = x.data.*V{i}(x.indices.array(:,dims(i)));
                
                x.indices.array(:,dims(i)) = [];
                x.indices.array = x.indices.array(a~=0,:);
                [x.indices.array, ~, ic] = unique(x.indices.array, 'rows');
                
                a = nonzeros(a);
                
                x.data = accumarray(ic,a);
                
                dims = dims - (dims > dims(i));
            end
            
            x.order = x.order - length(V);
            
            if isempty(x.sz)
                x = x.data;
            end
        end
        
        function x = timesMatrix(x,M,varargin)
            switch nargin
                case 2
                    assert(isa(M,'cell'),'M should be a cell.');
                    assert(numel(M) == x.order,'Wrong number of cells.');
                    dims = 1:x.order;
                case 3
                    dims = varargin{1};
                    if isa(M,'double')
                        M = {M};
                    end
                    assert(numel(M) == numel(dims),'Wrong number of cells.');
                otherwise
                    error('Wrong number of input arguments.');
            end
            
            M = M(:);
            k = 1;
            for mu = dims
                permDims = [mu,1:mu-1,mu+1:x.order];
                x = permute(x,permDims);
                if x.order == 1
                    x.sz(2) = 1;
                end
                [x1,x2] = ind2sub([x.sz(1),prod(x.sz(2:end))],sub2ind(x.sz,MultiIndices(x.indices.array(x.data ~= 0,:))));
                [x2u, ~, x2uind] = unique(x2);
                a = M{k}*sparse(x1,x2uind,nonzeros(x.data),x.sz(1),max(x2uind));
                [y1,y2] = find(a);
                x.sz(1) = size(M{k},1);
                ind = sub2ind([x.sz(1),prod(x.sz(2:end))],y1,reshape(x2u(y2),size(y1)));
                x.indices = MultiIndices.ind2sub(x.sz,ind);
                x.data = nonzeros(a(:));
                x = ipermute(x,permDims);
                k = k+1;
            end
        end
        
        function y = timesMatrixEvalDiag(x,H)
            y = H{1}(:,x.indices.array(:,1));
            for k=2:x.order
                y = y.*H{k}(:,x.indices.array(:,k));
            end
            y = y*x.data;
        end
        
        function x = permute(x,dims)
            % Permutation of the dimensions of the tensor
            %
            % x = permute(x,dims)
            % Permutes the dimensions of x to match the ones in dims
            % x: SparseTensor
            % dims: d-by-1 or 1-by-d double
            %
            % See also SparseTensor/ipermute
            
            x.indices.array = x.indices.array(:,dims);
            x.sz = x.sz(dims);
            x.sz = x.sz(:)';
        end
        
        function x = ipermute(x,dims)
            % Inverse permutation of the dimensions of the tensor
            %
            % x = ipermute(x,dims)
            % Performs the inverse permutation of f, which was so it matched the dimensions in dims
            % x: SparseTensor
            % dims: d-by-1 or 1-by-d double
            %
            % See also SparseTensor/permute
            
            idims(dims) = 1:numel(dims);
            x.indices.array = x.indices.array(:,idims);
            x.sz = x.sz(idims);
            x.sz = x.sz(:)';
        end
        
        function x = reshape(x,sz)
            % Reshapes the tensor
            %
            % y = reshape(x,sz)
            % x: SparseTensor
            % sz: array of integers
            % y: SparseTensor of order numel(sz) and size sz
            
            sz = sz(:)';
            ind = sub2ind(x.sz,x.indices) ;
            x.indices = MultiIndices.ind2sub(sz,ind) ;
            x.sz = sz;
            x.order = numel(sz);
        end
        
        function x = timesDiagMatrix(x,M,varargin)
            % Contraction with a set of diagonal matrices
            %
            % x = timesDiagMatrix(x,M,varargin)
            % If M is a cell array of length x.order, contracts x with
            % diag(M{mu}) along dimension mu, for all mu
            %
            % If M is a matrix with x.order columns, contracts x with
            % diag(M(:,mu)) along dimension mu, for all mu
            %
            % x: SparseTensor of order d
            % M: 1-by-d cell array containing vectors or matrix with d columns
            %
            % Not optimal: does not exploit sparsity
            
            if isa(M,'double')
                M = mat2cell(M,size(M,1),ones(1,size(M,2)));
            end
            M = cellfun(@(d) diag(d) , M ,'UniformOutput',false);
            x = timesMatrix(x,M,varargin{:});
        end
        
        function n = dot(x,y)
            if ~isa(y,'SparseTensor')
                y = sparse(y);
            end
            [~,ia,ib] = intersect(x.indices.array, y.indices.array,'rows');
            n = dot(x.data(ia),y.data(ib));
        end
        
        function z = times(x,y)
            if ~isa(y,'SparseTensor')
                y = sparse(y);
            end
            [~,ia,ib] = intersect(x.indices.array, y.indices.array,'rows');
            z = x;
            z.data = x.data(ia).*y.data(ib);
            z.indices.array = x.indices.array(ia,:);
        end
        
        function n = norm(x)
            n = sqrt(dot(x,x)) ;
        end
        
        function x = orth(x)
            error('Method not implemented.')
        end
        
        function x = cat(x,y)
            error('Method not implemented.')
        end
        
        function z = kron(x,y)
            dx = length(x.sz);
            dy = length(y.sz);
            dz = max(dx,dy);
            sx = [x.sz,ones(1,dz-dx)];
            sy = [y.sz,ones(1,dz-dy)];
            nx = numel(x.data);
            ny = numel(y.data);
            z = y.data(:)*x.data(:).';           
            z = z(:);
            ix = [x.indices.array , ones(length(x.indices),dz-dx)];
            iy = [y.indices.array , ones(length(y.indices),dz-dy)];
            [indy,indx] = ind2sub([ny,nx],1:nx*ny);
            ind = iy(indy,:) + repmat(sy,length(z),1) .* (ix(indx,:)-1);
            sizes = sx.*sy;
            z = SparseTensor(z,MultiIndices(ind),sizes);            

        end
        
        function s = dotWithRankOneMetric(x,y,M)
            s = timesMatrix(y,M);
            s = dot(x,s);
        end
        
        function s = timesTensorTimesMatrixExceptDim(x,y,M,dim)
            dims = 1:x.order;
            dims(dim) = [];
            s = timesMatrix(y,M(dims),dims);
            s = timesTensor(x,s,dims,dims);
        end
        
        function s = evalDiag(x,varargin)
            if nargin==1
                dims = 1:x.order;
            else
                dims=varargin{1};
            end
            if numel(dims)==1
                warning('Only one dimension: degenerate case for evalDiag. Returns the tensor itself.')
                s=x;
            else
                %s = zeros(x.order,1);
                %s(x.indices.array(~range(x.indices.array,2),1)) = x.data(~range(x.indices.array,2));
                %s = s(dims);
                I = repmat((1:size(x,dims(1)))',1,numel(dims));
                s = evalAtIndices(x,I,varargin{:});
            end
            
        end
        
        function s = subTensor(x,varargin)
            error('Method not implemented.')
        end
    end
end