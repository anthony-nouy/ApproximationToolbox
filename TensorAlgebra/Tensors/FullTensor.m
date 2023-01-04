% Class FullTensor

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

classdef FullTensor < AlgebraicTensor
    
    properties
        data            % Order-d array containing the entries of the tensor
        order           % Order of the tensor
        sz              % Size of the tensor
        isOrth = false; % The flag is true if the representation of the tensor is orthogonal (i.e. one mu-matricization is orthogonal)
        orthDim = []    % If isOrth = true, the dimension mu for which the mu-matricization of the tensor is orthogonal
    end
    
    methods
        function x = FullTensor(varargin)
            % FULLTENSOR - Constructor for the class FullTensor
            %
            % x = FullTensor(data)
            % data: full ND array
            %
            % x = FullTensor(data,order,sz)
            % order: order of the tensor
            % sz: size of the tensor, array of size 1xorder
            % data: array of size prod(sz)-by-1
            
            switch nargin
                case 0
                    x.data = [];
                    x.order = 0;
                    x.sz = [];
                case 1
                    y = varargin{1};
                    if isa(y,'double')
                        x.data = full(y);
                        x.order = ndims(y);
                        x.sz = size(y);
                    elseif isa(y,'FullTensor')
                        x.data = y.data;
                        x.order = y.order;
                        x.sz = y.sz;
                    else
                        error('Wrong type of input argument.');
                    end
                case 3
                    x.order = varargin{2};
                    x.sz = varargin{3}(:)'; % make sure it is a row
                    if x.order == 1
                        x.data = full(varargin{1}(:));
                    else
                        x.data = reshape(full(varargin{1}),x.sz);
                    end
                otherwise
                    error('Wrong number of input arguments.');
            end
        end
        
        function x = treeBasedTensor(y)
            % TREEBASEDTENSOR - Conversion of a FullTensor into TreeBasedTensor
            %
            % x = treeBasedTensor(y)
            
            x.tree = DimensionTree.trivial(y.order);
            x.tensors = cell(1,y.order);
            x.tensors{1} = y;
            for mu=1:y.order
                x.tensors{mu+1}=FullTensor(eye(y.sz(mu)),2,[y.sz(mu),y.sz(mu)]);
            end
            x = TreeBasedTensor(x.tensors,x.tree);
            
        end
        
        function y = sparse(x)
            % SPARSE - Conversion of a FullTensor into a SparseTensor
            %
            % y = sparse(x)
            % x: FullTensor
            % y: SparseTensor
            
            ind = find(x.data);
            dat = nonzeros(x.data);
            indices = MultiIndices.ind2sub(x.sz,ind);
            y = SparseTensor(dat, indices, x.sz);
        end
        
        function x = plus(x,y)
            x.data = x.data+y.data;
        end
        
        function x = uminus(x)
            x.data = -x.data;
        end
        
        function n = storage(x)
            n = numel(x);
        end
        
        function n = sparseStorage(x)
            n = nnz(x.data);
        end
        
        function n = nnz(x)
            % NNZ - Number of non-zero tensor entries
            % n = nnz(x)
            
            n = nnz(x.data);
        end
        
        function x = minus(x,y)
            x = x+(-y);
        end
        
        function x = mtimes(x,s)
            if isa(x,'FullTensor') && isa(s,'double')
                x.data = x.data*s;
            elseif isa(x,'double') && isa(s,'FullTensor')
                x = s*x;
            else
                error('Wrong type of input arguments.')
            end
        end
        
        function x = times(x,y)
            x.data = x.data.*y.data;
        end
        
        function x = power(x,y)
            if ~isa(y, 'double')
                error('The power must be a double.')
            end
            x.data = x.data.^y;
        end
        
        function x = mrdivide(x,c)
            
            if isa(c,'double') && numel(c)==1
                x.data = x.data/c;
            else
                error('Method mrdivide not implemented.')
            end

        end

        function z = dot(x,y)
            z = x.data(:)'*y.data(:);
        end
        
        function n = norm(x)
            n = norm(x.data(:));
        end
        
        function x = full(x)
        end
        
        function x = cat(x,y,varargin)
            switch nargin
                case 2
                    xa = zeros(x.sz+y.sz);
                    indx = cell(1,x.order);
                    for mu = 1:x.order
                        indx{mu} = 1:x.sz(mu);
                    end
                    xa(indx{:}) = x.data;
                    for mu = 1:y.order
                        indx{mu} = (x.sz(mu)+1):(x.sz(mu)+y.sz(mu));
                    end
                    xa(indx{:}) = y.data;
                    x.data = xa;
                    x.sz = x.sz+y.sz;
                case 3
                    dim = varargin{1};
                    otherDims = setdiff(1:x.order,dim);
                    assert(all(x.sz(otherDims) == y.sz(otherDims)),...
                        'Dimensions are not compatible.');
                    xsz = x.sz;
                    xsz(dim) = xsz(dim)+y.sz(dim);
                    if numel(dim) == 1
                        x.data = cat(dim,x.data,y.data);
                        x.sz = xsz;
                    else
                        d = x.order;
                        xd = zeros(xsz);
                        indx = cell(1,d);
                        for mu = 1:d
                            indx{mu} = 1:x.sz(mu);
                        end
                        xd(indx{:}) = x.data;
                        for mu = dim
                            indx{mu} = x.sz(mu)+1:xsz(mu);
                        end
                        xd(indx{:}) = y.data;
                        x.data = xd;
                        x.sz = xsz;
                    end
                otherwise
                    error('Wrong number of input arguments.');
            end
        end
        
        function x = sum(x,i)
            % Computes the sum of a tensor along given dimensions
            %
            % y = sum(x,i)
            % x: FullTensor of order d
            % i: array of integers or 'all'
            % y: FullTensor of order d
            % Use squeeze to remove dimensions
            
            if x.order==1
                x = sum(x.data);
            elseif isa(i,'char') && strcmpi(i,'all')
                x = sum(x.data,'all');
            else
                %i=sort(i);
                %for k=length(i):-1:1
                %    x.data = sum(x.data,i(k));
                %end
                x.data = sum(x.data,i);
                x.sz(i) = [];
                if isempty(x.sz)
                    x = x.data;
                else
                    x = FullTensor(x.data,x.order-length(i),x.sz);
                end
            end
        end
        
        function z = kron(x,y)
            dx = length(x.sz);
            dy = length(y.sz);
            dz = max(dx,dy);
            sx = [x.sz,ones(1,dz-dx)];
            sy = [y.sz,ones(1,dz-dy)];
            z = reshape(permute(reshape(y.data(:)*x.data(:).',[sy sx]),...
                reshape(reshape(1:2*dz,dz,2)',1, ...
                2*dz)),(sx.*sy));
            sizes = x.sz.*y.sz;
            z = FullTensor(z,length(sizes),sizes);
        end
        
        function x = outerProductEvalDiag(x,y,xDims,yDims,diagKron)
            % Computes the diagonal of the outer product of two tensors
            %
            % z = outerProductEvalDiag(x,y,xDims,yDims)
            % x, y: FullTensor
            % xDims, yDims: arrays
            % Returns the diagonal (accordings to dimensions xDims in x and yDims in y) of the outer product of x and y
            % Example: for order-3 tensors x and y
            % z = outerProductEvalDiag(x,y,2,3) returns an order-5 tensor
            % z(i1,k,i3,j1,j2) = x(i1,k,i3)y(j1,j2,k)
            %
            % z = outerProductEvalDiag(x,y,xDims,yDims,diagKron)
            % x, y: FullTensor
            % xDims, yDims: arrays
            % diagKron: logical (false by default)
            % if K = numel(xDims)>1 and diagKron=true, returns a tensor of order x.order+y.order-K
            % Example: for order-3 tensors x and y
            % z = outerProductEvalDiag(x,y,[1 2],[2 3],true) returns an order-4 tensor
            % z(k,l,i3,j1) = x(k,l,i3)y(j1,k,l)
            
            if nargin==2
                xDims = 1:x.order;
                yDims = 1:y.order;
                diagKron = false;
            end
            
            if nargin<5 || numel(xDims)==1
                diagKron=false;
            end
            
            if ~diagKron && numel(xDims)>1
                x = evalDiag(x,xDims);
            end
            if ~diagKron && numel(yDims)>1
                y = evalDiag(y,yDims);
            end
            
            if ~isa(x,'FullTensor') && ~isa(y,'FullTensor')
                x = x.*y;
            else
                
                if ~isa(x,'FullTensor')
                    x = FullTensor(x,1,size(x,1));
                end
                if ~isa(y,'FullTensor')
                    y = FullTensor(y,1,size(x,1));
                end
                if ~diagKron
                    rx = xDims(1);
                    ry = yDims(1);
                else
                    rx = xDims;
                    ry = yDims;
                end
                
                dx = x.order;
                dy = y.order;
                rnx = setdiff(1:dx,rx);
                rny = setdiff(1:dy,ry);
                szx = x.sz;
                szy = y.sz;
                
                K = length(rx);
                
                
                if dx>1
                    x = permute(x.data,[rx,rnx]);
                else
                    x = x.data;
                end
                if dy>1
                    y = permute(y.data,[ry,dy+1:dx+dy-K,rny]);
                else
                    y = y.data;
                end
                if verLessThan('matlab','9.1')
                    x = repmat(x,[ones(1,dx),szy(rny)]);
                    y = repmat(y,[ones(1,K),szx(rnx),ones(1,dy-K)]);
                end
                x = x.*y;
                
                rep = [rx,rnx,dx+1:dx+dy-K];
                x = ipermute(x,rep);
                x = FullTensor(x,dx+dy-K,[szx,szy(rny)]);
            end
        end
        
        function x = reshape(x,s)
            % Reshapes the tensor
            %
            % y = reshape(x,s)
            % x: FullTensor
            % s: array of integers
            % y: FullTensor of order numel(s) and size s
            
            s = s(:)';
            if numel(s) == 1
                x.data = x.data(:);
            else
                x.data = reshape(x.data,s);
            end
            x.sz = s;
            x.order = numel(s);
        end
        
        function x = permute(x,dims)
            % Permutes array dimensions
            % x = permute(x,dims)
            %
            % See also permute
            
            if length(dims)~=x.order
                error('Length of dim should be the order of the tensor.')
            end
            if x.order>1
                x.data = permute(x.data,dims);
                x.sz = x.sz(dims);
                x.sz = x.sz(:)';
            elseif dims~=1
                error('For a tensor of order 1, dims should be equal to 1.')
            end
        end
        
        function x = ipermute(x,dims)
            % Computes the inverse permutation
            %
            % x = ipermute(x,dims)
            % x: FullTensor
            % dims: array
            %
            % See also ipermute
            
            x.data = ipermute(x.data,dims);
            idims(dims) = 1:numel(dims);
            x.sz = x.sz(idims);
            x.sz = x.sz(:)';
        end
        
        function x = squeeze(x,dims)
            % Remove singleton dimensions.
            %
            % See also squeeze
            
            if nargin == 1
                dims = 1:x.order;
                dims(x.sz > 1) = [];
            end
            x.sz(dims) = [];
            if isempty(x.sz)
                x = x.data;
            else
                x = reshape(x,x.sz);
            end
        end
        
        function x = subTensor(x,varargin)
            % Extracts a subtensor of a tensor
            %
            % y = subTensor(x,I1,I2,...,Id)
            % x: FullTensor
            % Ik: array of integers or char ':'
            % y: FullTensor
            %
            % Example: subTensor(x,[1,2],':',[2,5,6]) returns a tensor y with size 2-by-x.sz(2)-by-3
            
            assert(nargin == 1+x.order, ...
                'Wrong number of input arguments.');
            xsz = x.sz;
            for k = 1:x.order
                if ~strcmp(':',varargin{k})
                    xsz(k) = numel(varargin{k});
                end
            end
            x.data = x.data(varargin{:});
            x.sz = xsz;
            x = FullTensor(x.data,x.order,xsz);
        end
        
        function x = timesVector(x,V,varargin)
            if isa(V,'double')
                V = {V};
            end
            V = cellfun(@(v) v',V,'uniformoutput',false);
            x = timesMatrix(x,V,varargin{:});
            x = squeeze(x,varargin{:});
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
            if x.order == 1
                x.data = M{1}*x.data;
                x.sz = size(x.data,1);
            else
                M = cellfun(@full,M(:),'UniformOutput',false) ;
                k = 1;
                for mu = dims
                    permDims = [mu,1:mu-1,mu+1:x.order];
                    x = permute(x,permDims);
                    szx0 = x.sz;
                    szx = [szx0(1) prod(szx0(2:end))];
                    x = reshape(x,szx);
                    x.data = M{k}*x.data;
                    szx0(1) = size(M{k},1);
                    x = reshape(x,szx0);
                    x = ipermute(x,permDims);
                    k = k+1;
                end
            end
            x.isOrth = false;
            x.orthDim = [];
        end
        
        function c = timesMatrixEvalDiag(c,H,dims)
            if nargin==2
                dims = 1:c.order;
            end
            
            [~,I] = sort(dims,'descend');
            
            i = I(1);
            c = timesTensor(FullTensor(H{i}),c,2,dims(i));
            for i=I(2:end)
                c = timesTensorEvalDiag(FullTensor(H{i}),c,2,dims(i)+1,1,1);
            end
            if c.order==1
                c = c.data;
            end
        end
        
        function x = timesDiagMatrix(x,M,varargin)
            if isa(M,'double')
                M = mat2cell(M,size(M,1),ones(1,size(M,2)));
            end
            M = cellfun(@(d) diag(d) , M ,'UniformOutput',0);
            x = timesMatrix(x,M,varargin{:});
        end
        
        function x = timesTensor(x,y,xDims,yDims)
            % Contraction of two tensors along given dimensions
            %
            % z = timesTensor(x,y,xDims,yDims)
            % x: algebraic tensor of order dx
            % y: algebraic tensor of order dy
            % z: algebraic tensor of order dz = dx + dy - 2*m
            % xDims, yDims : arrays of the same size m
            %
            % Example: For x and y of order 4
            % z = timesTensor(x,y,[1,3],[3,2])
            % z(i2,i4,j1,j4) = sum_{k,l} x(k,i2,l,i4)x(j1,l,k,j4)
            
            assert(all(x.sz(xDims) == y.sz(yDims)),...
                'Dimensions of tensors are not compatible.');
            if (numel(xDims) == x.order) && (numel(yDims) == y.order)
                x = permute(x,xDims);
                y = permute(y,yDims);
                x = dot(x,y);
                x = FullTensor(x);
            else
                nbDims = numel(xDims);
                % Permute and reshape x
                xDims = xDims(:)';
                xperm = 1:x.order;
                xperm(xDims) = [];
                xperm = [xperm xDims];
                szx0 = x.sz;
                x = permute(x,xperm);
                szx = x.sz;
                x = reshape(x,[prod(szx(1:end-nbDims)) ...
                    prod(szx(end-nbDims+1:end))]);
                % Permute and reshape y
                yDims = yDims(:)';
                yperm = 1:y.order;
                yperm(yDims) = [];
                yperm = [yDims yperm];
                szy0 = y.sz;
                y = permute(y,yperm);
                szy = y.sz;
                y = reshape(y,[prod(szy(1:nbDims)) prod(szy(nbDims+1:end))]);
                % Contract the result and reshape to the final size
                x.data = x.data*y.data;
                x.sz(2) = y.sz(2);
                szx0(xDims) = [];
                szy0(yDims) = [];
                x = reshape(x,[szx0 szy0]);
                x.isOrth = false;
                x.orthDim = [];
            end
            
        end
        
        function z = timesTensorEvalDiag(x,y,xDims,yDims,xDiagDims,yDiagDims)
            % Evaluation of the diagonal of a tensor obtained by contraction of two tensors
            %
            % z = timesTensorEvalDiag(x,y,xDims,yDims,xDiagDims,yDiagDims)
            % x, y: FullTensor
            % xDims, yDims, xDiagDims, yDiagDims: arrays of integers
            % Contracts tensors x and y along dimensions xDims for x and yDims for y, and evaluates the diagonal according to dimensions xDiagDims for x and yDiagDims for y
            %
            % Examples:
            % for x and y of order 4
            % timesTensorEvalDiag(x,y,2,3,3,1) returns an order-4 tensor
            % z(i1,k,i4,j2,j4) = sum_{l} x(i1,l,k,i4) y(k,j2,l,j4)
            %
            % timesTensorEvalDiag(x,y,[2,4],[3,4],3,1) returns an order-3 tensor
            % z(i1,k,j2) = sum_{l1,l2} x(i1,l1,k,l2) y(k,j2,l1,l2)
            
            if numel(xDiagDims)>1
                dx = x.order;xDiagDims=sort(xDiagDims);
                x = evalDiag(x,xDiagDims);
                xDimsNew = setdiff(1:dx,xDiagDims(2:end));
                [~,xDims]=ismember(xDims,xDimsNew);
                xDiagDims = xDiagDims(1);
            end
            if numel(yDiagDims)>1
                dy=y.order;yDiagDims=sort(yDiagDims);
                y = evalDiag(y,yDiagDims);
                yDimsNew = setdiff(1:dy,yDiagDims(2:end));
                [~,yDims]=ismember(yDims,yDimsNew);
                yDiagDims = yDiagDims(1);
            end
            z = outerProductEvalDiag(x,y,[xDims,xDiagDims],[yDims,yDiagDims],true);
            z = sum(z,xDims);
            z = squeeze(z,xDims);
        end
        
        function [x,varargout] = orth(x,mu)
            % ORTH - Orthogonalization of the tensor
            %
            % [z,R] = ORTH(x,mu) returns a tensor z whose mu-matricization is an orthogonal matrix and corresponds to the Q factor of a QR factorization of the mu-matricization of x
            % The second output argument is the R factor
            %
            % M_{mu}(x) = M_{mu}(z)*R
            
            assert(nargin <= 2, 'Wrong number of input arguments.');
            if nargin == 2
                d = x.order;
                dims = [1:mu-1 mu+1:d mu];
                x = permute(x,dims);
                sz0 = x.sz;
                x = reshape(x,[prod(x.sz(1:d-1)) x.sz(d)]);
                [xd,r] = qr(x.data,0);
                sz0(d) = size(r,1);
                x = reshape(FullTensor(xd),sz0);
                x = ipermute(x,dims);
                x.isOrth = true;
                x.orthDim = mu;
                varargout{1} = r;
            end
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
            s = s.data;
        end
        
        function x = matricize(x,dims1,dims2)
            % Matricization of a tensor
            %
            % y = matricize(x,alpha,beta)
            % For x a tensor of order d, and two complementary subsets alpha and beta in {1,...,d},
            % y(i_alpha,i_beta) = x(i) for i=(i_1,...,i_d)
            %
            % x: FullTensor of order d
            % alpha: array
            % beta: array (by default beta = setdiff(1:d,alpha))
            % y: FullTensor of order 2
            
            if nargin==2
                dims2=setdiff(1:x.order,dims1);
            end
            sz1 = size(x,dims1);
            sz2 = size(x,dims2);
            x.data =  permute(x.data,[dims1,dims2]);
            x.data = reshape(x.data,prod(sz1),prod(sz2));
            x.sz = size(x.data);
            x.order=2;
            x.isOrth=false;
        end
        
        function s = evalAtIndices(x,I,dims)
            if isa(I,'MultiIndices')
                I = I.array;
            end
            if nargin==2
                dims=1:x.order;
            else
                [dims,isort] = sort(dims);
                I = I(:,isort) ;
                assert(numel(dims)==size(I,2),'Wrong arguments.')
            end
            
            I = mat2cell(I,size(I,1),ones(1,size(I,2)));
            if length(dims)~=length(I)
                error('Wrong size of multiindices.')
            end
            if nargin==2 || numel(dims)==x.order
                I = sub2ind(x.sz,I{:});
                s = x.data(I);
            elseif numel(dims)==1
                J = repmat({':'},1,x.order);
                J{dims}=I{1};
                s = subTensor(x,J{:});
            else
                nodims = setdiff(1:x.order,dims);
                s = matricize(x,dims);
                I = sub2ind(x.sz(dims),I{:});
                s.data = s.data(I,:);
                s = reshape(s,[numel(I),size(x,nodims)]);
                leftdims = 1:dims(1)-1;
                s = permute(s,[2:numel(leftdims)+1,1,numel(leftdims)+2:x.order-numel(dims)+1]);
            end
        end
        
        function s = evalDiag(x,varargin)
            if nargin==1
                dims = 1:x.order;
            else
                dims=varargin{1};
            end
            if numel(dims)==1
                % warning('only one dimension:degenerate case for evalDiag. Return the tensor itself')
                s = x.data;
            else
                if ~all(x.sz(dims)==x.sz(dims(1)))
                    error('Sizes of the tensor in dimensions dims should be equal.')
                end
                I = repmat((1:size(x,dims(1)))',1,numel(dims));
                s = evalAtIndices(x,I,varargin{:});
            end
        end
        
        function [V,s] = principalComponents(x,t)
            % Computes r principal components of an order-2 tensor
            %
            % [V,s] = principalComponents(x,t)
            % x: FullTensor of order 2
            % t: double or integer
            % V: x.sz(1)-by-r double
            % s: r-by-r diagonal array
            % For an order-two tensor, returns a matrix V containing r principal components of x, such that
            % - if t is an integer, r=min(t,x.sz(2))
            % - if t<1, the rank r is determined such that || x - VV'x ||_F < t || x ||_F
            % By default, t = size(x,1)
            
            assert(ndims(x)==2,'The order of the tensor must be 2.')
            if nargin==1 || t>size(x,1)
                t = size(x,1);
            end
            if t<1
                tr = Truncator('tolerance',t,'maxRank',Inf);
            else
                tr = Truncator('tolerance',0,'maxRank',t);
            end
            xt = tr.truncate(x);
            V = xt.space.spaces{1};
            s = diag(xt.core.data);
        end
        
        function [V,s] = alphaPrincipalComponents(x,alpha,varargin)
            % alphaPrincipalComponents - Computes alpha-principal components
            %
            % [V,s] = alphaPrincipalComponents(x,alpha,t)
            % Principal Components of the alpha-matricisation M_alpha(x) of a tensor x of order d
            % alpha: subset of [1,...,d]
            % t: rank or tolerance (by default, t=size(M_alpha(x),1)
            
            x = matricize(x,alpha);
            [V,s] = principalComponents(x,varargin{:});
        end
        
        function x = alphaMatricisation(x,alpha)
            % Matricization of the tensor, equivalent to matricize, to be removed in a future release
            %
            % x = alphaMatricisation(x,alpha)
            % Equivalent to x = matricize(x,alpha)
            % To be removed in a future release
            %
            % See also FullTensor/matricize
            
            warning('To be removed in a future release, use matricize instead.')
            x = matricize(x,alpha);
        end
        
        function x = updateProperties(x)
            % Updates the properties of a tensor
            %
            % x = updateProperties(x)
            % x: FullTensor
            
            x.order = ndims(x.data) ;
            x.sz = size(x.data) ;
        end
        
        
        function sv = singularValues(x)
            % Computes the higher-order singular values of a tensor (the collection of singular values of d different matricizations)
            %
            % sv = singularValues(x)
            % x: FullTensor
            % sv: cell of arrays
            
            if x.order == 2
                sv = svd(x.data);
            else
                d = x.order;
                sv = cell(d,1);
                for mu = 1:d
                    y = matricize(x,mu);
                    sv{mu} = svd(y.data);
                end
            end
        end
    end
    
    methods (Static)
        function x = create(generator,sz)
            % CREATE - Creates a FullTensor of size sz using a given generator
            % x = create(generator,sz)
            
            if numel(sz == 1)
                szd = [sz 1];
            else
                szd = sz;
            end
            a = generator(szd);
            d = numel(sz);
            x = FullTensor(a,d,sz);
        end
        
        function x = zeros(sz)
            % Creates a tensor of size sz with entries equal to 0
            % x = zeros(sz)
            
            x = FullTensor.create(@zeros,sz);
        end
        
        function x = ones(sz)
            % Creates a tensor of size sz with entries equal to 1
            % x = ones(sz)
            
            x = FullTensor.create(@ones,sz);
        end
        
        function x = randn(sz)
            % Creates a tensor of size sz with i.i.d. entries drawn according to the standard gaussian distribution
            % x = randn(sz)
            
            x = FullTensor.create(@randn,sz);
        end
        
        function x = rand(sz)
            % Creates a tensor of size sz with i.i.d. entries drawn according to the uniform distribution on (0,1)
            % x = rand(sz)
            
            x = FullTensor.create(@rand,sz);
        end
        
        
        function x = diag(v,d)
            % Creates a diagonal tensor
            %
            % x = diag(v,d)
            % v: array
            % d: integer
            % x: FullTensor of order d
            %
            % x(i,...,i) = v(i) for i=1...length(v)
            
            onesv = ones(1,d);
            szv = numel(v) * onesv;
            x = FullTensor.zeros(szv);
            for k = 1:numel(v)
                ind = num2cell(k*onesv);
                x.data(ind{:}) = v(k);
            end
        end
    end
end