% Class TuckerLikeTensor: algebraic tensors in Tucker-like tensor format

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

classdef TuckerLikeTensor < AlgebraicTensor
    
    properties
        core     % core tensor
        space    % bases of subspaces
        order    % order of the tensor
        sz       % size of the tensor
        isOrth   % The flag is true if the representation of the tensor is orthogonal
    end
    
    methods
        function x = TuckerLikeTensor(core,space)
            % TuckerLikeTensor - Constructor for the class TuckerLikeTensor
            %
            % x = TuckerLikeTensor(core,space)
            % Creates a tensor of order d in a product space V_1 \otimes ... \otimes V_d,
            % with bases of spaces V_k given in space,
            % and coefficients in the corresponding product basis given by core
            %
            % core: AlgebraicTensor
            % space: TSpace (TSpaceVectors or TSpaceOperators)
            %
            % x = TuckerLikeTensor(y)
            % If y is a TreeBasedTensor or a TTTensor of size sz,
            % it converts y in a TuckerLikeTensor.
            % For dimensions k with no constraint on the
            % rank, the subspace V_k = R^{sz(k)} and the corresponding basis if the the
            % canonical basis.
            %
            % See also TSpaceVectors, TSpaceOperators
            
            if nargin==2
                x.core = core;
                x.space = space;
            elseif nargin==1 && isa(core,'TreeBasedTensor')
                isActiveNode = core.isActiveNode;
                y=core;
                space = cell(1,y.order);
                for mu = 1:y.order
                    nod = y.tree.dim2ind(mu);
                    if isActiveNode(nod)
                        space{mu} = double(y.tensors{nod});
                        y.tensors{nod} = [];
                    else
                        space{mu} = eye(y.sz(mu));
                    end
                end
                core = TreeBasedTensor(y.tensors,y.tree);
                space = TSpaceVectors(space);
                x.space = space;
                x.core = core;
                x.isOrth = y.isOrth;
            elseif nargin==1 && isa(core,'TTTensor')
                y=core;
                space = cell(1,y.order);
                for mu = 1:y.order
                    space{mu} = eye(y.sz(mu));
                end
                x.core = core;
                x.space = space;
            end
            
            x = updateProperties(x);
        end
        
        function x = treeBasedTensor(y,varargin)
            % Conversion into TreeBasedTensor
            %
            % x = treeBasedTensor(y)
            % converts TuckerLikeTensor into a TreeBasedTensor
            % y: TuckerLikeTensor
            % y.core: TreeBasedTensor with non active leaves,
            %         FullTensor (use a trivial tree)
            %
            % x = TreeBasedTensor(y,tree,activeNodes)
            % y: TuckerLikeTensor
            % y.core: DiagonalTensor
            % tree: DimensionTree (linear by default)
            % activeNodes: logical of size 1:tree.nbodes (~tree.isLeaf by default)
            
            if isa(y.space,'TSpaceOperators')
                error('Method not implemented.')
            end
            
            if isa(y.core,'TreeBasedTensor')
                x = y.core;
                x.tree = y.core.tree;
                x.tensors(x.tree.dim2ind) = y.space.spaces;
                
            elseif isa(y.core,'FullTensor')
                x.tree = DimensionTree.trivial(y.order);
                x.tensors = cell(1,y.order);
                x.tensors{1} = y;
                x.tensors(2:y.order+1) = y.space.spaces;
                
            elseif isa(y.core,'DiagonalTensor')
                x = treeBasedTensor(y.core,varargin{1});
                x.tensors(x.tree.dim2ind) = y.space.spaces;
            end
            
            x = TreeBasedTensor(x.tensors,x.tree);
            
            if nargin==3
                x = inactivateNodes(x,find(~varargin{2}));
            end
        end
        
        function r = representationRank(x)
            % Returns the representation rank of the tensor
            %
            % r = representationRank(x)
            
            r = representationRank(x.space);
        end
        
        function n = storage(x)
            n = storage(x.space)+storage(x.core);
        end
        
        function n=sparseStorage(x)
            n = sparseStorage(x.space)+sparseStorage(x.core);
        end
        
        function x = plus(x,y)
            if isempty(y)
                return
            elseif isempty(x)
                x = y ;
                return
            end
            [x,y] = convertCores(x,y);
            x.core = cat(x.core,y.core);
            x.space = cat(x.space,y.space);
            x = updateProperties(x);
        end
        
        
        function [y,flag] = sqrt(x,tol)
            % Computes the tensor whose entries are the square root of the entries of a given tensor
            % Uses Truncated Newton iterations (maximum 100 iterations, maximum rank = 100, to be modified within the code)
            %
            % [y,flag] = sqrt(x,tol)
            % x,y: TuckerLikeTensor
            % tol: tolerance (1e-5 by default)
            % flag: 2 if converged
            %
            % y(i) = sqrt(x(i)) for all i
            
            y0 = x;
            if nargin==1
                tol = 1e-5;
            end
            flag = 1;
            tr = Truncator();
            tr.tolerance = 1e-5;
            tr.maxRank = 100;
            for k=1:100
                y = 1/2*(y0+times(power(y0,-1,1e-2),x));
                st = norm(y-y0)/norm(y);
                fprintf('newton iteration %d, error = %d\n',k,st);
                if st<tol
                    flag = 2;
                    break
                end
                y0 =tr.truncate(y);
            end
        end
        
        function x = toOperator(x)
            % Converts a TuckerLikeTensor of type Vector into a TuckerLikeTensor of type Operator
            % y = toOperator(x)
            % x: TuckerLikeTensor with a TSpace of type TSpaceVectors
            % y: TuckerLikeTensor with a TSpace of type TSpaceOperators
            %
            % x = sum_{i1,...,id} c_{i1,...,id} v_{i_1}^1\otimes ... \otimes v_{i_d}^d
            % y = sum_{i1,...,id} c_{i1,...,id} A_{i_1}^1\otimes ... \otimes A_{i_d}^d
            % with A_i^k = sparse(diag(v_i^k))
            
            x.space = TSpaceOperators(x.space);
            x.space.spaces = cellfun(@(y) cellfun(@(x)spdiags(x,0,size(x,1),size(x,1)),y,'uniformoutput',false),...
                x.space.spaces,'uniformoutput',false);
            x.space =  updateProperties(x.space);
            x =  updateProperties(x);
        end
        
        function x = power(x,n,tol)
            % Computes the n-th power of the entries of a tensor
            %
            % y = power(x,n,tol)
            % x: TuckerLikeTensor
            % n: integer (positive or negative) or n=1/2
            % tol: tolerance
            
            if nargin<3
                tol=1e-4;
            end
            if ceil(n)~=floor(n)
                error('Power n must be integer.')
            end
            
            if n >= 1
                for k=1:n-1
                    x = times(x,x);
                end
            elseif n<0 && n~=-1
                x = power(x,-n);
                x = power(x,-1,tol);
            elseif n==-1
                b =  TuckerLikeTensor.ones(x.sz);
                xop = toOperator(x);
                %                sinv = GreedyLinearSolver(xop, b,...
                %                    'maxIterations',100,'display',true,'tolerance',tol);
                [xop.core,b.core] = convertTensors(xop.core,b.core);
                su = TuckerLikeTensorALSLinearSolver(xop, b, 'useDMRG', true, 'stagnation', tol, 'display', false);
                sinv = GreedyLinearSolver(xop,b,'maxIterations',9,'update',@(x) su.updateCore(x));
                sinv.tolerance = tol;
                sinv.stagnation = tol;
                x = sinv.solve();
            elseif n==1/2
                x=sqrt(x);
            else
                error('Method not implemented.')
            end
        end
        
        function x = mtimes(x,y)
            if isa(x,'TuckerLikeTensor') && isa(y,'TuckerLikeTensor')
                [x,y] = convertCores(x,y);
                x.core  = kron(x.core,y.core);
                x.space = mtimes(x.space,y.space);
                x = updateProperties(x);
            elseif isa(y,'double')
                x.core = x.core*y;
            elseif isa(x,'double')
                x = y*x;
            else
                error('Wrong type of input arguments.');
            end
        end
        
        function x = uminus(x)
            x.core = x.core*(-1);
        end
        
        function x = minus(x,y)
            x= x + (-y);
        end
        
        function x = mrdivide(x,a)
            % Divides a tensor by a scalar
            %
            % y = mrdivide(x,a)
            % x: TuckerLikeTensor
            % a: double
            % y = x/a
            
            x.core = x.core*(1/a);
        end
        
        function ps = dot(x,y)
            [x,y] = convertCores(x,y);
            M = dot(x.space,y.space);
            ps = dotWithRankOneMetric(x.core,y.core,M);
        end
        
        function n = norm(x,M)
            if nargin==1
                if x.isOrth
                    n = norm(x.core);
                else
                    n = sqrt(abs(dot(x,x)));
                end
            else
                n = sqrt(abs(dot(M*x,x)));
            end
        end
        
        function x = orth(x)
            dims = 1:x.order;
            [x.space,M] = orth(x.space);
            x.core = timesMatrix(x.core,M,dims);
            x.core = orth(x.core);
            x = updateProperties(x);
        end
        
        function xf = full(x)
            if isa(x.space,'TSpaceOperators')
                % Vectorize then unvectorize at the end.
                % May not be optimal.
                xsz = x.space.sz ;
                x = vectorize(x) ;
                vectorized = true ;
            else
                vectorized = false ;
            end
            xf = x.core;
            xf = timesMatrix(xf,x.space.spaces);
            xf = full(xf);
            if vectorized
                xf = reshape(xf,xsz(:)') ;
            end
        end
        
        function z = dotWithRankOneMetric(x,y,M)
            y = timesMatrix(y,M);
            z = dot(x,y);
        end
        
        function z = timesTensorTimesMatrixExceptDim(x,y,M,dim)
            otherDims = [1:dim-1 dim+1:x.order];
            Mc = M;
            for mu = otherDims
                Mc{mu} = x.space.spaces{mu}'*M{mu}*y.space.spaces{mu};
            end
            z = timesTensorTimesMatrixExceptDim(x.core,y.core,Mc,dim);
            z = x.space.spaces{dim}*z*y.space.spaces{dim}';
        end
        
        function x = cat(x,y)
            x.core = cat(x.core,y.core);
            x.space = diagCat(x.space,y.space);
            x = updateProperties(x);
        end
        
        function x = kron(x,y)
            x.core = kron(x.core,y.core);
            x.space.spaces = cellfun(@(xsp,ysp) kron(xsp,ysp),x.space.spaces, ...
                y.space.spaces,'uniformoutput',false);
            x.space = updateProperties(x.space);
            x = updateProperties(x);
        end
        
        function x = times(x,y)
            assert(strcmp(class(x.space),class(y.space)),'Spaces type mismatch.')
            if isa(x.space,'TSpaceOperators')
                x.core = kron(x.core,y.core);
                xs = cell(x.order,1);
                for mu = 1:x.order
                    xs{mu} = cell(y.space.dim(mu),x.space.dim(mu));
                    for j = 1:x.space.dim(mu)
                        for i = 1:y.space.dim(mu)
                            xs{mu}{i,j} = y.space.spaces{mu}{i}.*x.space.spaces{mu}{j};
                        end
                    end
                    xs{mu} = xs{mu}(:);
                end
                x.space = TSpaceOperators(xs);
            else
                x = toOperator(x);
                x = mtimes(x,y);
            end
        end
        
        function x = timesVector(x,V,varargin)
            assert(isa(x.space,'TSpaceVectors'),...
                'The TSpace must be of TSpaceVectors type.');
            if nargin == 2
                dims = 1:x.order;
            else
                dims = varargin{1};
            end
            if isa(V,'double')
                V = {V};
            end
            V = V(:);
            V = cellfun(@(v) v(:)',V,'uniformoutput',false);
            xs = matrixTimesSpace(x.space,V,dims);
            xs.spaces(dims) = cellfun(@(v) v',xs.spaces(dims),'uniformoutput',false);
            xc = timesVector(x.core,xs.spaces(dims),dims);
            if numel(dims) ~= x.order
                xs = removeSpace(xs,dims);
                x = TuckerLikeTensor(xc,xs);
            else
                x = xc;
            end
        end
        
        function x = timesDiagMatrix(x,M,varargin)
            assert(isa(x.space,'TSpaceVectors'),...
                'The TSpace must be of TSpaceVectors type.');
            if isa(M,'double')
                M = {M};
            end
            M = cellfun(@(m) diag(m),M,'uniformoutput',false);
            x.space = spaceTimesMatrix(x.space,M,varargin{:});
        end
        
        function x = timesMatrix(x,M,varargin)
            xs = matrixTimesSpace(x.space,M,varargin{:});
            x = TuckerLikeTensor(x.core,xs);
        end
        
        function x = squeeze(x,dims)
            if nargin == 1
                dims = 1:x.order;
                dims(x.sz > 1) = [];
            end
            xsp = cellfun(@(z) z',x.space.spaces(dims),'uniformoutput',false);
            x.core = timesVector(x.core,xsp,dims);
            x.space.spaces(dims) = [];
            x.space = updateProperties(x.space);
            x = updateProperties(x);
        end
        
        function x = subTensor(x,varargin)
            if isa(x.space,'TSpaceVectors')
                assert(nargin == 1+x.order, ...
                    'Wrong number of input arguments.');
                for k = 1:x.order
                    if ~strcmp(':',varargin{k})
                        x.space.spaces{k} = x.space.spaces{k}(varargin{k},:);
                    end
                end
            elseif isa(x.space,'TSpaceOperators')
                assert(nargin == 1+2*x.order, ...
                    'Wrong number of input arguments.');
                for k = 1:x.order
                    I = varargin([2*k-1 2*k]) ;
                    for n=1:x.space.dim(1,k)
                        if ~strcmp(':',I{1})
                            x.space.spaces{k}{n} = x.space.spaces{k}{n}(I{1},:);
                        end
                        if ~strcmp(':',I{2})
                            x.space.spaces{k}{n} = x.space.spaces{k}{n}(:,I{2});
                        end
                    end
                end
            end
            x.space = updateProperties(x.space);
            x = updateProperties(x);
        end
        
        function x = superTensor(x,sz,varargin)
            % Creates a tensor with given subtensor and zero entries elsewhere (use sparse storage)
            %
            % s = supertensor(x,sz,I1,...,Id)
            % s is a tensor of size sz(1)-by-...-by-sz(d) such that
            % x(I1(k1),...,Id(kd)) = s(k1,...,kd) and x(i) = 0 for entries i
            % not in I1 x ... x Id
            %
            % TSpace uses sparse storage.
            
            if isa(x.space,'TSpaceVectors')
                assert(nargin == 2+x.order, ...
                    'Wrong number of input arguments.');
                for k = 1:x.order
                    if ~strcmp(':',varargin{k})
                        xsk = x.space.spaces{k} ;
                        x.space.spaces{k} = sparse(sz(k), ...
                            x.space.dim(k));
                        x.space.spaces{k}(varargin{k},:) = xsk ;
                    end
                end
            elseif isa(x.space,'TSpaceOperators')
                assert(nargin == 2+2*x.order, ...
                    'Wrong number of input arguments.');
                for k = 1:x.order
                    I = varargin([2*k-1 2*k]) ;
                    for n=1:x.space.dim(1,k)
                        [i1,i2] = find(x.space.spaces{k}{n}) ;
                        i = sub2ind(size(x.space.spaces{k}{n}),i1,i2);
                        if ~strcmp(':',I{1})
                            i1 = I{1}(i1) ;
                        end
                        if ~strcmp(':',I{2})
                            i2 = I{2}(i2) ;
                        end
                        x.space.spaces{k}{n} = sparse(i1,i2,...
                            x.space.spaces{k}{n}(i),sz(1,k),sz(2,k)) ;
                    end
                end
            end
            x.space = updateProperties(x.space);
            x = updateProperties(x);
        end
        
        function x = ctranspose(x)
            % Compute the complex conjugate transpose of a TuckerLikeTensor of type Operator
            %
            % B=ctranspose(A)
            % A,B: TuckerLikeTensor with TSpace of type TSpaceOperators
            % For any complex tensors x and y, dot(y,A*x) = dot(B*y,x), with dot(.,.) the Hermitian inner product
            
            
            assert(isa(x.space,'TSpaceOperators'),...
                'The TSpace must be of TSpaceOperators type.');
            x.space=x.space';
            x.core.data  = conj(x.core.data);
        end
        
        function x = transpose(x)
            % Compute the transpose of a TuckerLikeTensor of type Operator
            %
            % function B=transpose(A)
            % A,B : TuckerLikeTensor with TSpace of type TSpaceOperators
            % For any complex tensors x and y, dot(y,A*x) = dot(B*y,x),
            % with dot(.,.) the canonical inner product
            
            assert(isa(x.space,'TSpaceOperators'),...
                'The TSpace must be of TSpaceOperators type.');
            x.space=x.space.';
        end
        
        function x = normalizeBasis(x)
            % Normalize the elements of the bases of subspaces
            
            N = dot(x.space,x.space);
            N = cellfun(@(N) diag(sqrt(N)) ,N,'UniformOutput',0);
            x.core = timesDiagMatrix(x.core,N);
            N = cellfun(@(N) spdiags(1./N,0,size(N,1),size(N,1)) ,N,'UniformOutput',0);
            x.space = spaceTimesMatrix(x.space,N);
        end
        
        function x = updateAllProperties(x)
            % Updates all properties including properties of the core and tensor spaces
            
            x.core = updateProperties(x.core);
            x.space = updateProperties(x.space);
            x = updateProperties(x);
            
        end
        
        function x = updateProperties(x)
            % Updates properties (order,sz,isOrth)
            
            x.order = x.space.order;
            x.sz = x.space.sz;
            x.isOrth = x.space.isOrth * x.core.isOrth;
        end
        
        function [x,y] = convertCores(x,y)
            % Converts the cores of two given TuckerLikeTensor into a common format
            %
            % [x,y] = convertCores(x,y)
            
            [x.core,y.core] = convertTensors(x.core,y.core);
        end
        
        function s= evalDiag(x,dims)
            if nargin==1
                dims=1:x.order;
            elseif numel(dims)==1
                warning('Only one dimension:degenerate case for evalDiag. Return the tensor itself.')
                s=x;
                return
            else
                dims=sort(dims);
                nodims = setdiff(1:x.order,dims);
                newdims=setdiff(1:x.order,dims(2:end));
            end
            
            if isa(x.core,'DiagonalTensor')
                s = x.space.spaces{dims(1)};
                if isnumeric(s) % TSpaceVectors
                    for k=dims(2:end)
                        s = s.*x.space.spaces{k};
                    end
                elseif iscell(s) % TSpaceOperators
                    for k=2:dims(2:end)
                        for n=1:x.space.dim(1,k)
                            s{n} = s{n}.*x.space.spaces{k}{n};
                        end
                    end
                else
                    error('Method not implemented.')
                end
                if nargin==1 || numel(dims)==x.order
                    if isnumeric(s) % TSpaceVectors
                        s = s*x.core.data(:);
                    elseif iscell(s) %  TSpaceOperators
                        t = s(2:end) ;
                        s = s{1}*x.core.data(1) ;
                        for n = 1:numel(t)
                            s = s+t{n}*x.core.data(n+1) ;
                        end
                    else
                        error('Method not implemented.')
                    end
                else
                    x.space.spaces{dims(1)} = s;
                    x.space = keepSpace(x.space,newdims);
                    x.core.sz = x.core.sz(newdims);
                    x.core.order = numel(newdims);
                    s=updateProperties(x);
                end
            elseif isa(x.core,'FullTensor')
                if isa(x.space,'TSpaceOperators')
                    % Vectorize then unvectorize at the end.
                    % May not be optimal.
                    xsz = x.space.sz ;
                    x = vectorize(x) ;
                    vectorized = true ;
                else
                    vectorized = false ;
                end
                if nargin==2
                    x=permute(x,[dims,nodims]);
                end
                s = timesMatrix(x.core,x.space.spaces{1},1);
                for k=2:numel(dims)
                    v = repmat(x.space.spaces{k},[1,1,s.sz(3:end)]);
                    s = times(s,FullTensor(v,s.order,s.sz));
                    s.data = sum(s.data,2);
                    s.sz(2)=1;
                    s=squeeze(s,2);
                end
                if nargin==1 || numel(dims)==x.order
                    s=s.data;
                else
                    x.core=s;
                    x.space.spaces{1} = speye(size(s,1));
                    x.space = keepSpace(x.space,[1,numel(dims)+1:x.order]);
                    s = updateProperties(x);
                    leftdims = 1:dims(1)-1;
                    s = permute(s,[2:numel(leftdims)+1,1,numel(leftdims)+2:numel(newdims)]);
                    if vectorized
                        orderDiff = x.order-s.order;
                        xsz(:,dims(2:1+orderDiff)) = [] ;
                        s = unvectorize(s,xsz) ;
                    end
                end
            elseif isa(x.core,'TreeBasedTensor')
                s = evalDiag(treeBasedTensor(x));
                
            elseif isa(x.core,'SparseTensor')
                s = x.space.spaces{1}(:,x.core.indices.array(:,1));
                for k=2:x.order
                    s = s.*x.space.spaces{k}(:,x.core.indices.array(:,k));
                end
                s = s*x.core.data;
            else
                s = timesMatrix(x.core,x.space.spaces);
                s = evalDiag(s,dims);
            end
        end
        
        function x = permute(x,dims)
            % Permutes array dimensions
            % x = permute(x,dims)
            %
            % See also permute
            
            x.core = permute(x.core,dims);
            x.space = permute(x.space,dims);
            x=updateProperties(x);
        end
        
        function [x,P] = vectorize(x,dims)
            % Vectorization of an operator in Tucker like format (returns a sparsity pattern for sparse operators)
            %
            % [x,P] = vectorize(x,dims)
            % P is sparsity pattern (from full to sparse) in vector format
            % A single output argument disables sparsity handling
            
            if nargin<2 || isempty(dims)
                dims=1:x.order;
            end
            if nargout > 1
                [x.space,P] = vectorize(x.space,dims);
            else
                x.space = vectorize(x.space,dims);
            end
            x = updateProperties(x);
        end
        
        function x = unvectorize(x,sz,dims,P)
            % Unvectorization of a tensor in Tucker like format (returns a sparse operator if a sparsity pattern is provided)
            %
            % x = unvectorize(x,sz,dims,P)
            % Convert to operators of size sz along orders dims
            % sz is a 2-by-numel(dims) matrix
            % P is a cell array with numel(dims) elements, each of which is the sparsity pattern (from full to sparse) for the corresponding dims value.
            % Omitting dims will process all orders.
            % Omitting P, or providing an empty P, disables sparsity
            
            if nargin < 4
                P = [] ;
                if nargin < 3 || isempty(dims)
                    dims=1:x.order;
                end
            end
            x.space = unvectorize(x.space,sz,dims,P);
            x = x.updateProperties;
        end
        
        function sv = singularValues(x)
            % Returns the singular values of the tensor of order d
            %
            % sv = singularValues(x)
            % orthogonalizes the bases and returns the singular values of the core tensor
            %
            % x: TuckerLikeTensor
            
            if x.order  == 2
                x = orth(x);
                sv = svd(x.core.data);
            else
                switch class(x.core)
                    case 'DiagonalTensor'
                        error('Method not implemented for this tensor format.')
                    case 'FullTensor'
                        x = orth(x);
                        sv = singularValues(x.core);
                    case 'TTTensor'
                        x = orth(x);
                        d = x.order;
                        sv = cell(2,1);
                        sv{1} = singularValues(x.core);
                        sv{2} = cell(d,1);
                        for mu = 1:d
                            x = orth(x);
                            x.core = orth(x.core,mu);
                            [x.core.cores{mu},R] = orth(x.core.cores{mu},2);
                            x.space = spaceTimesMatrix(x.space,R',mu);
                            sv{2}{mu} = svd(x.space.spaces{mu});
                        end
                    case 'TreeBasedTensor'
                        x=orth(x);
                        sv = singularValues(x.core);
                        %sv{1} = s;
                        %sv{1}(x.core.tree.dim2ind)={[]};
                        %sv{2} = s(x.core.tree.dim2ind);
                        
                    otherwise
                        error('Method not implemented for this tensor format.')
                end
            end
        end
    end
    
    methods (Static)
        function z = create(generator,varargin)
            % Builds a TuckerLikeTensor of rank 1 using a given generator
            %
            % z=create(generator,sz1,sz2)
            % Builds a TuckerLikeTensor of type Vector (if nargin=3) or Operator (if nargin=4) using the function generator (randn, ones, ...).
            % The core is a DiagonalTensor
            %
            % sz1, sz2: array of length d
            
            d = numel(varargin{1});
            c = DiagonalTensor(1,d);
            if nargin == 2
                s = TSpaceVectors.create(generator,varargin{:});
            elseif nargin > 2
                s = TSpaceOperators.create(generator,varargin{:});
            end
            z = TuckerLikeTensor(c,s);
        end
        
        function z = rand(varargin)
            % Creates a tensor in TuckerLikeTensor format with i.i.d. parameters drawn according to the uniform distribution on (0,1)
            % z = rand(rank,sz1)
            % z = rand(rank,sz1,sz2)
            
            z = TuckerLikeTensor.create(@rand,varargin{:});
        end
        
        function z = randn(varargin)
            % Creates a tensor in TuckerLikeTensor format with i.i.d. parameters drawn according to the standard gaussian distribution
            % function z = randn(rank,sz1)
            % function z = randn(rank,sz1,sz2)
            
            z = TuckerLikeTensor.create(@randn,varargin{:});
        end
        
        function z = ones(varargin)
            % Creates a tensor in TuckerLikeTensor format with parameters equal to one
            % function z = ones(rank,sz1)
            % function z = ones(rank,sz1,sz2)
            
            z = TuckerLikeTensor.create(@ones,varargin{:});
        end
        function z = zeros(varargin)
            % Creates a tensor in TuckerLikeTensor format with parameters equal to zero
            % function z = zeros(rank,sz1)
            % function z = zeros(rank,sz1,sz2)
            
            z = TuckerLikeTensor.create(@zeros,varargin{:});
        end
        
        function I = eye(sz)
            %   Constructs the identity operator in TuckerLikeTensor format.
            %
            %   I = eye(sz) construct an operator of order numel(sz). The
            %   operator related to the dimension mu is sz(mu)
            
            d = numel(sz);
            Icore = DiagonalTensor(1,d);
            Ispace = cell(d,1);
            for mu = 1:d
                Ispace{mu} = {speye(sz(mu))};
            end
            Ispace = TSpaceOperators(Ispace);
            I = TuckerLikeTensor(Icore,Ispace);
        end
    end
end