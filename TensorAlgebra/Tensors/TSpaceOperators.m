% Class TSpaceOperators: tensor product of spaces of operators
%
% See also TSpaceVectors

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

classdef TSpaceOperators < TSpace
    
    properties
        spaces % Cell array of cell array of operator : spaces{i}{j} is the j-th operator of the i-th subspace
        order  % The order of the tensor space.
        dim    % Dimension of the subspaces involved in the tensor space.
        sz     % Matrix containing the dimension of the operator of the ith TSpace : sz(i,j) = size(spaces{i}{1},j)
        isOrth % Flag indicating if each basis of each subspace is orthogonal.
    end
    
    methods
        function x = TSpaceOperators(s)
            % Constructor for the class TSpaceOperators
            %
            % x = TSpaceOperators(u) creats the tensor space of operator contained in u.
            % u: cell array of cell array of matrices so that u{i}{j} is the jth operator of the ith subspace
            %
            % See also TSpaceVectors
            
            assert(isa(s,'cell'),'The input must be a cell.');
            if ~isa(s{1},'cell')
                s = cellfun(@(x) {x},s,'uniformoutput',false);
            end
            
            x.spaces = s(:);
            x = updateProperties(x);
            x.isOrth=0;
        end
        
        function n=storage(x)
            % Returns the storage complexity
            
            n = sum(cellfun(@(c) sum(cellfun(@numel,c)),x.spaces));
        end
        
        function n=sparseStorage(x)
            % Returns the sparse storage complexity
            
            n = sum(cellfun(@(c) sum(cellfun(@nnz,c)),x.spaces));
        end
        
        function x = cat(x,y)
            % Concatenates the elements of two TSpaceOperators
            %
            % x = cat(x,y) concatenates the elements of x and y.
            % x,y: two TSpaceOperators of the same order and the same size
            
            x.spaces = cellfun(@(u,v) [u;v] ,x.spaces,y.spaces,'UniformOutput',0);
            x = updateProperties(x);
            x.isOrth=0;
        end
        
        function r = representationRank(x)
            % Returns the representation rank (dimensions of subspaces)
            
            r = cellfun(@(x) length(x), x.spaces(:)');
        end
        
        function y = mtimes(x,y,dims)
            % y = mtimes(x,y) multiplies each components of x and y
            %
            % y = mtimes(x,y,dims) multiplies each components of x and y associated with dimensions in dims
            
            if nargin<3
                dims=1:x.order;
            end
            xs = x.spaces;
            if isa(y,'TSpaceVectors')
                for mu=dims
                    xymu=cellfun(@(xs) xs*y.spaces{mu},xs{mu},'UniformOutput',0);
                    y.spaces{mu}=[xymu{:}];
                end
            elseif isa(y,'TSpaceOperators')
                for mu=dims
                    [I,J] = ind2sub([y.dim(mu),x.dim(mu)],...
                        1:(x.dim(mu)*y.dim(mu)));
                    y.spaces{mu} = cellfun( @(xs,ys) xs*ys, xs{mu}(J(:)), ...
                        y.spaces{mu}(I(:)),'UniformOutput',0);
                end
            end
            y.spaces = y.spaces(dims);
            y.isOrth = 0;
            y = updateProperties(y);
        end
        
        function M = dot(x,y,dims)
            if nargin<3
                dims = 1:x.order;
            end
            M = cell(x.order,1);
            for mu = dims
                M{mu} = zeros(x.dim(mu),y.dim(mu));
                for i = 1:x.dim(mu)
                    for j = 1:y.dim(mu)
                        M{mu}(i,j) = x.spaces{mu}{i}(:)'*y.spaces{mu}{j}(:);
                    end
                end
            end
            M = M(dims);
        end
        
        function x = matrixTimesSpace(x,M,dims)
            if nargin<3
                dims=1:x.order;
            end
            if isa(M,'double')
                M = {M};
            end
            x.spaces(dims) = cellfun(@(x,M) cellfun(@(xs) M*xs,x, ...
                'uniformoutput',false), ...
                x.spaces(dims),M(:),'uniformoutput',false);
            x.isOrth = 0;
            x = updateProperties(x);
        end
        
        function x = spaceTimesMatrix(x,M,dims)
            if nargin<3
                dims = 1:x.order;
            end
            if isa(M,'double') && numel(dims) == 1
                M = {M};
            elseif ~isa(M,'cell') || numel(M) ~= numel(dims)
                error('Wrong input arguments.');
            end
            xs = cell(numel(dims),1);
            z = 1;
            for mu = dims
                Mmu = M{z};
                xs{mu} = cell(size(Mmu,2),1);
                for i = 1:size(Mmu,2)
                    for k = 1:size(Mmu,1)
                        xs{mu}{i} = x.spaces{mu}{k}*Mmu(k,i);
                    end
                end
                z = z + 1;
            end
            x = TSpaceOperators(xs);
            x.isOrth = 0;
            x = updateProperties(x);
        end
        
        function xc = evalInSpace(x,dim,c)
            % Evaluates a matrix in a given subspace of matrices from its coefficients on the basis
            %
            % A = evalInSpace(x,dim,c)
            % Returns the matrix A in the dim-th subspace given its coefficients c in the basis
            % A = sum_i c_i x.spaces{dim}{i}
            
            xc = x.spaces{dim}{1} * c(1);
            for i=2:x.dim(dim)
                xc = xc + x.spaces{dim}{i} * c(i);
            end
        end
        
        function x = ctranspose(x)
            % Applies ctranspose to all operators
            % x = ctranspose(x)
            
            for mu=1:x.order
                x.spaces{mu} = cellfun(@(t)t',x.spaces{mu},'UniformOutput',0);
            end
            x = updateProperties(x);
        end
        
        function x = transpose(x)
            % Applies transpose to all operators
            % x = transpose(x)
            
            for mu=1:x.order
                x.spaces{mu} = cellfun(@(t)t.',x.spaces{mu},'UniformOutput',0);
            end
            x = updateProperties(x);
        end
        
        function x = updateProperties(x)
            % Update properties
            
            x.order=length(x.spaces);
            x.dim=cellfun(@(xs)length(xs),x.spaces)';
            x.sz=zeros(2,x.order);
            for mu=1:x.order
                x.sz(:,mu)=size(x.spaces{mu}{1});
            end
        end
        
        function [x,P] = vectorize(x,dims)
            % Vectorization of the elements of a TSpaceOperators
            %
            % [x,P] = vectorize(x,dims) vectorizes the elements of the TspaceOperators x along orders dims.
            % P is sparsity pattern (from full to sparse) in vector format
            % A single output argument disables sparsity handling
            
            if nargin<2 || isempty(dims)
                dims=1:x.order;
            end
            useSparsity = nargout > 1 ; % Whether sparsity exploitation is requested
            if useSparsity
                P = cell(numel(dims),1) ;
                for mu = dims
                    unionP = [] ;
                    for i = 1:x.dim(mu)
                        unionP = union(unionP,find(x.spaces{mu}{i}));
                        x.spaces{mu}{i} = sparse(x.spaces{mu}{i}(:));
                    end
                    x.spaces{mu} = [x.spaces{mu}{:}];
                    x.spaces{mu} = x.spaces{mu}(unionP,:);
                    x.spaces{mu} = full(x.spaces{mu}) ;
                    P{dims==mu} = unionP ;
                end
            else
                for mu = dims
                    for i = 1:x.dim(mu)
                        x.spaces{mu}{i} = x.spaces{mu}{i}(:);
                    end
                    x.spaces{mu}=full([x.spaces{mu}{:}]);
                end
            end
            x = TSpaceVectors(x.spaces);
            x = updateProperties(x);
            x.isOrth=0;
        end
        
        function [x,M] = orth(x,order)
            % Orthogonalization of bases [To be tested]
            %
            % [x,M] = orth(x,dims)
            % orthogonalization of the bases associated with dimensions in dims (by default dims = 1:x.order)
            % TSpacesVectors then call its orth method. Exploits sparsity.
            
            if nargin < 2
                order = 1:x.order;
            end
            xSz = x.sz;
            [v,P] = vectorize(x,order);
            [v,M] = orth(v,order);
            x = unvectorize(v,xSz,order,P);
            x = updateProperties(x);
        end
    end
    
    methods (Static)
        function x = randn(varargin)
            % Creates a TSpaceOperators with given dimensions whose basis matrices have entries generated with randn
            % x = randn(sz,dim)
            % See also TSpaceOperators/create
            
            x = TSpaceOperators.create(@randn,varargin{:});
        end
        
        function x = rand(varargin)
            % Creates a TSpaceOperators with given dimensions whose basis matrices have entries generated with rand
            % x = rand(sz,dim)
            % See also TSpaceOperators/create
            
            x = TSpaceOperators.create(@rand,varargin{:});
        end
        
        function x = ones(varargin)
            % Creates a TSpaceOperators with given dimensions whose basis matrices have entries equal to 1
            % x = ones(sz,dim)
            % See also TSpaceOperators/create
            
            x = TSpaceOperators.create(@ones,varargin{:});
        end
        
        function x = zeros(varargin)
            % Creates a TSpaceOperators with given dimensions whose basis matrices have zero entries
            % x = zeros(sz,dim)
            % See also TSpaceOperators/create
            
            x = TSpaceOperators.create(@zeros,varargin{:});
        end
        
        function x = eye(varargin)
            % Creates a TSpaceOperators with given dimensions whose basis matrices are identity
            % x = eye(sz1,sz2,dim)
            % sz2=sz1 by default
            % dim = 1 by default
            % See also TSpaceOperators/create
            
            x = TSpaceOperators.create(@eye,varargin{:});
        end
        
        function x = speye(varargin)
            % Creates a TSpaceOperators with given dimensions whose basis matrices are identity matrices (sparse storage)
            % x = speye(sz1,sz2,dim)
            % sz2=sz1 by default
            % dim = 1 by default
            % See also TSpaceOperators/create
            
            x = TSpaceOperators.create(@speye,varargin{:});
        end
        
        function x = create(generator,sz1,sz2,dims)
            % Creates a TSpaceOperators from a given generator
            %
            % x = create(generator,sz1,sz2,dim)
            % creates a TspaceOperators of size sz1xsz2
            % dims is an array containing the dimensions of the spaces
            % dims is equal to [1,...1] by default
            % generator is a pointer to a function generator(n,m) that generates a matrix of size n-by-m, such as rand, zeros, ones, randn...
            d = numel(sz1);
            if nargin < 4
                dims = ones(d,1);
                if nargin < 3
                    sz2 = sz1;
                end
            end
            sz1 = sz1(:);
            sz2 = sz2(:);
            u = cell(d,1);
            for mu = 1:d
                u{mu} = cell(dims(mu),1);
                u{mu} = cellfun(@(x) generator(sz1(mu),sz2(mu)),u{mu},'uniformoutput',false);
            end
            x = TSpaceOperators(u);
        end
    end
end