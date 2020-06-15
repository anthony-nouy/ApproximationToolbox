% Class TSpaceVectors: tensor product of vector spaces
%
% See also TSpaceOperators

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

classdef TSpaceVectors < TSpace
    
    properties
        spaces % Cell array of matrices : u{i}(:,j) is the j-th vector of the i-th space
        order  % The order of the tensor space.
        dim    % Dimension of the subspaces involved in the tensor space.
        sz     % Dimensions of the underlying spaces
        isOrth % Flag indicating if each basis of each subspace is orthogonal.
    end
    
    methods
        function x = TSpaceVectors(s)
            % Constructor for the class TSpaceVectors
            %
            % x = TSpaceVectors(u) creates a tensor space from the basis vectors contained in u
            % u: cell array of matrices such that u{i}(:,j) is the jth vector of the ith space
            %
            % See also TSpaceOperators
            
            assert(isa(s,'cell'),'The input must be a cell.');
            x.spaces = s(:);
            x.isOrth = 0;
            x = updateProperties(x);
        end
        
        function s = TSpaceOperators(x)
            % Converts a TSpaceVectors into a TSpaceOperators
            
            s = cell(x.order,1);
            for mu=1:x.order
                s{mu} = cell(x.dim(mu),1);
                for i=1:x.dim(mu)
                    s{mu}{i}=x.spaces{mu}(:,i);
                end
            end
            s = TSpaceOperators(s);
        end
        
        function x = full(x)
            % Returns a representation with full storage
            
            x.spaces = cellfun(@(y) full(y),x.spaces,'uniformoutput',false);
        end
        
        function n = storage(x)
            % Returns the storage complexity
            
            n = sum(cellfun(@numel,x.spaces));
        end
        
        function n = sparseStorage(x)
            % Returns the sparse storage complexity
            
            n = sum(cellfun(@nnz,x.spaces));
        end
        
        function r = representationRank(x)
            % Returns the representation rank (dimensions of subspaces)
            
            r = cellfun(@(x) size(x,2), x.spaces(:)');
        end
        
        function x = cat(x,y)
            % Concatenates the elements of two TSpaceVectors
            % (sum of corresponding subspaces)
            %
            % x = cat(x,y) concatains the elements of x and y.
            % x,y: two TSpaceVectors of the same order and the same size
            
            x.spaces = cellfun(@(xs,ys) [xs,ys], x.spaces,y.spaces,'UniformOutput',0 );
            x.isOrth = 0;
            x = updateProperties(x);
        end
        
        function x = diagCat(x,y)
            % z = diagCat(x,y) concatanates the elements of x and y in a particular way.
            % x,y,z: TSpaceVectors of the same order
            % For Vx = x.spaces{mu} and Vy = y.spaces{mu}, Vz = z.spaces{mu}
            % is such that Vz is the block matrix [Vx , O ; 0,Vy]
            
            for mu = 1:x.order
                szx = size(x.spaces{mu});
                szy = size(y.spaces{mu});
                xmu = zeros(szx+szy);
                xmu(1:szx(1),1:szx(2)) = x.spaces{mu};
                xmu(szx(1)+1:end,szx(2)+1:end) = y.spaces{mu};
                x.spaces{mu} = xmu;
            end
            x.isOrth = 0;
            x = updateProperties(x);
        end
        
        function M = dot(x,y,dims)
            if nargin<3
                dims = 1:x.order;
            end
            M = cellfun(@(xs,ys) xs'*ys , x.spaces(dims),y.spaces(dims),'UniformOutput',0 );
        end
        
        function x = matrixTimesSpace(x,M,dims)
            if nargin<3
                dims = 1:x.order;
            end
            if isa(M,'double')
                M = {M};
            end
            x.spaces(dims) = cellfun(@(xs,m) m*xs, x.spaces(dims),M(:),'UniformOutput',0 );
            x.isOrth = 0;
            x = updateProperties(x);
        end
        
        function x = spaceTimesMatrix(x,M,dims)
            if nargin<3
                dims=1:x.order;
            end
            if isa(M,'double') && numel(dims) == 1
                M = {M};
            elseif ~isa(M,'cell') || numel(M) ~= numel(dims)
                error('Wrong input arguments.');
            end
            x.spaces(dims) = cellfun(@(xs,M) xs*M, x.spaces(dims),M,'UniformOutput',0);
            x.isOrth = 0;
            x = updateProperties(x);
        end
        
        function lc = evalInSpace(x,dim,c)
            % Evaluates a vector in a given subspace from its coefficients on the basis
            %
            % y = evalInSpace(x,dim,c)
            % Returns the vector in the dim-th subspace given its coefficients c in the basis
            % y = x.spaces{dim} * c
            
            lc = x.spaces{dim} * c;
        end
        
        function M = dotWithMetrics(x,y,A,order)
            % Inner products of basis vectors with a given metric provided by a positive definite operator
            %
            % M = dotWithMetrics(x,y,A)
            % x,y: TSpaceVectors
            % A: TSpaceOperators
            % M: cell
            %
            % The output is such that M{mu}(i1,i2,i3) = (x{mu}{i2})' * (A{mu}{i1}) * (y{mu}{i3}) for mu = 1...x.order
            %
            % function M = dotWithMetrics(x,y,A,dims)
            % returns a cell array of length length(dims) such that
            % M{l}(i1,i2,i3) = (x{dims(l)}{i2})' * (A{dims(l)}{i1}) * (y{dims(l)}{i3})
            % for l=1...length(dims)
            
            if nargin<4
                order = 1:x.order;
            end
            M = cell(x.order,1);
            for mu = order
                M{mu} = zeros(A.dim(mu),x.dim(mu),y.dim(mu));
                for m = 1:A.dim(mu)
                    M{mu}(m,:,:) = x.spaces{mu}'*A.spaces{mu}{m}*y.spaces{mu};
                end
            end
            M = M(order);
        end
        
        function [x,M] = orth(x,order)
            % Orthogonalization of bases
            %
            % [x,M] = orth(x,dims)
            % orthogonalization of the bases associated with dimensions in dims (by default dims = 1:x.order)
            
            if nargin<2
                order=1:x.order;
            end
            xs = x.spaces(order);
            M = cell(numel(order),1);
            for i = 1:numel(order)
                [xs{i},M{i}] = qr(xs{i},0);
            end
            x.spaces(order) = xs(:);
            if numel(order) == x.order
                x.isOrth=1;
            end
            x = updateProperties(x);
        end
        
        function x = updateProperties(x)
            x.order = length(x.spaces);
            x.dim = cellfun(@(xs)size(xs,2),x.spaces)';
            x.sz = cellfun(@(xs)size(xs,1),x.spaces)';
        end
        
        function x = evalAtIndices(x,I)
            % Evaluation of basis vectors at given indices
            % z = evalAtIndices(x,I)
            %
            % x: TSpaceVectors of order d
            % I: array of size N-by-d
            % z: TSpaceVectors of order d and size Nx...xN
            % z.spaces{mu} = z.spaces{mu}(I(:,k),:) for mu=1...d
            
            for k=1:x.order
                x.spaces{k} = x.spaces{k}(I(:,k),:);
            end
            x = updateProperties(x);
            x.isOrth = false;
        end
        
        function x = unvectorize(x,sz,dims,P)
            % Convert a TSpaceVectors into a TSpaceOperators
            %
            % x = unvectorize(x,sz,dims,P)
            % Convert the elements of the TSpaceVectors x to operators of size sz along dimensions dims
            %
            % dims: 1-by-k array (dims = 1:x.order by default)
            % sz: 2-by-numel(dims) matrix
            % P: cell array with numel(dims) elements, each of which is the sparsity pattern (from full to sparse) for the corresponding dims value.
            % Omitting P, or providing an empty P, disables sparsity
            
            if nargin < 4
                useSparsity = false; % Sparsity not requested
                if nargin<3 || isempty(dims)
                    dims=1:x.order;
                end
            else
                useSparsity = ~isempty(P); % True unless P is empty (safety)
            end
            for mu = dims
                x.spaces{mu} = mat2cell(x.spaces{mu},x.sz(mu),ones(1,x.dim(mu)))' ;
                for i = 1:x.dim(mu)
                    if useSparsity
                        x.spaces{mu}{i} = sparse(P{dims==mu},1,...
                            x.spaces{mu}{i},prod(sz(:,mu)),1) ;
                    end
                    x.spaces{mu}{i} = reshape(x.spaces{mu}{i},sz(:,mu)') ;
                end
            end
            x = TSpaceOperators(x.spaces);
            x = updateProperties(x);
            x.isOrth=0;
        end
    end
    
    methods (Static)
        function x = create(generator,sz,dim)
            % Creates a TSpaceVectors from a given generator
            %
            % x = create(generator,sz,dim)
            % creates a TspaceVectors of size sz
            % dim is an array containing the dimensions of the spaces
            % dim is equal to [0,...0] by default
            % generator is a pointer to a function generator(n,m) that generates a matrix of size n-by-m, such as rand, zeros, ones, randn...
            
            sz = sz(:);
            if nargin < 3
                dim = ones(size(sz));
            end
            dim = dim(:);
            x = TSpaceVectors(cellfun(@(s,d) generator(s,d),...
                num2cell(sz),...
                num2cell(dim),'UniformOutput',false));
        end
        
        function x = zeros(sz,varargin)
            % Creates a TSpaceVectors with given dimensions whose basis vectors have zero entries
            % x = zeros(sz,dim)
            % See also TSpaceVectors/create
            
            x = TSpaceVectors.create(@zeros,sz,varargin{:});
        end
        function x = ones(sz,varargin)
            % Creates a TSpaceVectors with given dimensions whose basis vectors have entries equal to 1
            % x = ones(sz,dim)
            % See also TSpaceVectors/create
            
            x = TSpaceVectors.create(@ones,sz,varargin{:});
        end
        function x = rand(sz,varargin)
            % Creates a TSpaceVectors with given dimensions whose basis vectors have entries generated with rand
            % x = rand(sz,dim)
            % See also TSpaceVectors/create
            
            x = TSpaceVectors.create(@rand,sz,varargin{:});
        end
        function x = randn(sz,varargin)
            % Creates a TSpaceVectors with given dimensions whose basis vectors have entries generated with randn
            % x = randn(sz,dim)
            % See also TSpaceVectors/create
            
            x = TSpaceVectors.create(@randn,sz,varargin{:});
        end
        function x = eye(sz,varargin)
            % Creates a TSpaceVectors with given dimensions whose basis vectors are the canonical vectors
            % x = eye(sz,dim)
            % dim = sz by default
            % See also TSpaceVectors/create
            
            if nargin==1
                varargin={sz};
            end
            x = TSpaceVectors.create(@eye,sz,varargin{:});
        end
        function x = speye(sz,varargin)
            % Creates a TSpaceVectors with given dimensions whose basis vectors are the canonical vectors (sparse storage)
            % x = speye(sz,dim)
            % dim = sz by default
            % See also TSpaceVectors/create
            
            if nargin==1
                varargin={sz};
            end
            x = TSpaceVectors.create(@speye,sz,varargin{:});
        end
    end
end