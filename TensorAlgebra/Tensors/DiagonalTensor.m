% Class DiagonalTensor: diagonal algebraic tensors

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

classdef DiagonalTensor < AlgebraicTensor
    
    properties
        data   % Diagonal entries
        order  % Order of the tensor
        sz     % Size of the tensor
        isOrth = 1 % The flag is true if the representation of the tensor is orthogonal
    end
    
    methods
        function x = DiagonalTensor(y,varargin)
            % DiagonalTensor - Constructor of the class DiagonalTensor
            %
            % x = DiagonalTensor(y,d)
            % Creates a diagonal tensor x of order d and size nx...xn such that x(i,...,i) = y(i)
            %
            % y: array of length n
            % d: integer
            
            switch nargin
                case 1
                    assert(isa(y,'DiagonalTensor'),['Input must be of ' ...
                        'DiagonalTensor type.']);
                    x = y;
                case 2
                    assert(isa(y,'double'),['Input must be of ' ...
                        'double type.']);
                    d = varargin{1};
                    x.data = y(:);
                    x.order = d;
                    x.sz = length(y)*ones(1,d);
                otherwise
                    error('Wrong number of input arguments.');
            end
        end
        
        function x = treeBasedTensor(y,varargin)
            % Conversion into a TreeBasedTensor
            %
            % x = treeBasedTensor(y,tree,isActiveNodes)
            % Converts a DiagonalTensor y into a TreeBasedTensor
            %
            % y: DiagonalTensor
            % tree: DimensionTree (linear tree by default)
            % isActiveNodes = logical of size 1-by-tree.nbNodes (~tree.isLeaf by default)
            % x: TreeBasedTensor
            
            if nargin==1 || isempty(varargin{1})
                x.tree=DimensionTree.linear(y.order);
            else
                x.tree=varargin{1};
            end
            if nargin<3
                isActiveNode = true(1,x.tree.nbNodes);
                isActiveNode(x.tree.isLeaf)=false;
            else
                isActiveNode = varargin{2};
            end
            
            x.tensors = cell(1,x.tree.nbNodes);
            r = y.sz(1);
            for i = 1:x.tree.nbNodes
                c = x.tree.children(:,i);
                if x.tree.parent(i) == 0
                    ord = nnz(c);
                    x.tensors{i} = FullTensor.diag(y.data,ord);
                elseif x.tree.isLeaf(i) && isActiveNode(i)
                    x.tensors{i} = FullTensor(eye(r),2,[r,r]);
                elseif isActiveNode(i)
                    ord = nnz(c) + 1;
                    x.tensors{i} = FullTensor.diag(ones(r,1),ord);
                elseif ~x.tree.isLeaf(i) && ~isActiveNode(i)
                    error('Internal nodes should be active.')
                end
            end
            x = TreeBasedTensor(x.tensors,x.tree);
        end
        
        function n = storage(x)
            n = length(x.data);
        end
        
        function n = sparseStorage(x)
            n = nnz(x.data);
        end
        
        function x = plus(x,y)
            x.data = x.data + y.data;
        end
        
        function x = minus(x,y)
            x.data = x.data-y.data;
        end
        
        function x = uminus(x)
            x.data = -x.data;
        end
        
        function x = mtimes(x,s)
            if isa(x,'DiagonalTensor') && isa(s,'double')
                x.data = x.data*s;
            elseif isa(x,'double') && isa(s,'DiagonalTensor')
                x = s*x;
            else
                error('Wrong type of input arguments.')
            end
        end
        
        function x = reshape(x,s)
            % Reshapes the diagonal tensor (this function has no effect)
            
            assert(all(s == x.sz(1)),...
                's should be a uniform vector such that s(i) = x.sz(1).');
            x.sz = s(:)';
        end
        
        function x = times(x,y)
            x.data = x.data.*y.data;
        end
        
        function x = subTensor(x,varargin)
            x = full(x);
            x = subTensor(x,varargin{:});
        end
        
        function x = updateProperties(x)
            % Updates the property sz from the property data
            
            x.sz(:) = length(x.data);
        end
        
        function x = timesVector(x,V,varargin)
            if isa(V,'double')
                V = {V};
            end
            x.data = x.data.*prod(horzcat(V{:}),2);
            x.order = x.order-length(V);
            x.sz = x.sz(1:x.order);
            if x.order==0
                x = sum(x.data);
            end
        end
        
        function x = timesMatrix(x,M,varargin)
            x = full(x);
            x = timesMatrix(x,M,varargin{:});
        end
        
        function x = timesDiagMatrix(x,M,varargin)
            if iscell(M)
                M = [M{:}] ;
            end
            x.data = prod(M,2).*x.data;
        end
        
        function z = timesTensor(x,y,xd,yd)
            % Contraction of two tensors along given dimensions
            %
            % z = timesTensor(x,y,xd,yd)
            % First converts tensors x and y into FullTensor
            %
            % See also FullTensor/timesTensor
            
            x = full(x);
            y = full(y);
            z = timesTensor(x,y,xd,yd);
        end
        
        function n = dot(x,y)
            n = dot(x.data,y.data);
        end
        
        function n = norm(x)
            n = norm(x.data);
        end
        
        function x = full(x)
            x = FullTensor.diag(x.data,x.order);
        end
        
        function y = sparse(x)
            % Conversion into a SparseTensor
            %
            % y = sparse(x)
            % Converts a DiagonalTensor to a SparseTensor
            % x: DiagonalTensor
            % y: SparseTensor
            
            ind = find(x.data);
            dat = nonzeros(x.data);
            indices = MultiIndices(repmat(ind,1,x.order));
            y = SparseTensor(dat, indices, x.sz);
        end
        
        function x = TTTensor(x)
            % Conversion into TTTensor
            
            cores = cell(x.order,1);
            xr = x.sz(1);
            szz = [1 xr xr];
            z = reshape(diag(x.data),szz);
            cores{1} = FullTensor(z,3,szz);
            z = ones(xr,1);
            for mu = 2:x.order-1
                cores{mu} = FullTensor.diag(z,3);
            end
            cores{x.order} = FullTensor(eye(xr),3,[xr xr 1]);
            x = TTTensor(cores);
        end
        
        function x = cat(x,y)
            x.data = [x.data;y.data];
            x.sz = x.sz + y.sz;
        end
        
        function x = kron(x,y)
            xa = y.data*x.data.';
            x.data = xa(:);
            x.sz = x.sz.*y.sz;
        end
        
        function z = dotWithRankOneMetric(x,y,M)
            MM = [M{:}];
            MM = reshape(full(MM),x.sz(1),y.sz(1),x.order);
            MM = prod(MM,3);
            z = x.data'*MM*y.data;
        end
        
        function z = timesTensorTimesMatrixExceptDim(x,y,M,order)
            ord = 1:x.order;
            ord(order)=[];
            MM = [M{ord}];
            MM = reshape(full(MM),x.sz(1),y.sz(1),numel(ord));
            MM = prod(MM,3);
            z = MM.*(x.data*y.data');
        end
        
        function x = orth(x)
        end
        
        function x = evalDiag(x,dims)
            if nargin==1 || (numel(dims)==x.order)
                x = x.data;
            else
                dims = sort(dims);
                rep = [1:dims(1),dims(end)+1:x.order];
                x.sz = x.sz(rep);
                x.order = numel(rep);
            end
        end
        
        function s = evalAtIndices(x,I,dims)
            r = true(size(I,1),1);
            for k=1:size(I,2)
                r = r & (I(:,1)==I(:,k));
            end
            
            if nargin<=2 || numel(dims)==x.order
                s = zeros(size(I,1),1);
                s(r)=x.data(I(r,1));
            else
                error('Method not implemented.')
            end
        end
        
        function x = permute(x,dims)
            % Permutes array dimensions
            % x = permute(x,dims)
            %
            % See also permute
            
            x.sz = x.sz(dims);
            x.sz = x.sz(:)';
        end
    end
    
    methods (Static)
        function x = create(generator,rank,order)
            % Creates a DiagonalTensor from a given generator
            %
            % x = create(generator,rank,order)
            
            a = generator(rank,1);
            x = DiagonalTensor(a,order);
        end
        
        function x = rand(rank,order)
            % Creates a DiagonalTensor with entries obtained from generator rand
            %
            % x = rand(rank,order)
            
            x = DiagonalTensor.create(@rand,rank,order);
        end
        
        function x = zeros(rank,order)
            % Creates a DiagonalTensor with zero entries
            %
            % x = zeros(rank,order)
            
            x = DiagonalTensor.create(@zeros,rank,order);
        end
        
        function x = randn(rank,order)
            % Creates a DiagonalTensor with entries obtained from generator randn
            %
            % x = randn(rank,order)
            
            x = DiagonalTensor.create(@randn,rank,order);
        end
        
        function x = ones(rank,order)
            % Creates a DiagonalTensor with all entries equal to 1
            %
            % x = ones(rank,order)
            
            x = DiagonalTensor.create(@ones,rank,order);
        end
    end
end