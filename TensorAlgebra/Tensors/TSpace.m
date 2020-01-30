% Class TSpace: abstract class for tensor product spaces
%
% See also TSpaceVectors, TSpaceOperators

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

classdef (Abstract) TSpace
    
    properties (Abstract)
        % Cell containing the basis.
        spaces
        % The order of the tensor space.
        order
        % Dimension of each subspace involved in the tensor space.
        dim
        % Dimension of the space containing each subspace.
        sz
        % Flag indicating if each basis of each subspace is orthogonal.
        isOrth
    end
    
    methods
        function x = permute(x,dims)
            % Permutes spaces
            % x: TSpace
            % dims: 1-by-x.order array
            
            if numel(dims)~=x.order
                error('Wrong arguments.')
            end
            x.spaces = x.spaces(dims);
            x = updateProperties(x);
        end
    end
    
    methods (Abstract)
        % CAT - Concatenate subspaces
        x = cat(x,y)
        
        % DOT - Inner product between elements of the subspaces
        s = dot(x,y)
        
        % EVALINSPACE - Evaluates a vector (or operator) in a subspace from its coefficients in the basis
        v = evalInSpace(x,coef,varargin)
        
        % MATRIXTIMESSPACE - Multiplication of the bases vectors (or matrices) by matrices
        % For the k-th subspace with basis v^k_1,...v^k_n (of vectors or matrices)
        % returns a new basis Mk*v^k_1,...,Mk*v^ _n
        %
        % v = matrixTimesSpace(x,M)
        % v.spaces{k} = Mk * x.spaces{k} for all k
        % M is a cell array with x.order elements containing matrices Mk = M{k}
        %
        % v = matrixTimesSpace(x,M,dims)
        % for dimensions k not in dims, Mk is taken as the identity
        % M is a cell array with numel(dims) elements, and Mk = M{dims(i)} with k=dims(i)
        x = matrixTimesSpace(x,M,varargin)
        
        % SPACETIMESMATRIX - Linear combinations of basis vectors (or matrices)
        % For the k-th subspace with basis v^k_1,...v^k_n (of vectors or matrices)
        % returns a new basis w^k_1,...,w^k_m, with
        % w^k_i = sum_j v^k_i A_ij, where M is a given matrix
        %
        % x = spaceTimesMatrix(x,M)
        % x.spaces{k} = x.spaces{k}*M{k} for all k
        %
        % x = spaceTimesMatrix(x,M,dims)
        % x.spaces{dims(i)} = x.spaces{dims(i)}*M{i} for all i = 1...length(dims)
        x = spaceTimesMatrix(x,M,varargin)
        
        % UPDATEPROPERTIES - Updates the properties
        % x = updateProperties(x) updates the properties of x
        x = updateProperties(x)
    end
    
    methods
        function [x,M] = orth(x,dims)
            % Orthogonalization of bases
            %
            % [x,M] = orth(x,dims)
            % orthogonalization of the bases associated with dimensions in dims (by default dims = 1:x.order)
            
            if nargin<2
                dims=1:x.order;
            end
            tol = 1e-16;
            xu = x.spaces(dims);
            N = dot(x,x,dims);
            M = cell(numel(dims),1);
            for i=1:numel(dims)
                [L,d,~] = svd(N{i},0);
                d = diag(d);
                err = sqrt(1-cumsum(d.^2)/sum(d.^2));
                m = find(err<tol);
                if isempty(m)
                    m = x.dim(dims(i));
                else
                    m = min(m);
                end
                L = L(:,1:m);
                d = d(1:m);
                xu{i} = L*diag(1./sqrt(d));
                M{i} = diag(sqrt(d))*L' ;
            end
            x = spaceTimesMatrix(x,xu,dims);
            if numel(dims)==x.order
                x.isOrth=1;
            end
        end
        function x = keepSpace(x,dims)
            % Keeps spaces associated with given dimensions
            %
            % y = keepSpace(x,dims)
            % x: TSpace
            % dims: array
            % y: TSpace of order length(dims)
            
            x.spaces = x.spaces(dims);
            x = updateProperties(x);
        end
        
        function x = removeSpace(x,dims)
            % Remove spaces associated with given dimensions
            %
            % y = removeSpace(x,dims)
            % x: TSpace
            % dims: array
            % y: TSpace of order length(dims)
            
            x = keepSpace(x,setdiff(1:x.order,dims));
        end
    end
end