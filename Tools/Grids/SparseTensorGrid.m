% Class SparseTensorGrid: sparse tensor product grid
%
% See also FullTensorGrid

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

classdef SparseTensorGrid < TensorGrid
    
    properties
        dim     % dimension
        grids   % cell array containing the grids
        sz      % sizes of the grids
        indices % multi-indices
    end
    
    methods
        
        function G = SparseTensorGrid(T,indices,varargin)
            % Constructor for the class SparseTensorGrid
            %
            % G = SparseTensorGrid(grids,indices)
            % grids: cell array of length dim containing arrays of size n_ix1, 1<=i<=dim.
            % indices: MultiIndices (indices start at 1)
            %
            % G = SparseTensorGrid(grid,indices,dim)
            % provides an isotropic tensor grid
            % grid: array of size nx1 (unidimensional grid)
            % dim: dimension
            
            T = FullTensorGrid(T,varargin{:});
            G.dim = T.dim;
            G.grids = T.grids;
            G.sz = T.sz;
            G.indices = indices;
        end
        
        
        function n = numel(G)
            n = numel(G.indices);
        end
        
        function x = array(G)
            I = cell(G.indices);
            x = G.grids;
            for k=1:length(x)
                x{k} = x{k}(I{k},:);
            end
            x = [x{:}];
        end
    end
end