% Class FullTensorGrid: tensor product grid
%
% See also SparseTensorGrid

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

classdef FullTensorGrid < TensorGrid
    
    properties
        dim   % dimension
        grids % cell array containing the grids
        sz    % sizes of the grids
    end
    
    methods
        function G = FullTensorGrid(grids,dim)
            % Constructor for the class FullTensorGrid
            %
            % G = FullTensorGrid(grids)
            % grids: cell array of length dim containing arrays of size n_ixd_i, 1<=i<=dim.
            %
            % G = FullTensorGrid(grid,dim)
            % provides an isotropic tensor grid G
            % G is grid x ... x grid (dim times)
            % grid: array of size nxd
            % dim: dimension
            
            if nargin==1
                dim = length(grids);
            else
                if ~isa(grids,'double')
                    error('The first argument must be a double (unidimensional grid).')
                end
                grids = repmat({grids},1,dim);
            end
            
            G.dim = dim;
            G.sz = zeros(1,dim);
            for i=1:G.dim
                G.sz(i) = size(grids{i},1);
            end
            G.grids = grids;
        end
        
        function x = array(G)
            I = multiIndices(G);
            x = cell(1,G.dim);
            for i=1:G.dim
                x{i}=G.grids{i}(I.array(:,i),:);
            end
            x = [x{:}];
        end
        
        function n = numel(G)
            n = prod(G.sz);
        end
        
        function x = multiIndices(G)
            % Returns a set of multi-indices for indexing the grid
            %
            % x = multiIndices(G)
            % G: FullTensorGrid
            % x: MultiIndices
            
            x = MultiIndices.boundedBy(G.sz,1);
        end
        
        function plot(x,y,varargin)
            % Plots the grid
            %
            % plot(x,y,varargin)
            % varargin: plotting parameters
            
            d = ndims(x);
            switch d
                case 1
                    plot(x.grids{1},y,varargin{:});
                case 2
                    y = reshape(y,x.sz);
                    surf(x.grids{1},x.grids{2},y,varargin{:});
                otherwise
                    error('Dimension must be less than 2.')
            end
        end
    end
    
    methods (Static)
        function G = random(X,n)
            % Generation of a random FullTensorGrid grom a RandomVector
            %
            % G = random(X,n)
            % generate the tensor product of grids of d grids of size n(k)
            % X: RandomVector
            % n: array of size d (or an integer for isotropic grid)
            % G: FullTensorGrid
            
            d = numel(X);
            if numel(n)==1
                n=repmat(n,1,d);
            end
            G = cell(1,d);
            for k=1:d
                G{k}=random(X.randomVariables{k},n(k),1);
            end
            G = FullTensorGrid(G);
        end
    end
end