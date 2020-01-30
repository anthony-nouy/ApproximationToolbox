% Class TensorGrid: grids in product sets
%
% See also FullTensorGrid, SparseTensorGrid

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

classdef (Abstract) TensorGrid
    
    properties (Abstract)
        dim   % dimension
        grids % cell array containing the grids
        sz    % sizes of the grids
    end
    
    methods
        
        function n = ndims(x)
            % Returns the dimension of the underlying space
            
            n = x.dim;
        end
        
        function s = size(x,d)
            % Returns the size of the grid
            % s = size(x) returns the size along all dimensions of the grid. It is equivalent to x.sz.
            % s = size(x,d) returns the size of the grid along dimension d. It is equivalent to x.sz(d).
            
            if nargin == 1
                s = x.sz;
            else
                s = x.sz(d);
            end
        end
        
        function plotGrid(x,varargin)
            % Plots the grid
            %
            % plotGrid(x,varargin)
            % varargin: parameters for plot
            
            x = array(x);
            d=size(x,2);
            switch d
                case 1
                    plot(x(:,1),varargin{:});
                case 2
                    plot(x(:,1),x(:,2),varargin{:});
                case 3
                    plot3(x(:,1),x(:,2),x(:,3),varargin{:});
                otherwise
                    error('Dimension must be less than 3.')
            end
        end
    end
    
    methods (Abstract)
        % ARRAY - Returns an array of size nxd where n is the number of grid points and d is the dimension
        x = array(G)
        
        % NUMEL - Returns the number of grid points
        x = numel(G)
    end
end