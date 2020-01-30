% Class SparseTensorProductIntegrationRule

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

classdef SparseTensorProductIntegrationRule < IntegrationRule
    
    properties
        dim
    end
    
    methods
        function G = SparseTensorProductIntegrationRule(points,weights)
            % class SparseTensorProductIntegrationRule
            % G = SparseTensorProductIntegrationRule(points,weights)
            % points: SparseTensorGrid
            % weights
            
            G.points = points;
            G.weights = weights;
        end
        
        
        function y = integrate(I,f)
            % y = integrate(I,f)
            % Integration of function f.
            % call f(x) with an array x of size numel(I)xd, where d is the
            % dimension
            x = array(I.points);
            w = weights(:);
            y = w'*f.eval(x);
        end
    end
end