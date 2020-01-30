% Class FullTensorProductIntegrationRule

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

classdef FullTensorProductIntegrationRule < IntegrationRule
    
    methods
        function G = FullTensorProductIntegrationRule(points,weights)
            % FullTensorProductIntegrationRule : Class constructor
            % G = FullTensorProductIntegrationRule(points,weights)
            % points: cell of length dim or FullTensorGrid containing 1D points
            % weights: cell of length dim containing 1D weights
            
            G@IntegrationRule(points,weights)
            if isa(points,'cell')
                points = FullTensorGrid(points);
            elseif ~isa(points,'FullTensorGrid')
                error('points must be a FullTensorGrid or a cell')
            end
            if ~isa(weights,'cell') || length(weights)~=ndims(points)
                error('weights must be a cell of length dim')
            end
            G.points = points;
            for k=1:length(weights)
                weights{k} = weights{k}(:);
            end
            G.weights = weights(:);
        end
        
        function n = ndims(I)
            n = ndims(I.points);
        end
        
        function y = integrate(I,f)
            % y = integrate(I,f)
            % Integration of function f.
            % call f(x) with an array x of size numel(I)xd, where d is the dimension
            x = array(I.points);
            w = weightsOnGrid(I);
            y = w'*f.eval(x);
        end
        
        function w = weightsOnGrid(I)
            w = CanonicalTensor(I.weights,1);
            w = double(full(w));
            w = w(:);
        end
    end
end