% Class IntegrationRule

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

classdef IntegrationRule

    properties
        points
        weights
    end
    
    methods
        
        function I = IntegrationRule(points,weights)
            % I = IntegrationRule(points,weights)
            % points: integration points, array of size nxd
            % weights: integration weights, array of size 1xn
            
            I.points = points;
            I.weights = weights(:).';
            if length(weights)~=size(points,1)
                error('number of points is not equal to number of weights')
            end
        end
        
        function n = ndims(I)
            n = size(I.points,2);
        end
        
        function ok = eq(~,J)
            ok = true;
            if ~isa(J,'IntegrationRule')
                ok = false;
            else
                ok = false;
                warning('to be improved')
            end

        end

        function y = integrate(I,f)
            % y = integrate(I,f)
            % Integration of function f.
            % call f(x) with an array x of size numel(I.points)xd, where d is the dimension
            % f(x) should return an array of size numel(I.points) x m
            % Then y is of size m-by-1
            
            fx = f(I.points);
            y = (I.weights(:).'*fx).';
            
        end
        
        function I = tensorize(I,d)
            p = FullTensorGrid(I.points(:),d);
            w = repmat({I.weights},d,1);
            I = FullTensorProductIntegrationRule(p,w);
        end

        function mu = discreteMeasure(I)
            mu = DiscreteMeasure(I.points,I.weights);
        end
    end
    
    methods (Static)
        function I = gauss(rv,varargin)
            % gauss(rv,varargin)
            % rv: RandomVariable or RandomVector
            % call the method gaussIntegrationRule of RandomVariable or RandomVector
            
            I = rv.gaussIntegrationRule(varargin{:});
        end
    end
end
