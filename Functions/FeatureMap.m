% Class FeatureMap

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

classdef FeatureMap < Function
    
    properties
        featureMap
    end
    
    methods
        function f = FeatureMap(map)
            % f = FeatureMap(map)
            % Constructor for the FeatureMap class, that defines a feature map
            % map: function
            % f: FeatureMap
            
            f.featureMap = map;
        end
        
        function y = eval(f,x)
            % y = eval(f,x)
            % Evaluates the function map at the points x. The function map must map to R^n
            % f: FeatureMap
            % x: N-by-d array
            % y: N-by-n array
            
            y = f.featureMap(x);
        end


    end
end