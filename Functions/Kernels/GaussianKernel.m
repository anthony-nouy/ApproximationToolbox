% Class GaussianKernel

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

classdef GaussianKernel < Kernel
    
    properties
        h
    end
    
    methods
        function k = GaussianKernel(h)
            % k = GaussianKernel(h)
            % Constructor for the GaussianKernel class, that defines a gaussian kernel parametrized by the real scalar h
            % h: 1-by-1 array
            % k: FeatureMapKernel
            
            k.h = h;
        end
        
        function kxy = eval(k,x,y)
            % kxy = eval(k,x,y)
            % Evaluates the kernel at the points x and y
            % k: GaussianKernel
            % x: n-by-d array
            % y: m-by-d array
            % kxy: n-by-m array
            
            if size(x,2) ~= size(y,2)
                error('x and y must be of same dimension.')
            end
            
            y = permute(y,[3,1,2]);
            xmy = bsxfun(@(x,y)x-y,x,y);
            kxy = exp(-sum(xmy.^2,3)/k.h^2);
        end
    end
end