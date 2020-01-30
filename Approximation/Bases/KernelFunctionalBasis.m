% Class KernelFunctionalBasis

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

classdef KernelFunctionalBasis < FunctionalBasis
    
    properties
        kernel
        points
    end
    
    methods
        function h = KernelFunctionalBasis(kernel,points)
            % h = KernelFunctionalBasis(kernel,points)
            % Constructor for the KernelFunctionalBasis class, which defines a functional basis based on a kernel and a set of points
            % kernel: Kernel
            % points: n-by-d array
            % h: KernelFunctionalBasis
            
            h.kernel = kernel;
            h.points = points;
        end
        
        function hx = eval(h,x)
            % hx = eval(h,x)
            % Evaluates the functional basis at the points x
            % h: KernelFunctionalBasis
            % x: N-by-d array
            % hx: N-by-n array
            
            hx = h.kernel(x,h.points);
        end
        
        function gx = evalDerivative(h,n,x)
            % gx = evalDerivative(h,x)
            % Evaluates the n-derivative of h at the points x
            % k: KernelFunctionalBasis
            % n: 1-by-d array of integers
            % x: N-by-d array
            % hx: N-by-n array
            
            gx = evalDerivative(h.kernel,n,x,h.points);
        end
        
        function n = cardinal(h)
            n = size(h.points,1);
        end
        
        function d = ndims(h)
            d = size(h.points,2);
        end
    end
end