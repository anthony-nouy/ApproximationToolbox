% Class CubicSplineKernel

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

classdef CubicSplineKernel < Kernel
    
    methods
        function k = CubicSplineKernel()
            % k = CubicSplineKernel()
            % Constructor for the CubicSplineKernel class, that defines a cubic spline kernel
            % k: CubicSplineKernel
            
        end
        
        function kxy = eval(k,x,y)
            % kxy = eval(k,x,y)
            % Evaluates the kernel at the points x and y
            % k: CubicSplineKernel
            % x: n-by-1 array
            % y: m-by-1 array
            % kxy: n-by-m array
            
            if size(x,2) ~= 1 && size(y,2) ~= 1
                error('x and y must be of dimension 1.')
            end
            
            k = @(x,y) min(x,y).^3 / 3 + min(x,y).^2 / 2 .* abs(x-y) + 1 + x.*y;
            kxy = bsxfun(k,x(:),y(:).');
        end
        
        function gxy = evalDerivative(k,n,x,y)
            % gxy = evalDerivative(k,n,x,y)
            % Evaluates the nth-order derivative of the kernel function  at the points x and y
            % Only implemented for n = 1
            % k: CubicSplineKernel
            % n: integer
            % x: n-by-1 array
            % y: m-by-1 array
            % kxy: n-by-m array
            
            if n ~= 1
                error('Not implemented for n > 1.')
            end
            if size(x,2) ~= 1 && size(y,2) ~= 1
                error('x and y must be of dimension 1.')
            end
            
            g = @(x,y) (x < y) .* min(x,y).^2 + (x < y) .* min(x,y) .* abs(x-y) + ...
                min(x,y).^2 / 2 .* (x-y) ./ abs(x-y) + y;
            
            gxy =  bsxfun(g,x(:),y(:).');
            gxy(ismember(x(:),y(:)),:) = [];
        end
    end
end