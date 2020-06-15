% Class TensorizedFunction

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

classdef TensorizedFunction < Function
    
    properties
        f % Function of d+1 variables
        t % Tensorizer
    end
    
    
    methods
        
        function g = TensorizedFunction(f,t)
            % g = TensorizedFunction(f,t)
            %
            % function g(x1,...,xdim) identified with a function f(z) of 
            % (d+1)*dim variables using the Tensorizer t (base b, 
            % resolution d, dimension dim)
            %
            % For a univariate function g(x) (dim=1),
            % g(x) = f(i_1,...,i_d,y) with y in [0,1] and i_k in {0,...,b-1}
            % where x = (i + y)b^(-d) with i in {0,...,b^d-1}
            % having the following representation in base b:
            % i = \sum_{k=1}^d i_k b^(k-1) in [0,b^d-1]
            %
            % For a bivariate function g(x1,x2) (dim=2)
            % - if t.orderingType=1 then
            % g(x) = f(i_1,...,i_d,j_1,....,j_d,y1,y2) with yk in [0,1]
            % and x1 = (i + y1)b^(-d), x2 = (j + y2)b^(-d)
            % - if t.orderingType=2 then
            % g(x) = f(i_1,j_1,...,i_d,j_d,y1,y2)
            %
            % f : Function (function of (d+1)*dim variables)
            % t : Tensorizer
            
            g.f = f;
            if nargin==2 && isa(t,'Tensorizer')
                g.t = t;
            else
                error('Must provide a Tensorizer.')
            end
            g.dim = t.dim;
        end
        
        function fx = eval(f,z)
            % fx = eval(f,z)
            if size(z,2)==f.t.dim
                z = f.t.map(z);
            end
            fx = f.f(z);
        end
        
        function s = domain(f)
            s = support(f.t.X);
        end
    end
end