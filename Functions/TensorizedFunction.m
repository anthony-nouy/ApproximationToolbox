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
            % resolution d>0, dimension dim). 
            % 
            % For a univariate function g(x) (dim=1),
            % g(x) = f(i_1,...,i_d,y) with y in [0,1] and i_k in {0,...,b-1}
            % where x = (i + y)b^(-d) with i in {0,...,b^d-1}
            % having the following representation in base b:
            % i = \sum_{k=1}^d i_k b^(d-k) in [0,b^d-1]
            %
            % For a d-variate function g(x1,...,xd) (dim=d)
            % - if t.orderingType=1 then
            % g(x) = f(i_1,...,i_d,j_1,....,j_d,y1,...yd) with yk in [0,1]
            % and x1 = (i + y1)b^(-d), x2 = (j + y2)b^(-d), ...
            % - if t.orderingType=2 then
            % g(x) = f(i_1,j_1,...,i_d,j_d,y1,...,yd)
            % - if t.orderingType=3 then
            % g(x) = f(i_1,j_1,y1,...,i_d,j_d,yd)            %
            % f : Function (function of (d+1)*dim variables)
            % t : Tensorizer
            
            if t.d==0
                error('resolution d should not be 0')
            end
            g.f = f;
            if nargin==2 && isa(t,'Tensorizer')
                g.t = t;
            else
                error('Must provide a Tensorizer.')
            end
            g.dim = t.dim;
            
            if ~isempty(t.X)
                g.measure = t.X;
            else
                g.measure = ProductMeasure.tensorize(LebesgueMeasure(0,1),t.dim);
            end

        end
        
        function gx = eval(g,z)
            % evaluate tensorized function g(x1,...,xdim), which is identified 
            % with a function f(z) of 
            % (d+1)*dim variables, where d>0 is the resolution of the
            % tensorizer g.t
            % 
            % function gx = eval(g,x)
            % g : TensorizedFunction
            % x : n-by-dim array with k = dim or   k = (d+1)*dim 
            % gx : n-by-sz array, where sz the the size of the output of f
            % 
            % if k=dim, returns returns g.f(z) with z=g.t.map(x)
            %
            % if k=(d+1)*dim, returns g.f(x)

            if size(z,2)==g.t.dim
                z = g.t.map(z);
            end
            gx = g.f(z);
        end
        
        
        function s = domain(f)
            if ~isempty(f.t.X)
                s = support(f.t.X);
            else
                s = cell(1,f.t.dim);
                s(:) = {[0,1]};
            end
        end
    end
end