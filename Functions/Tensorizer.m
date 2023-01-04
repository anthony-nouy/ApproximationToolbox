% Class Tensorizer

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

classdef Tensorizer
    
    properties
        b
        d
        dim
        X
        orderingType = 1;
    end
    
    
    methods
        
        function t = Tensorizer(b,d,dim,X)
            % t = Tensorizer(b,d)
            % Tensorizer : defines a map t from [0,1] to
            % {0,...,b-1}^d x [0,1]^dim
            % t(x) = (i_1,...,i_d,y) with y in [0,1] and i_k in {0,...,b-1}
            % such that x = (i + y)b^(-d) with i in {0,...,b^d-1}
            % having the following representation in base b:
            % i = \sum_{k=1}^d i_k b^(d-k) in [0,b^d-1]
            %
            % t = Tensorizer(b,d,dim)
            % defines a map t from [0,1]^dim to 
            % - {{0,...,b-1}^d}^dim x [0,1]^dim if property orderingType = 1
            % - {{0,...,b-1}^dim}^d x [0,1]^dim if property orderingType = 2
            % - or {{0,...,b-1}^d x [0,1]}^dim  if property orderingType = 3   
            % 
            % t = Tensorizer(b,d,dim,X)
            % Provides a measure X for the input variable x 
            % X : RandomVariable if dim=1 or RandomVector (by default a Uniform measure over [0,1]^dim)
            
            
     
            
            if nargin<3
                dim=1;
            end
            if d==0
                warning('Tensorizer should not be used with d=0')
            end
            t.dim = dim;
            t.b=b;
            t.d=d;
            if nargin>=4
                if isa(X,'RandomVariable')
                    X = RandomVector(X,t.dim);
                end
                t.X = X;
            else
                t.X = RandomVector(UniformRandomVariable(0,1),t.dim);
            end
        end
        
        function [y,i] = map(t,x)
            % function [y,i] = map(t,x)
            % t: Tensorizer
            % x : n-by-dim array
            % y : n-by-dim array
            % i : n-by-(dxdim) array
            y = cell(1,t.dim);
            i = cell(1,t.dim);
            for k=1:t.dim
                if ~isempty(t.X)
                    uk = t.X.randomVariables{k}.cdf(x(:,k));
                else
                    uk = x(:,k);
                end
                y{k} = Tensorizer.u2z(uk,t.b,t.d);
                i{k} = y{k}(:,1:end-1);
                y{k} = y{k}(:,end);
            end
            y = [y{:}];
            i = [i{:}];
            switch t.orderingType
                case 1
                    if nargout~=2
                        y = [i,y];
                    end
                case 2
                    j = cell(1,t.d);
                    for k=1:t.d
                        j{k} = i(:,k:t.d:end);
                    end
                    i = [j{:}];
                    if nargout~=2
                        y = [i,y];
                    end
                case 3
                    if nargout~=2
                        j = cell(1,t.dim);
                        for k=1:t.dim
                            j{k} = [i(:,(k-1)*t.d + (1:t.d)) , y(:,k)];                            
                        end
                        y = [j{:}];
                    end
            end
            
            
        end

        function x = inverseMap(t,z)
            % function x = inverseMap(t,z)
            % t: Tensorizer
            % z : n-by-((d+1)xdim) array
            % x : n-by-dim array

            u = cell(1,t.dim);
            for k=1:t.dim
                switch t.orderingType
                    case 1
                        ik = z(:,(1:t.d)+(k-1)*t.d);
                        zk = z(:,end-t.dim+k);
                    case 2
                        ik = z(:,k:t.dim:t.dim*t.d);
                        zk = z(:,end-t.dim+k);
                    case 3
                        ik = z(:,(1:t.d)+(k-1)*(t.d+1));
                        zk = z(:,k*(t.d+1));
                end
                u{k} = Tensorizer.z2u([ik,zk],t.b);
                if ~isempty(t.X)
                    u{k} = t.X.randomVariables{k}.icdf(u{k});
                end
            end
            x = [u{:}];
        end
        
        function g = tensorize(t,fun)
            % g = tensorize(t,fun)
            % creates a TensorizedFunction g
            % from a function fun defined on support(t.X)
            %
            % t: Tensorizer
            % fun: Function or function_handle
            % g: TensorizedFunction
            
            if ~isa(fun,'Function') && ~isa(fun,'function_handle')
                error('argument must be a Function or function_handle')
            end
            
            if isa(fun,'function_handle')
                fun = UserDefinedFunction(fun,t.dim);
            end
            
            fun = UserDefinedFunction(@(z) fun(t.inverseMap(z)),(t.d+1)*t.dim);
            g = TensorizedFunction(fun,t);
        end
        
        function g = tensorizedFunctionFunctionalBases(t,h)
            % g = tensorizedFunctionFunctionalBases(t,h)
            %
            % t: Tensorizer
            % h: FunctionalBasis or function_handle or integer (for polynomial basis function of degree m) or FunctionalBases
            % h=0 by default
            
            if nargin==1
                h=0;
            end
            
            %if isa(h,'double') && h==0
            %    h = @(y) h*ones(size(y));
            %end
            
            if isa(h,'double')
                h = PolynomialFunctionalBasis(orthonormalPolynomials(LebesgueMeasure(0,1)),0:h);             
            end
                        
            if isa(h,'function_handle')
                h = UserDefinedFunctionalBasis({h});
                h.measure = LebesgueMeasure(0,1);
            end
            
            if isa(h,'FunctionalBasis')
                h = FunctionalBases.duplicate(h,t.dim);
            end
            
            if ~isa(h,'FunctionalBases')
                error('wrong type of argument for h')
            end
            
            bases = cell(1,(t.d+1)*t.dim);
            bases(end-t.dim+1:end)=h.bases;
            
            p = DiscretePolynomials(DiscreteRandomVariable((0:t.b-1).'));
            p = PolynomialFunctionalBasis(p,0:t.b-1);
            
            bases(1:end-t.dim)={p};
            g = FunctionalBases(bases);
        end
    end
    
    methods (Static)
        function [y,i] = u2z(u,b,d)
            u = u(:);
            su = u*(b^d);
            i = floor(su);
            i = min(i,b^d-1);
            y = su - i ;
            i = integer2baseb(i,b,d);
            if nargout==1
                y = [i,y];
            end
        end
        
        function u = z2u(z,b)
            d = size(z,2)-1;
            y = z(:,end);
            i = z(:,1:end-1);
            if size(i,2)>1
                i = baseb2integer(i,b);
            end
            u = (y+i)*b^(-d);
        end
    end
end