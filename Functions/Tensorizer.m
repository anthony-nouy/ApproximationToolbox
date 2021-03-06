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
        Y
        orderingType = 1;
    end
    
    
    methods
        
        function t = Tensorizer(b,d,dim,X,Y)
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
            % - {{0,...,b-1}^d}^dim x [0,1]^dim
            % if property orderingType = 1
            % - {{0,...,b-1}^dim}^d x [0,1]^dim
            % if property orderingType = 2
            % - or {{0,...,b-1}^d x [0,1]}^dim 
            % if property orderingType = 3   
            %
            % t = Tensorizer(b,d,dim,X,Y)
            % Provides a measure X for the input variable x and a measure Y for 
            % the local variable y
            % X : RandomVariable if dim=1 or RandomVector (by default a Lebesgue measure over [0,1]^dim)
            % Y : RandomVariable if dim=1 or RandomVector (by default a Lebesgue measure over [0,1]^dim)
            
            
            
            
            if nargin<3
                dim=1;
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
            if nargin==5
                if isa(Y,'RandomVariable')
                    Y = RandomVector(Y,t.dim);
                end
                t.Y = Y;
            else
                t.Y = RandomVector(UniformRandomVariable(0,1),t.dim);
            end
        end
        
        function [y,i] = map(t,x)
            y = cell(1,t.dim);
            i = cell(1,t.dim);
            for k=1:t.dim
                if ~isempty(t.X)
                    uk = t.X.randomVariables{k}.cdf(x(:,k));
                else
                    uk = x(:,k);
                end
                y{k} = Tensorizer.u2z(uk,t.b,t.d);
                if ~isempty(t.Y)
                    y{k}(:,end) = t.Y.randomVariables{k}.icdf(y{k}(:,end));
                end
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
                if ~isempty(t.Y)
                    zk = t.Y.randomVariables{k}.cdf(zk);
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
                if isempty(t.Y)
                    t.Y = UniformRandomVariable(0,1);
                    h = PolynomialFunctionalBasis(orthonormalPolynomials(t.Y),0:h);    
                else
                    h = cellfun(@(mu) PolynomialFunctionalBasis(orthonormalPolynomials(mu),0:h),t.Y.randomVariables,'uniformoutput',false);
                    h = FunctionalBases(h);
                end
            end
                        
            if isa(h,'function_handle')
                h = UserDefinedFunctionalBasis({h});
                if ~isempty(t.Y)
                    h.measure = t.Y.randomVariables{1};
                else
                    h.measure = UniformRandomVariable(0,1);
                end
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