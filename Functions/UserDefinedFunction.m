% Class UserDefinedFunction

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

classdef UserDefinedFunction < Function
    
    properties
        fun
    end
    
    methods
        function f = UserDefinedFunction(fun,dim,sz)
            % f = UserDefinedFunction(fun,dim,sz)
            % fun: function of dim variables
            % dim: number of input variables
            % sz: size of the output of f ([1,1] by default)
            %
            % fun(x) with x of size 1-by-dim returns an array of size sz
            % property evaluationAtMultiplePoints is set to false by default,
            % if set to true, fun(x) with x of size N-by-dim returns an array
            % of size N-by-sz
            
            f.evaluationAtMultiplePoints = false;
            f.dim = dim;
            if nargin==3
                f.outputSize = sz;
            end
            
            if ischar(fun)
                for i=dim:-1:1
                    s = ['x' , num2str(i)];
                    snew = ['x(:,' num2str(i) ')'];
                    fun = strrep(fun,s,snew);
                end
                f.fun = str2func(['@(x)' fun]);
            else
                f.fun = fun;
            end
        end
        
        function f = vectorize(f)
            if isa(f.fun,'function_handle')
                f.fun = str2func(vectorize(f.fun));
                f.evaluationAtMultiplePoints = true;
            else
                warning('Cannot vectorize non function_handle objects.')
            end
        end
        
        function y = eval(h,x)
            n = size(x,1);
            f = h.fun;
            if ismethod(f,'eval')
                f = @(x) eval(f,x);
            end
            if h.evaluationAtMultiplePoints
                y = f(x);
            else
                y = zeros(n,prod(h.outputSize));
                parfor i=1:n
                    fxi = f(x(i,:));
                    y(i,:) = fxi(:);
                end
            end
            y = reshape(y,[n,h.outputSize]);
        end
    end
end