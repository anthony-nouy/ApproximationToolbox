% Class UniformRandomVariable

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

classdef UniformRandomVariable < RandomVariable
    
    properties
        a
        b
    end
    
    methods
        function  X = UniformRandomVariable(a,b)
            % X = UniformRandomVariable(a,b)
            % Uniform random variable on [a,b] (by default [-1,1])
            % a: 1-by-1 double (optional)
            % b: 1-by-1 double (optional)
            % X: UniformRandomVariable with parameters a and b
            %
            % X = UniformRandomVariable([a,b])
            
            X@RandomVariable('unif');
            
            if nargin==0
                X.a = -1;
                X.b = 1;
            elseif nargin==1
                X.a = a(1);
                X.b = a(2);
            else
                X.a = a;
                X.b = b;
            end
        end
        
        function X = shift(X,b,s)
            % Y = shift(X,b,s)
            % returns the uniform random variable Y = sX + b
            % X,Y: UniformRandomVariable
            
            X.a = s*X.a+b;
            X.b = s*X.b+b;
        end
        
        function Xstd = getStandardRandomVariable(X)
            % Xstd = getStandardRandomVariable(X)
            % Returns the standard uniform random variable on [-1,1]
            % X: UniformRandomVariable
            % Xstd: UniformRandomVariable
            
            Xstd = UniformRandomVariable();
        end
        
        function s = support(X)
            % s = support(X)
            % Returns the support of the uniform random variable X
            % X: UniformRandomVariable
            % s: 1-by-2 double
            
            s = [X.a,X.b];
        end
        
        function p = orthonormalPolynomials(X,varargin)
            % p = orthonormalPolynomials(X,n)
            % Returns the n first orthonormal polynomials according to
            % the UniformRandomVariable X
            % X: UniformRandomVariable
            % n: integer (optional)
            % p: LegendrePolynomials
            
            p = LegendrePolynomials(varargin{:});
            if X~=UniformRandomVariable(-1,1)
%                 warning('ShiftedOrthonormalPolynomials are created')
                p = ShiftedOrthonormalPolynomials(p,(X.a+X.b)/2,(X.b-X.a)/2);
            end
        end
        
        function p = getParameters(X)
            % p = getParameters(X)
            % Returns the parameters of the uniform random variable X in an
            % array
            % X: UniformRandomVariable
            % p: 1-by-2 cell
            
            p = {X.a,X.b};
        end
        
        function [m,v] = randomVariableStatistics(X)
            % [m,v] = randomVariableStatistics(X)
            % Computes the mean m and the variance v of the uniform random
            % variable X
            % X: UniformRandomVariable
            % m: 1-by-1 double
            % v: 1-by-1 double
            
            [m,v] = unifstat(X.a,X.b);
        end
    end
end