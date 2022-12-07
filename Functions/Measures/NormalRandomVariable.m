% Class NormalRandomVariable

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

classdef NormalRandomVariable < RandomVariable
    
    properties
        mu
        sigma
    end
    
    methods
        function  X = NormalRandomVariable(mu,sigma)
            %  X = NormalRandomVariable(mu,sigma)
            % Normal random variable with mean mu and standard deviation sigma (by default mu=0, sigma=1)
            % mu: 1-by-1 double (optional)
            % sigma: 1-by-1 double (optional)
            % X: NormalRandomVariable with parameters mu and sigma
            
            X@RandomVariable('normal');
            if nargin==0
                X.mu = 0;
                X.sigma = 1;
            else
                X.mu = mu;
                X.sigma = sigma;
            end
        end
        
        function X = shift(X,b,s)
            X.mu = X.mu+b;
            X.sigma = X.sigma*s;
        end
        
        function Xstd = getStandardRandomVariable(~)
            % Xstd = getStandardRandomVariable(X)
            % Returns the standard normal random variable with mean 0 and standard deviation 1
            % X: NormalRandomVariable
            % Xstd: NormalRandomVariable
            
            Xstd = NormalRandomVariable();
        end
        
        function s = support(~)
            % s = support(X)
            % Returns the support of the normal random variable X
            % X: NormalRandomVariable
            % s: 1-by-2 double
            
            s = [-Inf,Inf];
        end
        
        function p = orthonormalPolynomials(X,varargin)
            % p = orthonormalPolynomials(X)
            % Returns the orthonormal polynomials according to the NormalRandomVariable X
            % X: NormalRandomVariable
            % p: HermitePolynomials or ShiftedOrthonormalPolynomials
            
            p = HermitePolynomials(varargin{:});
            if X~=NormalRandomVariable(0,1)
                %warning('ShiftedOrthonormalPolynomials are created')
                p = ShiftedOrthonormalPolynomials(p,X.mu,X.sigma);
            end
            
        end
        
        function p = getParameters(X)
            % p = getParameters(X)
            % Returns the parameters of the normal random variable X in an array
            % X: NormalRandomVariable
            % p: 1-by-2 cell
            
            p = {X.mu,X.sigma};
        end
        
        function [m,v] = randomVariableStatistics(X)
            % [m,v] = randomVariableStatistics(X)
            % Computes the mean m and the variance v of the normal random variable X
            % X: NormalRandomVariable
            % m: 1-by-1 double
            % v: 1-by-1 double
            
            m = X.mu;
            v = (X.sigma).^2;
        end
    end
end