% Class BetaRandomVariable

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

classdef BetaRandomVariable < RandomVariable
    
    properties
        % ALPHA - Shape parameter
        alpha
        % BETA - Shape parameter
        beta
    end
    
    methods
        function  X = BetaRandomVariable(alpha,beta)
            % BETARANDOMVARIABLE - Constructor of the class BetaRandomVariable.
            %
            % Beta random variable with parameters alpha and beta
            %
            % alpha: 1-by-1 double greater than 0 (optional)
            % beta: 1-by-1 double greater than 0 (optional)
            % X: BetaRandomVariable with parameters alpha and beta
            
            X@RandomVariable('beta');
            
            if nargin==0
                X.alpha = 0.5;
                X.beta = 0.5;
            elseif nargin==1
                X.alpha = alpha(1);
                X.beta = alpha(2);
            else
                X.alpha = alpha;
                X.beta = beta;
            end
        end
        
        function s = support(X)
            % SUPPORT - Support of the beta random variable
            %
            % s = SUPPORT(X)
            % X: BetaRandomVariable
            % s: 1-by-2 double
            
            s = [0 1];
        end
        
        function p = orthonormalPolynomials(X,varargin)
            % ORTHONORMALPOLYNOMIALS - orthonormal polynomials according to the measure of the BetaRandomVariable
            %
            % p = ORTHONORMALPOLYNOMIALS(X)
            % X: BetaRandomVariable
            % p: ShiftedOrthonormalPolynomials (shifted Jacobi polynomials)
            
            p = ShiftedOrthonormalPolynomials(JacobiPolynomials(X.alpha-1,X.beta-1),1/2,1/2);
        end
        
        function p = getParameters(X)
            % GETPARAMETERS - Parameters of the beta random variable
            %
            % p = GETPARAMETERS(X)
            % X: BetaRandomVariable
            % p: 1-by-2 cell
            
            p = {X.alpha,X.beta};
        end
        
        function [m,v] = randomVariableStatistics(X)
            % RANDOMVARIABLESTATISTICS - Mean m and the variance v of the beta random variable
            %
            % [m,v] = RANDOMVARIABLESTATISTICS(X)
            % X: BetaRandomVariable
            % m: 1-by-1 double
            % v: 1-by-1 double
            
            [m,v] = betastat(X.alpha,X.blpha);
        end
    end
end