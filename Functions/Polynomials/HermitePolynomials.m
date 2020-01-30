% Class HermitePolynomials

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

classdef HermitePolynomials < OrthonormalPolynomials
    
    methods
        function  p = HermitePolynomials(n)
            %  p = HermitePolynomials(varargin)
            % Polynomials defined on R and orthonormal with respect to the standard gaussian measure 1/sqrt(2*pi)*exp(-x^2/2)
            % n: integer, indicates the highest degree for which a polynomial can be computed with the stored recurrence coefficients (optional, default value: 50)
            % p: HermitePolynomials
            
            p.measure= NormalRandomVariable(0,1);
            if ~exist('n','var')
                n = 50; % Default maximum degree for the polynomials
            end
            
            [p.recurrenceCoefficients, p.orthogonalPolynomialsNorms] = p.recurrence(n);
        end
    end
    
    methods (Static, Hidden)
        function [recurr, norms] = recurrence(n)
            % [recurr, norms] = recurrence(n)
            % Computes the coefficients of the three-term recurrence used to construct the Hermite polynomials, and the norms of the polynomials
            % The three-term recurrence writes:
            % p_{n+1}(x) = (x-a_n)p_n(x) - b_n p_{n-1}(x), a_n and b_n are
            % the three-term recurrence coefficients
            % n: integer
            % recurr: 2-by-n double
            % norms: 1-by-n double
            
            a = zeros(1,n+1);
            b = 0:n;
            recurr = sparse([a;b]);
            norms = sqrt(factorial(0:n));
        end
    end
end