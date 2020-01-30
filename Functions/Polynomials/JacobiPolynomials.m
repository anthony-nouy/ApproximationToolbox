% Class JacobiPolynomials

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

classdef JacobiPolynomials < OrthonormalPolynomials
    
    properties
        alpha
        beta
    end
    
    methods
        function p = JacobiPolynomials(alpha,beta,n)
            p.measure = BetaRandomVariable(alpha,beta);
            p.alpha = alpha;
            p.beta = beta;
            
            if ~exist('n','var')
                n = 50;
            end
            
            [p.recurrenceCoefficients, p.orthogonalPolynomialsNorms] = p.recurrence(alpha,beta,n);
        end
    end
    
    methods (Static, Hidden)
        function [recurr, norms] = recurrence(alpha,beta,n)
            n = 0:n;
            a = (beta^2 - alpha^2) ./ ((alpha + beta + 2*n).*(alpha + beta + 2*n + 2));
            b = (4*n.*(n + alpha).*(n + beta).*(n + alpha + beta)) ./ ...
                ((2*n + alpha + beta).^2.*(2*n + alpha + beta + 1).*(2*n + alpha + beta - 1));
            recurr = sparse([a;b]);
            norms = (2.^(alpha+beta+2*n+1).*gamma(alpha+n+1).*gamma(beta+n+1).*factorial(n)) ./ ...
                (gamma(alpha+beta+2*n+1).*gamma(alpha+beta+2*n+2)./gamma(alpha+beta+n+1));
            norms = sqrt(norms);
        end
    end
end