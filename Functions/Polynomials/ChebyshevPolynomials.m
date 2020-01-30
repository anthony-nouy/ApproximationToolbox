% Class ChebyshevPolynomials

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

classdef ChebyshevPolynomials < OrthonormalPolynomials
    
    properties
        % KIND - Kind of Chebyshev polynomial
        kind
    end
    
    methods
        function p = ChebyshevPolynomials(kind,n)
            if nargin == 0 || isempty(kind)
                p.kind = 1;
            end
            if p.kind == 1
                p.measure = BetaRandomVariable(0.5,0.5);
            end
            if ~exist('n','var')
                n = 50;
            end
            
            [p.recurrenceCoefficients, p.orthogonalPolynomialsNorms] = p.recurrence(p.kind,n);
        end
    end
    
    methods (Static, Hidden)
        function [recurr, norms] = recurrence(kind,n)
            if kind == 1
                a = zeros(1,n);
                b = [0 , 0.5 , 0.25*ones(1,n-2)];
                recurr = sparse([a;b]);
                norms = [sqrt(pi) , sqrt(pi)./(2.^((1:n)-0.5))];
            else
                error('Only the Chebyshev polynomials of first kind are implemented.')
            end
        end
    end
end