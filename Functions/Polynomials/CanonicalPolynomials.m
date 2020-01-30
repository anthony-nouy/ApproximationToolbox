% Class CanonicalPolynomials

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

classdef CanonicalPolynomials < UnivariatePolynomials
    
    methods
        function p = CanonicalPolynomials(measure)
            if nargin==1
                if ~isa(measure,'Measure')
                    error('must provide a Measure')
                end
                p.measure = measure;
            end
        end
        
        function ok = isOrthonormal(p)
            % ok = isOrthonormal(p)
            % Checks the orthonormality of the basis created by the
            % functions of p
            % p: UnivariatePolynomials
            % ok: boolean
            
            ok = false;
        end
        
        function c = polyCoeff(p,list)
            % c = polyCoeff(p,list)
            % Computes the coefficients of the monomials used to create the
            % polynomials of degree specified in list
            % p: CanonicalPolynomials
            % list: 1-by-n or n-by-1 array of integer
            % c: n-by-max(list)+1 double
            
            i = max(list);
            c = eye(i+1,i+1);
            c = c(list+1,:);
        end
        
        function px = dPolyval(p,list,x)
            % px = dPolyval(P,list,x)
            % Computes the first order derivative of polynomials of p of degrees in list at points x
            % p: CanonicalPolynomials
            % list: d-by-1 or 1-by-d double
            % x: n-by-1 or 1-by-n double
            % px: n-by-d double
            
            px = (x.^(list-1)) .* repmat(list,length(x),1);
            px(isnan(px)) = 0; % Prevents NaN when x = 0
        end
        
        function [c,I] = one(~)
            c = 1;
            I = 0;
        end
        
        function s = domain(~)
            % s = domain()
            % Returns the domain of the canonical polynomials
            % s: 1-by-2 double
            s= [-Inf,Inf];
        end
        
        function ok = eq(p1,p2)
            ok = isa(p2,'CanonicalPolynomials');
        end
    end
end