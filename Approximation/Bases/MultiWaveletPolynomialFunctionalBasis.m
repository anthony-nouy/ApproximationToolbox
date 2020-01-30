% Class MultiWaveletPolynomialFunctionalBasis

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

classdef MultiWaveletPolynomialFunctionalBasis < MultiWaveletFunctionalBasis
    
    properties
        p % Polynomial degree
    end
    
    methods
        function h = MultiWaveletPolynomialFunctionalBasis(mu,p,s,a)
            % MULTIWAVELETPOLYNOMIALFUNCTIONALBASIS - Constructor for the class MultiWaveletPolynomialFunctionalBasis
            %
            % h = MultiWaveletPolynomialFunctionalBasis(mu,p,s,a)
            % mu: LebesgueMeasure or UniformRandomVariable
            % p: integer (degree)
            % s: integer (resolution)
            % a: integer (scaling factor, default =2)
            % h: MultiWaveletPolynomialFunctionalBasis
            
            if nargin<4
                a=2;
            end
            if ~isa(mu,'LebesgueMeasure') && ~isa(mu,'UniformRandomVariable')
                error('Must provide a LebesgueMeasure or UniformRandomVariable.')
            end
            
            supp = support(mu);
            F = PolynomialFunctionalBasis(orthonormalPolynomials(mu),0:p);
            points = linspace(supp(1),supp(2),a+1);
            V = PiecewisePolynomialFunctionalBasis(points,p);
            F = V.interpolate(@(x) F.eval(x));
            F = F.data;
            M = null(F');
            h.F = SubFunctionalBasis(V,F);
            h.M = SubFunctionalBasis(V,M);
            h.measure = mu;
            h.s = s;
            h.a = a;
            h.p = p;
            
            if norm(eye(cardinal(h)) - gramMatrix(h)) < eps
                h.isOrthonormal = true;
            end
        end
        
        function m = mean(h)
            m = zeros(cardinal(h),1);
            m(1) = sqrt(h.measure.mass);
        end
        
        function G = gramMatrix(h)
            % GRAMMATRIX - Computes the Gram matrix of the basis h
            %
            % G = GRAMMATRIX(h)
            % h: MultiWaveletPolynomialFunctionalBasis
            % G: cardinal(h)-by-cardinal(h) double
            
            G = diag([ones(1,h.p+1), h.measure.mass*ones(1,cardinal(h)-h.p-1)]);
        end
    end
end