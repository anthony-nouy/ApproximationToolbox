% Class DiscretePolynomials

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

classdef DiscretePolynomials < OrthonormalPolynomials
    
    methods
        function p = DiscretePolynomials(x,varargin)
            % p = DiscretePolynomials(x,varargin)
            % Polynomials orthonormal with respect to a discrete measure
            %
            % x: DiscreteRandomVariable
            % p: DiscretePolynomials
            
            if nargin==0 || ~isa(x,'DiscreteRandomVariable')
                error('Must specify a DiscreteRandomVariable')
            end
            
            p.measure = x;
                        
        end


        function [recurr,norms] = recurrenceMonic(p,n)
            % function [recurr,norms] = recurrenceMonic(p,n)
            % Computes the coefficients of the three-term recurrence used to construct the monic polynomials
            % p_{n+1}(x) = (x-a_n)p_n(x) - b_n p_{n-1}(x), a_n and b_n are
            % the three-term recurrence coefficients
            % p : DiscretePolynomials
            % n: integer
            % recurr: 2-by-(n+1) double
            % norms: 1-by-(n+1) double (norms of monic polynomials)

            r = p.measure;

            max = numel(r.values)-1;

            if n>max
                error('the requested degree exceeds the maximal degree')
            end

            norm = zeros(1,n+1);
            a = zeros(1,n+1);
            b = zeros(1,n+1);
            
            i = 1;
            cond = 1;
            
            norm(1) = 1;
            a(1) = dotProduct(@(x) x.^0, @(x) x,r);
            b(1) = 0;
            pn = @(x) 1;
            pnp1 = @(x) (x-a(1)).*pn(x);
            
            while cond && i <= n+1
                i = i+1;
                pnm1 = pn;
                pn = pnp1;
                
                norm(i) = dotProduct(pn, pn,r);
                a(i) = dotProduct(pn, @(x) x.*pn(x),r)/norm(i);
                b(i) = norm(i)/norm(i-1);
                
                pnp1 = @(x) (x-a(i)).*pn(x) - b(i)*pnm1(x);
                cond = isOrth(pnp1,pn,pnm1,r); % Condition of orthogonality
            end
            if cond==0
                warning('problem of conditioning - requested number of recurrence terms not reached')
            end
            recurr = [a(1:i-1) ; b(1:i-1)]; % Recurrence coefficients
            norms = sqrt(norm(1:i-1)); % Polynomial norms

            function b = isOrth(pnp1,pn,pnm1,r)
                % function b = isOrth(pnp1,pn,pnm1,r)
                % Function that determines if the polynomial pnp1 is orthogonal to the polynomials pn and pnm1, according to the RandomVariable r
                % pnp1: anonymous function handle
                % pn: anonymous function handle
                % pnm1: anonymous function handle
                % r: RandomVariable
                % b: boolean
                
                tol = 1e-5; % Tolerance for the inner product to be considered as 0
                
                d1 = abs(dotProduct(pnp1, pn, r));
                d2 = abs(dotProduct(pnp1, pnm1, r));
                
                if d1 < tol && d2 < tol
                    b = 1;
                else
                    b = 0;
                end
            end
            
            function d = dotProduct(p1,p2,r)
                % d = dotProduct(p1,p2,r)
                % Function that computes the inner product between two polynomials p1 and p2, according to the RandomVariable r
                % p1: anonymous function handle
                % p2: anonymous function handle
                % r: RandomVariable
                % d: 1-by-1 double
                G = integrationRule(r);
                d = G.integrate(@(x) p1(x).*p2(x));
            end

        end


    end

end