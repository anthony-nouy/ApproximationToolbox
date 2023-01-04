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


    properties (Hidden)
        recurr
        norms
    end

    methods
        function p = DiscretePolynomials(x,n)
            % p = DiscretePolynomials(x,n)
            % Polynomials orthonormal with respect to a discrete measure
            %
            % x: DiscreteRandomVariable or DiscreteMeasure
            % p: DiscretePolynomials
            % n: maximum degree of polynomials 
            %    (by default, the number of values of x minus 1)

            if nargin==0 || (~isa(x,'DiscreteRandomVariable') && ~isa(x,'DiscreteMeasure'))
                error('Must specify a DiscreteRandomVariable or DiscreteMeasure')
            end
            if nargin==1
                n=length(x.values)-1;
            end

            p.measure = x;
            [p.recurr,p.norms] = DiscretePolynomials.precomputeRecurrenceMonic(x,n);
        end

        function [recurr,norms] = recurrenceMonic(p,n)
            % function [recurr,normsMonic] = recurrenceMonic(p,n)
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

            recurr = p.recurr(:,1:n+1);
            norms = p.norms(:,1:n+1);

        end



    end

    methods (Static , Access = private)


        function [recurr,norms] = precomputeRecurrenceMonic(r,n)
            % function [recurr,norms] = precomputeRecurrenceMonic(r,n)
            % Computes the coefficients of the three-term recurrence used to construct the monic polynomials
            % p_{n+1}(x) = (x-a_n)p_n(x) - b_n p_{n-1}(x), a_n and b_n are
            % the three-term recurrence coefficients
            % p : DiscretePolynomials
            % n: integer
            % recurr: 2-by-(n+1) double
            % norms: 1-by-(n+1) double (norms of monic polynomials)

            max = numel(r.values)-1;

            if n>max
                error('the requested degree exceeds the maximal degree')
            end

            norm2 = zeros(1,n+1);
            a = zeros(1,n+1);
            b = zeros(1,n+1);

            i = 1;
            cond = 1;

            G = integrationRule(r);
            x = G.points;
            weights = G.weights(:);

            norm2(1) = sum(weights);
            a(1) = dot(weights,x)/norm2(1);
            b(1) = 0;
            pn = ones(length(x),1);
            pnp1 = (x-a(1)).*pn;
            
            while cond && i <= n
                i = i+1;
                pnm1 = pn;
                pn = pnp1;

                norm2(i) = dot(weights,pn.^2);
                a(i) = dot(weights,pn.^2.*x)/norm2(i);
                b(i) = norm2(i)/norm2(i-1);

                pnp1 = (x-a(i)).*pn - b(i)*pnm1;
                d1 = abs(dot(pnp1.*pn, weights));
                d2 = abs(dot(pnp1.*pnm1, weights));
                tol = 1e-5;

                if d1 < tol && d2 < tol
                    cond = 1;
                else
                    cond = 0;
                end
            end
            if cond==0 && i~=n+1
                warning('problem of conditioning - requested number of recurrence terms not reached')
            end

            recurr = [a(1:i) ; b(1:i)]; % Recurrence coefficients
            norms = sqrt(norm2(1:i)); % Polynomial norms


        end


    end

end