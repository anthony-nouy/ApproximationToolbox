% Class EmpiricalPolynomials

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

classdef EmpiricalPolynomials < OrthonormalPolynomials

    methods
        function p = EmpiricalPolynomials(x, varargin)
            % p = EmpiricalPolynomials(x)
            % Polynomials defined on R and orthonormal with respect to the gaussian kernel smoothed distribution based on a sample x, which was centered and normalized (unit variance)
            % x: 1-by-n or n-by-1 double or an EmpiricalRandomVariable
            % p: EmpiricalPolynomials

            if nargin==0
                error('Must specify a sample')
            end

            if isa(x,'EmpiricalRandomVariable')
                rv = getStandardRandomVariable(x);
            else
                % Standardization of the sample
                x = reshape(x,length(x),1);
                x = (x - repmat(mean(x,1),size(x,1),1))...
                    ./ repmat(std(x),size(x,1),1);
                rv = EmpiricalRandomVariable(x);
            end
            p.measure= rv;
        end

        function [recurr,norms] = recurrenceMonic(p,n)
            % function [recurr,norms] = recurrenceMonic(p,n)
            % Computes the coefficients of the three-term recurrence used to construct the monic polynomials
            % p_{n+1}(x) = (x-a_n)p_n(x) - b_n p_{n-1}(x), a_n and b_n are
            % the three-term recurrence coefficients
            % p : EmpiricalPolynomials
            % n: integer
            % recurr: 2-by-(n+1) double
            % norms: 1-by-(n+1) double (norms of monic polynomials)

            r = p.measure;


            norms = zeros(1,n+1);
            a = zeros(1,n+1);
            b = zeros(1,n+1);

            nint = ceil((2*n+3)/2);
            xi = r.sample;
            G = gaussIntegrationRule(NormalRandomVariable,nint);
            wts = G.weights / length(xi) ;

            xij = r.bandwidth*repmat(G.points(:).',length(xi),1) + ...
                repmat(xi(:),1,length(G.points));

            i = 1;
            cond = 1;

            norms(1) = 1;
            a(1) = sum(xij*wts(:));
            b(1) = 0;

            while cond && i <= n+1
                i = i+1;

                pnm1 = reshape(eval(i-2,a,b,xij),size(xij));
                pn = reshape(eval(i-1,a,b,xij),size(xij));

                norms(i) = sum((pn.*pn)*wts(:));
                a(i) = sum((pn.*(xij.*pn))*wts(:)) / norms(i);
                b(i) = norms(i)/norms(i-1);

                pnp1 = (xij-a(i)).*pn - b(i)*pnm1;

                if nargin == 2 % Imposed max, no check on orthogonality
                    cond = 1;
                else
                    cond = isOrth(pnp1,pn,pnm1,wts); % Condition of orthogonality
                end
            end

            if cond == 0 && i-2 ~= n
                error(['Maximum degree: ', num2str(i-2) ' (', num2str(n), ' asked)']);
            end

            recurr = sparse([a(1:i-1) ; b(1:i-1)]); % Recurrence coefficients
            norms = sqrt(norms(1:i-1)); % Polynomial norms

            function pN = eval(i,a,b,x)
                % pN = eval(i,a,b,x)
                % Evaluates the polynomial of degree i defined by the coefficients a and b at points x
                % i: 1-by-1 integer
                % a: 1-by-max+1 array
                % b: 1-by-max+1 array
                % x: n-by-m array
                % pN: n.m-by-1 array

                if i < 0
                    pN = zeros(size(x));
                elseif i == 0
                    pN = ones(size(x));
                else
                    pNm2 = 1;
                    pNm1 = (x(:) - a(1));
                    pN = pNm1;
                    for N = 3:i+1
                        pN = (x(:) - a(N-1)).*pNm1 - b(N-1)*pNm2;
                        pNm2 = pNm1;
                        pNm1 = pN;
                    end
                end
            end

            function b = isOrth(pnp1,pn,pnm1,wts)
                % b = isOrth(pnp1,pn,pnm1,wts,N)
                % Determines if the polynomial pnp1 is orthogonal to the polynomials pn and pnm1, according to RandomVariable r
                % pnp1: n-by-numel(wts) array
                % pn: n-by-numel(wts) array
                % pnm1: n-by-numel(wts) array
                % wts: 1-by-numel(wts) array
                % b: boolean

                tol = 1e-4; % Tolerance for the inner product to be considered as 0

                d1 = abs(sum((pnp1.*pn)*wts(:)));
                d2 = abs(sum((pnp1.*pnm1)*wts(:)));
                if d1 < tol && d2 < tol
                    b = 1;
                else
                    b = 0;
                end
            end

        end

    end
end

