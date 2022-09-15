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
        function p = JacobiPolynomials(alpha,beta,varargin)
            p.measure = ShiftedRandomVariable(BetaRandomVariable(alpha+1,beta+1),-1,2);
            p.alpha = alpha;
            p.beta = beta;

        end


        function [recurr,norms] = recurrenceMonic(p,n)
            % function [recurr,norms] = recurrenceMonic(p,n)
            % Computes the coefficients of the three-term recurrence used to construct the monic polynomials
            % p_{n+1}(x) = (x-a_n)p_n(x) - b_n p_{n-1}(x), a_n and b_n are
            % the three-term recurrence coefficients
            % p : JacobiPolynomials
            % n: integer
            % recurr: 2-by-(n+1) double
            % norms: 1-by-(n+1) double (norms of monic polynomials)
            alpha= p.alpha;
            beta = p.beta;

            j = 1:n;
            a = (beta^2 - alpha^2) ./ ((alpha + beta + 2*j).*(alpha + beta + 2*j + 2));
            a =[(beta-alpha)/(alpha + beta + 2),a];
            j = 1:n;
            b = (4*j.*(j + alpha).*(j + beta).*(j + alpha + beta)) ./ ...
                ((2*j + alpha + beta).^2.*(2*j + alpha + beta + 1).*(2*j + alpha + beta - 1));
            b = [0,b];
            recurr = [a;b];


            if nargout==2
                n=0:n;
                mu = 2^(alpha+beta+1)*gamma(alpha+1)*gamma(beta+1)/gamma(alpha+beta+1);
                norms = 2.^n./gamma(alpha+beta+2*n+1).*sqrt(2.^(2*n+alpha+beta+1)./(2*n+alpha+beta+1).*gamma(n+alpha+1).*gamma(n+beta+1)...
                    .*gamma(n+alpha+beta+1)./gamma(alpha+beta+2*n+1).^2.*factorial(n)/mu);
            end

        end

    end


end