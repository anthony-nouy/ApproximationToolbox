% Class LegendrePolynomials

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

classdef LegendrePolynomialsLebesgue < OrthonormalPolynomials

    methods
        function p = LegendrePolynomialsLebesgue(varargin)
            % p = LegendrePolynomialsLebesgue(varargin)
            % Polynomials defined on [-1,1], orthonormal with respect to the Lebesgue measure  
            % p: LegendrePolynomialsLebesgue  

            p.measure= LebesgueMeasure(-1,1);

        end

        function [recurr,norms] = recurrenceMonic(~,n)
            % function [recurr,norms] = recurrenceMonic(p,n)
            % Computes the coefficients of the three-term recurrence used to construct the monic polynomials
            % p_{n+1}(x) = (x-a_n)p_n(x) - b_n p_{n-1}(x), a_n and b_n are
            % the three-term recurrence coefficients
            % p : LegendrePolynomials
            % n: integer
            % recurr: 2-by-(n+1) double
            % norms: 1-by-(n+1) double (norms of monic polynomials)
            a = zeros(1,n+1);
            b = (0:n).^2 ./ (4*(0:n).^2 - 1);
            recurr = [a;b];
            if nargout==2
                norms = sqrt(1./(2*(0:n)+1)).*2.^(0:n).*factorial(0:n).^2 ./ factorial(2*(0:n)) * sqrt(2);
            end

        end

    end


end