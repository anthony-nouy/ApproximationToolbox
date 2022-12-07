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
        function p = ChebyshevPolynomials(kind,varargin)
            if nargin == 0 || isempty(kind)
                kind = 1;
            end

            if kind~=1
                error('Only the Chebyshev polynomials of first kind are implemented.')
            end

            p.kind = kind;

            switch p.kind
                case 1
                    p.measure = ShiftedRandomVariable(BetaRandomVariable(0.5,0.5),-1,2);
                otherwise
                    error('Only the Chebyshev polynomials of first kind are implemented.')
            end

        end



        function [recurr,norms] = recurrenceMonic(p,n)
            % function [recurr,norms] = recurrenceMonic(p,n)
            % Computes the coefficients of the three-term recurrence used to construct the monic polynomials
            % p_{n+1}(x) = (x-a_n)p_n(x) - b_n p_{n-1}(x), a_n and b_n are
            % the three-term recurrence coefficients
            % p : ChebyshevPolynomials
            % n: integer
            % recurr: 2-by-(n+1) double
            % norms: 1-by-(n+1) double (norms of monic polynomials)

            switch p.kind
                case 1
                    a = zeros(1,n+1);
                    b = [0 , ones(1,n)/4];

                    recurr = [a;b];
                    if nargout==2
                        norms = [1 , 2.^(1/2-1:n)];
                    end                    

                otherwise
                    error('Only the Chebyshev polynomials of first kind are implemented.')


            end


        end


    end



end