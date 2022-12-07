% Class RandomVariable

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

classdef ShiftedRandomVariable < RandomVariable

    properties
        rv % RandomVariable
        b % shift
        s % scaling
    end

    methods

        function X = ShiftedRandomVariable(rv,b,s)
            % X = ShiftedRandomVariable(rv,b,s)
            % rv: RandomVariable
            % b: shift
            % s: scaling


            X@RandomVariable(['shifted' rv.name]);
            X.rv = rv;
            X.s = s;
            X.b = b;
            X.moments = [];

        end

        function p = orthonormalPolynomials(X,varargin)
            % ORTHONORMALPOLYNOMIALS - orthonormal polynomials according to the measure of the RandomVariable
            %
            % p = ORTHONORMALPOLYNOMIALS(X)
            % X: ShiftedRandomVariable
            % p: ShiftedOrthonormalPolynomials
            p = orthonormalPolynomials(X.rv);
            p = ShiftedOrthonormalPolynomials(p,X.b,X.s);
        end

        function P = cdf(X,x)
            % P = cdf(X,x)
            % Computes the cumulative distribution function of the random variable X at point(s) x
            % X: RandomVariable
            % x: double
            % P: double

            P = cdf(X.rv,(x-X.b)/X.s);
            P(P==1) = 1-eps;
            P(P==0) = eps;
        end

        function P = icdf(X,x)
            % P = cdf(X,x)
            % Computes the inverse cumulative distribution function of the random variable X at point(s) x
            % X: RandomVariable
            % x: double
            % P: double

            P = icdf(X.rv,x)*X.s+X.b;
        end


        function ok = eq(r1,r2)
            % ok = eq(r1,r2)
            % Checks if two random variables r1 and r2 are equal
            % r1: RandomVariable
            % r2: RandomVariable
            % ok: boolean

            if ~strcmp(class(r1),class(r2))
                ok = 0;
            else
                ok = (r1.s==r2.s) && (r1.b==r2.b) && (r1.rv == r2.rv);
            end

        end



        function m = max(X)
            % m = max(X)
            % Computes the maximum value that can take the inverse cumulative distribution function of the random variable X
            % X: RandomVariable
            % m: 1-by-1 double

            m = max(X.rv)*X.s+X.b;
        end

        function s = support(X)
            % s = support(X)
            % Returns the support of the random variable
            % X: ShiftedRandomVariable
            % s: 1-by-2 double

            s = support(X.rv)*X.s + X.b;
        end


        function m = mean(X)
            % m = mean(X)
            % Computes the mean of the random variable X
            % X: RandomVariable
            % m: 1-by-1 double

            m = mean(X.rv)*X.s+X.b;
        end

        function m = min(X)
            % m = min(X)
            % Computes the minimum value that can take the inverse cumulative distribution function of the random variable X
            % X: RandomVariable
            % m: 1-by-1 double

            m = min(X.rv)*X.s+X.b;
        end


        function px = pdf(X,x)
            % px = pdf(X,x)
            % Computes the probability density function of the random variable X at points x
            % X: RandomVariable
            % x: double
            % px: double

            px = pdf(X.rv,(x-X.b)/X.s)/X.s;
        end

        function r = random(X,n,varargin)
            % r = random(X,n)
            % Generates n random numbers according to the distribution of the RandomVariable X
            % X: RandomVariable
            % n: integer
            % r: n-by-1 double

            if nargin==1
                n=1;
            end

            r = random(X.rv,n,varargin{:})*X.s+X.b;
        end

        function s = std(X)
            % s = std(X)
            % Computes the standard deviation of the random variable X
            % X: RandomVariable
            % s: 1-by-1 double

            s = std(X.rv)*X.s;
        end

        function v = variance(X)
            % s = variance(X)
            % Computes the variance of the random variable X
            % X: RandomVariable
            % v: 1-by-1 double

            v = variance(X.rv)*X.s^2;
        end


        function Y = shift(X,b,s)

            Y = X;
            Y.b = X.b*s + b;
            Y.s = X.s*s;
            if Y.b == 0 && Y.s ==1
                Y=X.rv;
            end

        end


    end
end