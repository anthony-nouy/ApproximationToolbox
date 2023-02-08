% Class BSplinesFunctionalBasis

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

classdef BSplinesFunctionalBasis < FunctionalBasis


    properties
        knots
        degree
    end

    methods
        function B = BSplinesFunctionalBasis(x,s)
            % B = BSplinesFunctionalBasis(x,s)
            % BSplines with knots x and degree s
            %
            % x: N-by-1 array
            % B: BSplinesFunctionalBasis
            % s: integer

            B.knots = x(:);
            B.degree = s;

        end

        function n = cardinal(B)
            n = length(B.knots)-1-B.degree;
        end

        function n = ndims(B)
            n=1;
        end        


        function [Bxn,Bx] = eval(B,x)
            % Evaluate the B-Splines
            %
            % function Bx = eval(B,x)
            % B : BSplinesFunctionalBasis
            % x : n-by-1 array
            % Bx : n-by-(length(B.knots)-1-B.degree) array

            t = B.knots;
            m = length(t);
            n = length(x);
            x = x(:);

            Bx = zeros(n,length(t)-1,B.degree+1);
            for i=1:m-1
                Bx(:,i,1) = (x>t(i)).*(x<=t(i+1));
            end
            for j=1:B.degree
                for i=1:m-1-j
                    Bx(:,i,j+1) = (x - t(i)) / (t(i+j)-t(i)).*Bx(:,i,j)+ ...
                        (t(i+j+1)-x) / (t(i+j+1)-t(i+1)).*Bx(:,i+1,j);
                end
            end
            Bxn = Bx(:,1:m-1-B.degree,end);

        end


        function dBx = evalDerivative(B,n,x)
            % Evaluate the n-th derivative of the B-Spline
            %
            % function dBx = evalDerivative(B,n,x)
            % B : BSplinesFunctionalBasis
            % n : integer
            % x : n-by-1 array
            % dBx : n-by-(length(B.knots)-1-B.degree) array

            t = B.knots;
            m = length(t);
            x = x(:);

            [~,dBx] = B.eval(x);

            for k = 1:n
                dBxold = dBx;

                dBx = zeros(length(x),length(t)-1,B.degree+1);
                for j=1:B.degree
                    for i=1:m-1-j
                        dBx(:,i,j+1) = 1 / (t(i+j)-t(i)) *dBxold(:,i,j) -  ...
                            1 / (t(i+j+1)-t(i+1)) *dBxold(:,i+1,j) + ...
                            (x - t(i)) / (t(i+j)-t(i)).*dBx(:,i,j)+ ...
                            (t(i+j+1)-x) / (t(i+j+1)-t(i+1)).*dBx(:,i+1,j);
                    end
                end

            end
            dBx = dBx(:,1:m-1-B.degree,end);

        end

        function I = gaussIntegrationRule(h,n)
            % function I = gaussIntegrationRule(h,n) 
            % returns an Integration Rule using n-points
            % gauss integration rule per interval
            % h : BSplinesFunctionalBasis
            % n : integer
            % I : IntegrationRule

            p = PiecewisePolynomialFunctionalBasis(h.knots,0);
            I = gaussIntegrationRule(p,n);
        end

    end


    methods (Static)

        function B = cardinalBspline(m)
            % Return the cardinal B-Spline of degree m
            %
            % function B = cardinalBspline(m)
            % m : integer
            % B : BsplinesFunctionalBasis with cardinal 1

            B = BSplinesFunctionalBasis(0:m+1 , m);

        end


    end


end