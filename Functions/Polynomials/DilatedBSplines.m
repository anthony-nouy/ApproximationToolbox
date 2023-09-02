% Class DilatedBSplines

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

classdef DilatedBSplines


    properties
        degree
        base = 2
    end

    methods
        function B = DilatedBSplines(n,b)
            % B = DilatedBSplines(n,b)
            % DilatedBSplines of degree n on (0,1), using b-adic dilations
            %
            % n: integer
            % b: integer (2 by default)
            % B: DilatedBSplines

            B.degree = n; 
            if nargin==2
                B.base = b;
            end
  
        end

        function Bx = eval(B,i,x)
            % function Bx = eval(B,i,x)
            %
            % Evaluate the dilated B-Splines of indices i=[l,j]
            % at points x, with l the level and j the local index
            %
            % B : DilatedBSplines
            % i : N-by-2 array of integers 
            % x : n-by-1 array
            % Bx : n-by-N array

            m=B.degree;
            b=B.base;
            psi = BSplinesFunctionalBasis.cardinalBspline(m);
            l=i(:,1);
            j=i(:,2);
            Bx = zeros(length(x),length(l));
            X = x(:)*b.^(l') - repmat(j',length(x),1);
            S = repmat(b.^(l'/2),length(x),1);
            Bx(:) = psi.eval(X(:)).*S(:);

        end

        function dBx = evalDerivative(B,k,i,x)
            % function dBx = evalDerivative(B,k,i,x)
            % 
            % Evaluate the k-th derivative of dilated B-Splines of indices i=[l,j]
            % at points x, with l the level and j the local index
            %
            % B : DilatedBSplines
            % k : integer
            % i : N-by-2 arrays of integers 
            % x : n-by-1 array
            % dBx : n-by-N array

            m=B.degree;
            b=B.base;
            psi = BSplinesFunctionalBasis.cardinalBspline(m);  
            l=i(:,1);
            j=i(:,2);
            dBx = zeros(length(x),length(l));
            X = x(:)*b.^(l') - repmat(j',length(x),1);
            S = repmat(b.^(l'*(k+1/2)),length(x),1);
            dBx(:) = psi.evalDerivative(k,X(:)).*S(:);

        end


        function [l,j] = indicesWithLevelBoundedBy(B,L)
            % function I = indicesWithLevelBoundedBy(B,L)
            %
            % returns le indices l and j of Dilated BSplines of level less
            % then L
            %
            % l,j: n-by-1 arrays
            
            l = zeros(0,1);
            j = zeros(0,1);
            for k=0:L
                [lk,jk] = B.indicesWithLevel(k);
                l = [l;lk];
                j = [j;jk];
            end
            if nargout==1
                l = [l,j];
            end

            
        end

        function [l,j] = indicesWithLevel(B,L)
            % function I = indicesWithLevel(B,L)
            % returns le indices (l,j) of Dilated BSplines of level L
            % then L
            % I: n-by-2 array  
            % (first column if level l, second column is the local index j) 
            % 
            % function [l,j] = indicesWithLevel(B,L)
            % l,j: n-by-1 arrays
            m = B.degree ;
            b = B.base;
            j = (-m:b^L-1)';
            l = zeros(length(j),1);
            l(:) = L;
            if nargout==1
                l = [l,j];
            end
        end



    end


    methods (Static)

    end


end