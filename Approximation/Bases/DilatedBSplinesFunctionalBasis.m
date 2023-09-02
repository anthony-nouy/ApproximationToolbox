% Class DilatedBSplinesFunctionalBasis

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

classdef DilatedBSplinesFunctionalBasis < FunctionalBasis


    properties
        basis
        indices
    end

    methods

        function BF = DilatedBSplinesFunctionalBasis(B,I)
            % BF = DilatedBSplinesFunctionalBasis(B,I)
            %
            % Functional basis of DilatedBSplines of degree n on 
            % (0,1), using b-adic dilations
            % I = [L,J] is a N-by-2 array providing the levels and local
            % indices of the dilated BSplines
            %
            % B: DilatedBSplines
            % I: N-by-2 array 
            % BF: DilatedBSplinesFunctionalBasis

            if ~isa(B,'DilatedBSplines')
                error('must provide a DilatedBSplines')
            end

            BF.basis = B; 
            BF.indices = I;
  
        end

        function Bx = eval(B,x)

            Bx = B.basis.eval(B.indices, x);

        end


        function n = cardinal(B)
            n = size(B.indices,1);
        end

        function n = ndims(~)
            n = 1;
        end


%         function i = indicesToIndex(B,l,j)
% 
%             if nargin==2
%                 j=l(:,2);
%                 l=l(:,1);
%             end
% 
%             b = B.basis.base;
%             m = B.basis.degree ; 
% 
%             nok = j < -m | j > (b.^l -1)/(b-1) + m ; 
%             if any(nok)
%                 error('local index not in the correct range')
%             end
% 
%             i = (b.^l - 1)/(b-1) + l*m + j + m ; 
% 
%         end
% 
%         function I = indexToIndices(B,i)
% 
%             m = B.basis.degree;
%             l = zeros(size(i));
%             j = zeros(size(i));
%             ok = false(size(i));
%             k = 0;
%             rep = find(i <= m);
%             l(rep)=0;
%             j(rep)=i(rep) - m;
%             ok(rep)=true;
%             while ~all(ok)
%                 k=k+1;
%                 ik = B.indicesToIndex(k,-m);
%                 ikp1 = B.indicesToIndex(k+1,-m);
%                 rep = find(i>=ik & i< ikp1);
%                 l(rep)=k;
%                 j(rep)=i(rep) - ik - m ; 
%                 ok(rep)=true;
%             end
% 
%             I = [l,j];
% 
%         end



        function dBx = evalDerivative(B,n,x)
            dBx = B.basis.evalDerivative(n,B.indices,x);
        end
    end


    methods (Static)

        function B = withLevelBoundedBy(n,b,l)
            % function B = withBoundedLevel(n,b,l)
            % create a DilatedBSplinesFunctionalBasis with degree n
            % b-adic dilation, and functions with level less than l

            B = DilatedBSplines(n,b);
            I = B.indicesWithLevelBoundedBy(l);
            B = DilatedBSplinesFunctionalBasis(B,I);


        end

    end


end