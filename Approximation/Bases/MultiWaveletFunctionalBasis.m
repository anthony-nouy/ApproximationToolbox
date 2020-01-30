% Class MultiWaveletFunctionalBasis

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

classdef MultiWaveletFunctionalBasis < FunctionalBasis
    
    properties
        F % Basis for the father wavelet
        M % Basis for the mother wavelet
        s % Resolution
        a % Scaling factor
    end
    
    methods
        function hx = eval(h,x,indices)
            % hx = eval(h,x)
            % Evaluates the functional basis at the points x
            % h: MultiWaveletFunctionalBasis
            % x: N-by-d array
            % hx: N-by-n array
            
            supp = support(h.measure);
            hx = cell(1,h.s+2);
            hx{1} = h.F.eval(x);
            m = cardinal(h.M);
            for i=0:h.s-1
                ni = (h.a-1)*h.a^i;
                j = 0:ni-1;
                u = (h.a^(i)*repmat((x - supp(1))/diff(supp),1,ni)-repmat(j,length(x),1));
                u = u*diff(supp) + supp(1);
                u=u(:);
                H = zeros(size(x,1)*ni,m);
                I = isIn(supp,u);
                if any(I)
                    H(I,:) = h.a^(i/2)*h.M.eval(u(I));
                end
                H = reshape(H,[size(x,1),ni,m]);
                H = permute(H,[1,3,2]);
                hx{i+2} = reshape(H,size(x,1),ni*m);
            end
            hx = [hx{:}];
            
            if nargin >= 3
                hx = hx(:,indices);
            end
        end
        
        function gx = evalDerivative(h,n,x)
            % gx = evalDerivative(h,x)
            % Evaluates the n-derivative of h at the points x
            % k: MultiWaveletFunctionalBasis
            % n: 1-by-d array of integers
            % x: N-by-d array
            % hx: N-by-n array
            
            error('Method not implemented.')
        end
        
        function n = cardinal(h)
            if h.s==0
                n = cardinal(h.F);
            else
                n = cardinal(h.F) + (h.a-1)*cardinal(h.M)*sum(h.a.^(0:h.s-1));
            end
        end
        
        function d = ndims(h)
            d = ndims(h.F);
        end
    end
end