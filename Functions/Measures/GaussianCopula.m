% Class GaussianCopula

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

classdef GaussianCopula
    
    properties
        R % Correlation matrix
    end
    
    methods
        function C = GaussianCopula(R)
            C.R = R;
        end
        
        function p = pdf(C,u)
            N = NormalRandomVariable(0,1);
            x = icdf(N,u);
            p = 1/sqrt(det(C.R))*exp(-1/2*sum(x'.*(C.R\x'-x'),1)');
        end
    end
    
    methods (Static)
        function C = random(d,n)
            if nargin==1
                n=1000;
            end
            R = GaussianCopula.randomCorrelationMatrix(d,n);
            C = GaussianCopula(R);
        end
        
        function C = randomBlockDiagonal(d,n)
            r = floor(d/n);
            nb = zeros(1,n);
            nb(:) = r;
            if sum(nb)<d
                nb = [nb,d-sum(nb)];
            end
            R = sparse([],[],[],d,d,sum(nb.^2));
            for i=1:length(nb)
                rep = sum(nb(1:i-1))+(1:nb(i));
                R(rep,rep) = GaussianCopula.randomCorrelationMatrix(length(rep),1000);
            end
            C = GaussianCopula(R);
        end
    end
    
    methods (Static, Hidden)
        function R = randomCorrelationMatrix(d,n)
            R = 2*rand(n,d)-1;R = R'*R;
            D = diag(diag(R).^-(1/2));
            R = D*R*D;
        end
    end
end