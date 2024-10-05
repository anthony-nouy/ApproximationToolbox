% GREEDYSUBSAMPLING - Remove greedily points from a sample xn to provide a sample stable for
% least-squares approximation in the sense of [1]
%
% [xnew, deltanew] = GREEDYSUBSAMPLING(x, basis, delta)
% x: initial sample of length n
% basis: Approximation basis to build the least-squares approximation
% delta: threshold for the stability bound
% xnew: subsaample of length k <= n
% deltanew: distance beween the empirical Gram matrix computed with xnew and delta
%
% Reference:
% [1] C. Haberstich, A. Nouy, and G. Perrin. Boosted optimal weighted least-squares. Mathematics of Com- putation, 91(335):1281–1315, 2022.

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

function [xnew, deltanew ] = greedySubsamplingBOWLS(x, basis, delta)

m = cardinal(basis);

% Initialisation
A = basis.eval(x);
W = diag(sqrt(m./sum(A.^2,2)));
WA = W*A;
G = 1/length(x(:,1))*(WA')*WA;
while (norm(G-eye(m)) < delta && length(x(:,1)) > m)
    l = length(x(:,1));
    residuals = ones(1,l);
    parfor i =1:l
        WAnew = W*A;
        WAnew(i,:) = [];
        G = 1/(l-1)*(WAnew')*(WAnew);
        residuals(i) = norm(G-eye(m));
    end
    [minires, indres] = min(residuals);
    xnew = x;
    xnew(indres,:) = [];
    WA = W*A;
    WA(indres,:) = [];
    G = 1/length(xnew(:,1))*(WA')*WA;
    if norm(G-eye(m)) > delta
        % gridalpha = gridalpha;
        break
    else
        x = xnew;
        A(indres,:) =[];
        W(indres,:) = [];
        W(:,indres) = [];
        deltanew = norm(G-eye(m));
    end
end


% outputs
xnew = x;
A = basis.eval(x);
W = diag(sqrt(m./sum(A.^2,2)));
WA = W*A;
G = 1/length(x(:,1))*(WA)'*WA;
deltanew = norm(G-eye(m));