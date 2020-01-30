% LYAPSYM - Solve AX+XA = C assuming A=A'
%
% X = LYAPSYM(A,C)

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

function X = lyapsym(A,C)
I = speye(size(A));

n = size(A,1);
[U,R] = eig(full(A));
F = U'*C*U;

R = diag(R);
O = ones(n,1);

R = R*O'+O*R';
Y = F./R;

X = U*Y*U';
end