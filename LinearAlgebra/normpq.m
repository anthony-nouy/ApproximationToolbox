% NORMPQ - Lpq norm of matrix A, with p,q>=1
%
% x = NORMPQ(A,p,q)

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

function x = normpq(A,p,q)
if p==Inf && q==Inf
    x = max(max(abs(A)));
elseif q==Inf
    x = max(sum(abs(A).^p,1).^(1/p));
elseif p==Inf
    x = max(sum(abs(A).^q,2).^(1/q));
else
    x = sum(sum(abs(A).^p,1).^(q/p))^(1/q);
end