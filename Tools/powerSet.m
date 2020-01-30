% POWERSET - Returns the power set of {1,...,d} or of a subset of {1,...,d} characterized by the logical array u
%
% v = POWERSET(d)
% Returns the power set of {1,...,d}
% d: integer
% v: N-by-d logical with N=2^d
%
% v = POWERSET(u)
% Returns the power set of a subset of {1,...,d} characterized by the logical array u
% u: 1-by-d logical
% v: N-by-d logical where N=2^s with s = nnz(u)

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

function v = powerSet(d)
if isa(d,'logical')
    u = d;
    d = size(u,2);
else
    u = true(1,d);
end

b = ones(1,nnz(u));
I = MultiIndices.boundedBy(b);
I = logical(I.array);
v = false(size(I,1),d);
v(:,u)=I;
end