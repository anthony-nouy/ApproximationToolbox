% ISADMISSIBLETREEBASEDRANK - Checks if a given tuple x is an admissible tree-based rank for the DimensionTree T
%
% ok = ISADMISSIBLETREEBASEDRANK(T,x)
% T: DimensionTree
% r: 1-by-T.nbNodes double of integers
% ok: logical

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

function a = isAdmissibleTreeBasedRank(T,r)
if r(T.parent==0)~=1
    a = 0;
    warning('The root rank should be 1.')
    return
else
    a=true;
end

for i= T.internalNodes
    ch = nonzeros(T.children(:,i));
    a = a & (r(i) <= prod(r(ch)));
    for mu=1:length(ch)
        nomu = setdiff(1:length(ch),mu);
        a = a & (r(ch(mu)) <= r(i)*prod(r(ch(nomu))));
    end
end
end