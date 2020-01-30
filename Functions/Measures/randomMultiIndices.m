% function I = randomMultiIndices(sz)
% Returns a random vector uniformly distributed on I1x...xId
% sz : 1-by-d array or cell array 
% I : RandomVector
% 
% if sz is a 1-by-d array, then Ik = {1,....sz(k)}
% if sz is a cell array, then Ik = sz{k}  

% Copyright (c) 2020, Loic Giraldi, Erwan Grelier, Anthony Nouy
% 
% This file is part of the Matlab Toolbox ApproximationToolbox.
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


function I = randomMultiIndices(sz)
d=length(sz);
I = cell(1,d);
if ~isa(sz,'cell')
    for k=1:d
        I{k}=(1:sz(k))';
    end
else
    I=sz;
end
for k=1:d
    I{k} = DiscreteRandomVariable(I{k}(:))';
end
I = RandomVector(I);
end