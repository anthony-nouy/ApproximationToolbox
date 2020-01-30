% ISIN - Returns a boolean array of length size(x,1)
% such that ok(k) = true if the point x(k,:) is in
% the box given by B
%
% ok = ISIN(B,x)
% B is of size 2-by-dim and contains the two extreme points
% B(1,:) and B(2,:) of the box

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

function ok = isIn(B,x)
if numel(B)==2
    B=B(:);
end
dim = size(B,2);
ok = true(size(x,1),1);
for k=1:dim
    ok = ok & (x(:,k)>=B(1,k)) & (x(:,k)<=B(2,k));
end
end