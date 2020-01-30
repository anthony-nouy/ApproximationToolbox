% CHEBYCHEVPOINTS - First n chebychev points in [s(1),s(2)]
%
% x = CHEBYCHEVPOINTS(n,s)
% n: integer
% s: 1-by-2 double (s=[-1,1] by default)

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

function x = chebychevPoints(n,s)
x = cos(pi*(2*(1:n)'-1)/2/n);
if nargin>1
    x = 1/2*(s(1)+s(2)) + 1/2*(s(2)-s(1))*x;
end

end