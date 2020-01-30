% SINGULARVALUES - Singular values of the array x
%
% sv = SINGULARVALUES(x)
% 
% if ndims(x)=2, sv = vector of singular values of matrix x
% if ndims(x)>2, sv = cell containing the vectors of singular values of
% matricisations of x

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

function sv = singularValues(x)
if ~isa(x,'double')
    warning('Input should be a double.')
end

if ndims(x) == 2
    sv = svd(x);
else
    x = FullTensor(x);
    sv = singularValues(x);
end