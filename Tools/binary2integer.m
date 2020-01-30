% BINARY2INTEGER - Provides the integers with given binary representations
%
% i = BINARY2INTEGER(I)
% I: array of size n-by-d
% A row of I contains the binary coefficients [i_1,...,i_d] of the integer \sum_{k=1}^d i_k 2^(d-k) contained in [0,2^d-1]

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

function i = binary2integer(I)
i = char(zeros(size(I)));
for k=1:size(I,2)
    i(:,k) = num2str(I(:,k));
end
i = bin2dec(i);
end