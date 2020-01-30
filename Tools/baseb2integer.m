% BASEB2INTEGER - Returns the integers with given representations in base b
% 
% i = BASEB2INTEGER(I,b)
% I: array of size n-by-d
% A row of I contains d integers [i_1,...,i_d] in {0,...,b-1} associated with an integer i = \sum_{k=1}^d i_k b^(d-k) in [0,b^d-1]

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

function i = baseb2integer(I,b)
d = size(I,2);
I = fliplr(I);
I = I+1;
I = mat2cell(I,size(I,1),ones(1,d));
i = sub2ind(repmat(b,1,d),I{:})-1;
end