% HADAMARDTRANSFORM - Returns the Hadamard Transform of a vector x whose length is a power of 2
%
% Hx = HADAMARDTRANSFORM(x)
% x, Hx: 2^d-by-1 double

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

function Hx = hadamardTransform(x)
d=log2(length(x));
if round(d)~=d 
    error('The length of the vector should be a power of 2')
end
H = repmat({[1,1;1,-1]/sqrt(2)},1,d);
Hx = timesMatrix(FullTensor(reshape(x,2*ones(1,d))),H);
Hx = double(Hx);
Hx = Hx(:);
