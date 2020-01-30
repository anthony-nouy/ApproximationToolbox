% INTEGER2BASEB - Returns the representation [i_1,...,i_d] in base b of a set of non negative integers
% i = \sum_{k=1}^d i_k b^(d-k) in [0,b^d-1]
%
% i = INTEGER2BASEB(i,b,d)
% i: array containing n integers
% b: integer (>=2)
% d: integer (if not specified, minimal integer allowing the representation of max(i))
% I: array of size n-by-d

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

function I = integer2baseb(i,b,d)
if nargin==2
    d = ceil(log(max(i)+1)/log(b));
end
I = cell(1,d);
[I{:}] = ind2sub(repmat(b,1,d),i(:)+1);
I = [I{:}]-1;
I = fliplr(I);
end