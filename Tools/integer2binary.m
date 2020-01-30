% INTEGER2BINARY - Returns the binary representation [i_1,...,i_d] of a set of non negative integers i= \sum_{k=1}^d i_k 2^(d-k) in [0,2^d-1]
%
% I = INTEGER2BINARY(i,d)
% i: array containing n integers
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

function I = integer2binary(i,d)
c = dec2bin(i);
if nargin==1
    d = size(c,2);
elseif size(c,2)>d
    error('Integer values must be less than 2^d.')
end

I = zeros(numel(i),size(c,2));
for k=1:size(c,2)
    I(:,k)=str2num(c(:,k));
end

I  = [zeros(numel(i),d-size(c,2)),I];
end