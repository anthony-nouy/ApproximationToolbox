% DECOMPOSEPERMUTATION - Decompose a permutation
%
% [pelem,psequence] = DECOMPOSEPERMUTATION(p)
% pfinal: a permutation of 1:d
% pelem: sequence of elementary permutations (permutation of two consecutive numbers) to go from 1:d to p
% psequence: sequence of permutations

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

function [pelem,psequence] = decomposePermutation(pfinal)
d = length(pfinal);
p0=1:d;
%mu = 1;
k=0;
psequence = p0;
p=p0;
pelem = zeros(0,2);
while any(p~=pfinal)
    dist = zeros(1,d-1);
    for mu=1:d-1
        i1 = find(pfinal==p(mu));
        i2 = find(pfinal==p(mu+1));
        dist(mu) = i2-i1;
    end
    mu = find(dist<0 & dist==min(dist));
    
    if ~isempty(mu)
        mu = mu(1);
        p(mu:mu+1)= p(mu+1:-1:mu);
        pelem = [pelem;mu,mu+1];
        psequence = [psequence;p];
    end
    % i1 = find(pfinal==p(mu));
    % i2 = find(pfinal==p(mu+1));
    % if i1>i2
    % p(mu:mu+1)= p(mu+1:-1:mu);
    % k=k+1;
    % pelem = [pelem;mu,mu+1];
    % psequence = [psequence;p];
    % end
    %
    % if mu<d-1
    %     mu=mu+1;
    % else
    %     mu=1;
    % end
end

end