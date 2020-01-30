% MAGICINDICES - Set of n magic indices (i_1,j_1)...(i_n,j_n) constructed by a greedy algorithm
% [ik,jk] = arg max_{i,j} | F_ij - I_{k-1}(F)_ij |
% where I_{k-1}(F) is the rank-(k-1) matrix which interpolates F
% on the cross corresponding
% to rows (i_1,...i_{k-1}) and columns (j_1,...j_{k-1})
%
% [I,J] = MAGICINDICES(F,n)
% F : matrix of size N-by-M
% I,J : arrays of size n-by-1 containing the n magic indices
%
% [I,J] = MAGICINDICES(F,n,'left')
% column index j_k is equal to k
% ik = arg max_{i} max_{1\le j \le M} | F_ij - I_{k-1}(F)_ij |
% I : array of size n-by-1
% J = (1:n)'
%
% [I,J] = MAGICINDICES(F,n,'right')
% row index i_k is equal to k and
% jk = arg max_{j} max_{1\le i \le N} | F_ij - I_{k-1}(F)_ij |
% J : array of size n-by-1
% I : (1:n)'

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

function [I,J] = magicIndices(F,n,option)
if nargin==1 || isempty(n)
    n=min(size(F));
end

if nargin<3
    option = 'leftright';
end

switch option
    case 'leftright'
        G = zeros(size(F));
        I=[];
        J=[];
        for k=1:n
            [~,i] = max(max(abs(F-G),[],2),[],1);
            [~,j] = max(abs(F(i,:)-G(i,:)),[],2);
            I=[I;i];
            J=[J;j];
            G = F(:,J)*(F(I,J)\F(I,:));
        end
        
    case 'left'
        G = zeros(size(F));
        I=[];
        J=[];
        for k=1:n
            [~,i] = max(max(abs(F(:,1:k)-G(:,1:k)),[],2),[],1);
            I=[I;i];
            J=[J;k];
            G = F(:,J)*(F(I,J)\F(I,:));
        end
        
    case 'right'
        [J,I] = magicIndices(F.',n,'left');
    otherwise
        error('Bad option.')
end