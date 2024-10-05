% GREEDYALGORITHML2 -
% Given a N-by-M matrix F = (F^1, ... , F^M), returns
% n column indices j_1 ... j_n constructed by a greedy algorithm
% j_k in arg max_{j} || F^j - P_{k-1}(F^j) ||
% where P_{k-1} is the l2-orthogonal projection
% onto the linear span V_{k-1} of columns F^{j_1} , ... , F^{j_{k-1}} ,
% an orthonormal basis of V_{k-1} is given by vectors
% v_i \propto F^{j_i} - P_{i-1}(F^{j_i})
%
%
% [J,V,L] = greedyAlgorithml2(F,n,vargargin)
% F : matrix of size N-by-M
% n : integer (min(N,M) by default)
% J : n-by-1 array  containing the n indices
% V : orthogonal matrix of size N-by-n whose columns are the vectors v_i
% L : n-M array whose row L(k,:) contains the vector to maximize to get the
% k-th index
%
% [J,V,L] = greedyAlgorithml2(F,n,'adaptive',true)
% adaptive version which constructs points sequentially from submatrices
% F(1:k,:), k=1...size(F,1)
%
% [J,V,L] = greedyAlgorithml2(F,n,'weights',w)
% The vector to maximize is defined by 
% || F^j - P_{k-1}(F^j) ||^2 w_j
%
% w : 1-by-M array  containing weigths w_j (by default or if empty, w_j=1)
%
%
% Optional arguments are provided in the form
% 'PropertyName1',PropertyValue1,'PropertyName2',PropertyValue2 and so on. 
% The list of option names and default values.
%

%
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

function [J,V,L] = greedyAlgorithml2(F,n,varargin)
N = size(F,1);
M = size(F,2);

if nargin==1 || isempty(n)
    n=min(N,M);
end

if n>min(N,M)
    error('number of rows and columns should not be smaller than the requested number of indices')
end

param = inputParser;
addOptional(param,'adaptive',false);
addOptional(param,'weights',ones(1,M));
parse(param,varargin{:});
w = param.Results.weights(:).';
adaptive = param.Results.adaptive;


switch adaptive
    case false
        V = zeros(N,0);
        J = zeros(n,1);
        L = zeros(n,M);
        for k=1:n
            if k>1
                F = F - V*(V'*F);
            end
            L(k,:) = sum(F.^2,1).*w;
            [err,J(k)]=selectMax(L(k,:));
            v = F(:,J(k));

            if err<eps
                L = L(1:k-1,:);
                J = J(1:k-1);
                warning('algorithm stopped')
                break
            else
                v = v - V*(V'*v);
                V = [V , v/norm(v)];
            end
        end

    case true
        J = zeros(n,1);
        L = zeros(n,M);
        for k=1:n
            if k>1
                V = orth(F(1:k,J(1:k-1)));
                G = F(1:k,:) - V*V'*F(1:k,:);
            else
                G = F(1,:);
            end
            L(k,:) = sum(G.^2,1).*w;
            [err,J(k)]=selectMax(L(k,:));

            if err<eps
                L = L(1:k-1);
                J = J(1:k-1);
                warning('algorithm stopped')
                break
            end
        end

    otherwise
        error('Bad option.')
end


end


function [umax,j] = selectMax(u)

            umax = max(u);
            idx = find(u==umax);
            if length(idx)==1
                j = idx;
            else
                j = idx(randi(length(idx)));
            end

end

