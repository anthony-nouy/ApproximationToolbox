% Tensor Truncation (TT and TuckerLikeFormat): multivariate function evaluated on a grid

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

clearvars; clc; close all

%% Definitions
d = 6;
Nk = 8;
fun = vectorize('1/(3+x(:,1)+x(:,3)) + x(:,2) + cos(x(:,4) + x(:,5))');
fun = str2func(['@(x)' fun]);
x = FullTensorGrid(linspace(0,1,Nk).',d);
u = fun(array(x));
u = FullTensor(u,d,x.sz);

%% Higher Order SVD - Truncation in Tucker Format
tr = Truncator();
tr.tolerance = 1e-8;
ur = tr.hosvd(u);
fprintf('Error = %d\n',(norm(full(ur)-u)/norm(u)))
fprintf('Storage = %d\n',storage(ur))
fprintf('Dimension of spaces = [ %s ]\n\n',num2str(ur.space.dim))

%% Tree-based format
rangOfArity = [2,3];
T = DimensionTree.random(d,rangOfArity);
tr = Truncator();
tr.tolerance = 1e-8;
ur = tr.hsvd(u,T);

fprintf('Error = %d\n',(norm(full(ur)-u)/norm(u)))
fprintf('Storage = %d\n',storage(ur))
fprintf('Ranks = [ %s ]\n\n',num2str(ur.ranks))

%% Truncation in TT format
tr = Truncator();
tr.tolerance = 1e-8;
ur = tr.ttsvd(u)
s = singularValues(ur);

fprintf('\nError = %d\n',(norm(full(ur)-u)/norm(u)))
fprintf('Storage = %d\n',storage(ur))
fprintf('TT-rank = [ %s ]\n\n',num2str(ur.ranks))

%% Truncation of the tensor in Tucker format with TT core
tr = Truncator();
tr.tolerance = 1e-8/2;
ur = tr.hosvd(u)
ur.core = tr.ttsvd(ur.core);
fprintf('\nError = %d\n',(norm(full(ur)-u)/norm(u)))
fprintf('Storage = %d\n',storage(ur))
fprintf('Dimension of spaces = [ %s ]\n',num2str(ur.space.dim))
fprintf('TT-rank of the core = [ %s ]\n\n',num2str(ur.core.ranks))

%% TT-Truncation after permutation
rperm = randperm(d);
uperm = permute(u,rperm);

tr = Truncator();
tr.tolerance = 1e-10;
tr.maxRank = 200;
ur = tr.ttsvd(uperm);
disp('With permutation...')
fprintf('\tOrdering = [ %s ]\n',num2str(rperm))
fprintf('\tError = %d\n',(norm(full(ur)-uperm)/norm(uperm)))
fprintf('\tStorage = %d\n',storage(ur))
fprintf('\tRanks = [ %s ]\n\n',num2str(ur.ranks))

%% TT tree optimization
disp('Tree optimization...')
[urperm,newperm,pelem] = optimizePermutation(ur,1e-10,10);
rperm = newperm;
fprintf('\tInitial storage = %d\n',storage(ur))
fprintf('\tNew ordering = [ %s ]\n',num2str(rperm))
fprintf('\tNumber of elementary permutations = %d\n',size(pelem,1))
fprintf('\tError = %d\n',(norm(full(urperm)-permute(uperm,newperm))/norm(u)))
fprintf('\tFinal storage = %d\n',storage(urperm))
fprintf('\tRanks = [ %s ]\n',num2str(urperm.ranks))