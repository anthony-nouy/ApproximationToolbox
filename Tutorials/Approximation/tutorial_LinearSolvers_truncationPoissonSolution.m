% Solution of Poisson equation using finite difference method in arbitrary dimension

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

d = 10;

%% 1D discretization
L = 1;
n = 50;
h = L/(n+1);

% Matrix corresponding to 1D laplacian operator discretized with a FD scheme
K = 1/h^2 * gallery('tridiag',n);
I = speye(n);
f = ones(n,1);

%% RHS
b = TuckerLikeTensor.ones(n*ones(d,1));

%% Operators
factory = LaplacianLikeOperatorFactory(K,I,d);
A = factory.makeCanonicalTensor();

%%
gs = GreedyLinearSolver(A,b,'maxIterations',50);
u = gs.solve();

%%
t = Truncator('tolerance',1e-4);
ucan = t.truncate(u);

%%
utt = TuckerLikeTensor(TTTensor(u.core),...
               u.space);
uttt = t.truncate(utt);

%%
uh = TuckerLikeTensor(treeBasedTensor(u.core,DimensionTree.balanced(d)),...
              u.space);
uht = t.truncate(uh);
