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

%% 1D discretization
L = 1;
n = 50;
h = L/(n+1);

% Matrix corresponding to 1D laplacian operator discretized with a FD scheme
K = 1/h^2 * gallery('tridiag',n);
I = speye(n);
f = ones(n,1);

%% UPDATES: CANONICAL CASE
d = 4;
factory = LaplacianLikeOperatorFactory(K,I,d);

A = factory.makeCanonicalTensor();
b = TuckerLikeTensor.ones(n*ones(d,1));

gs = GreedyLinearSolver(A,b);
su = TuckerLikeTensorALSLinearSolver(A,b);
gs.update = @(x) su.updateCore(x);
u1 = gs.solve();
su.maxIterations = 1;
su.maxIterationsTSpace = 1;
gs.update = @(x) su.solve(x); % Update both core and tspace
u2 = gs.solve();

disp(norm(A*u1-b)/norm(b))
disp(norm(A*u2-b)/norm(b))

%% UPDATES: TUCKER CASE
d = 4;
factory.order = d;
A = factory.makeTuckerTensor();
b = TuckerLikeTensor.ones(n*ones(d,1));
b = TuckerLikeTensor(full(b.core),b.space);
gs = GreedyLinearSolver(A,b,'maxIterations',10);
u3 = gs.solve();
su = TuckerLikeTensorALSLinearSolver(A,b);
gs.update = @(x) su.updateCore(x);
u4 = gs.solve();
su.maxIterations = 1;
su.maxIterationsTSpace = 1;
gs.update = @(x) su.solve(x);
u5 = gs.solve();

%% UPDATES: TREE BASED CORE
Ah = factory.makeTreeBasedTensor();
b = TuckerLikeTensor.ones(n*ones(d,1));
b = TuckerLikeTensor(treeBasedTensor(b.core,Ah.core.tree),b.space);
gs = GreedyLinearSolver(Ah,b,'maxIterations',10);
su = TuckerLikeTensorALSLinearSolver(Ah,b);
su.maxIterationsCore = 3;
gs.update = @(x) su.updateCore(x);
uh = gs.solve();