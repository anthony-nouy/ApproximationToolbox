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

d = 6;

%% 1D discretization
L = 1;
n = 100;
h = L/(n+1);

% Matrix corresponding to 1D laplacian operator discretized with a FD scheme
K = 1/h^2 * gallery('tridiag',n);
I = speye(n);

% RHS
f = ones(n,1);
fs = cell(d,1);
for mu = 1:d
    fs{mu} = f;
end
b = TuckerLikeTensor(DiagonalTensor(1,d),TSpaceVectors(fs));

% Operators
factory = LaplacianLikeOperatorFactory(K,I,d);

Acan = factory.makeCanonicalTensor();
At = factory.makeTuckerTensor();
Att = factory.makeTTTuckerLikeTensor();
Att2 = factory.makeOperatorTTTensor();
Ah = factory.makeTreeBasedTensor();

%% Pure greedy
s = GreedyLinearSolver(Acan,b,'maxIterations',30);
[ucan,outputucan] = s.solve();
n = numel(ucan);
if 8*n < 4*1e9 % Approximately 4GB of memory
    ucan = full(ucan);
end

%%
solUp = TuckerLikeTensorALSLinearSolver(Acan,b);
s.update = @(x) solUp.updateCore(x);
[ucan0,outputucan0] = s.solve();

solUp.maxIterationsTSpace = 3;
s.update = @(x) solUp.updateTSpaceByALS(x);
[ucan1,outputucan1] = s.solve();

%%
btt = TuckerLikeTensor(TTTensor(b.core),b.space);
solUp = TuckerLikeTensorALSLinearSolver(Att,btt, 'maxIterationsCore',5);
s.update = @(x) solUp.updateCore(x);
[utt,outpututt] = s.solve();

%%
btt = TuckerLikeTensor(TTTensor(b.core),b.space);
solUp = TuckerLikeTensorALSLinearSolver(Att,btt,...
    'maxIterationsCore',8,'useDMRG',true);
s.update = @(x) solUp.updateCore(x);
[utt2,outpututt2] = s.solve();

%%
bh = TuckerLikeTensor(treeBasedTensor(b.core,Ah.core.tree),b.space);
s = GreedyLinearSolver(Ah,bh,'maxIterations',30,'tolerance',1e-10,'stagnation',1e-10);
solUp = TuckerLikeTensorALSLinearSolver(Ah,bh,'maxIterationsCore',5);
s.update = @(x) solUp.updateCore(x);
[uh,outputuh] = s.solve();

%%
s = TuckerLikeTensorALSLinearSolver(Acan, b,...
    'maxIterations', 5, ...
    'maxIterationsTspace', 2, ...
    'x0', ucan);
uals = s.solve();