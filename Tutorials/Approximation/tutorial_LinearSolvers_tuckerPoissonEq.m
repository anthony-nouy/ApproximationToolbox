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

%% Construction of the operator in Tucker format
d = 3;
As = cell(d,1);
fs = cell(d,1);
for mu = 1:d
    As{mu} = {K;I};
    fs{mu} = f;
end
As = TSpaceOperators(As);
Ac = FullTensor(zeros(2,2,2));
Ac.data(1,2,2) = 1;
Ac.data(2,1,2) = 1;
Ac.data(2,2,1) = 1;
A = TuckerLikeTensor(Ac,As);
% A = \sum_{i=1}^2 \sum_{j=1}^2 \sum_{k=1}^2 Ac.data(i,j,k) As{1}{i}
% \otimes As{2}{j} \otimes As{3}{k}.

bs = TSpaceVectors(fs);
bc = DiagonalTensor(1,d);
b = TuckerLikeTensor(bc,bs);
% b = fs \otimes fs \otimes fs.

% Another possibility is by using some facilities:
factory = LaplacianLikeOperatorFactory(K,I,d);
A = factory.makeTuckerTensor();
b = TuckerLikeTensor.ones(n*ones(d,1));

T = Truncator();

%% Construction of a rank-8 preconditioner by approximating A^{-1}
disp('PRECONDITIONER')
precond = GreedyInverseApproximator(A,'maxIterations',8);
P = precond.solve();

%% Solution of Ax=b using truncated Richardson iterations method
disp('TRUNCATED RICHARDSON ITERATIONS')
s = TruncatedRichardsonIterations(A,b,T,'P',P);
[uRichardson,output] = s.solve();

%% Solution of Ax=b using truncated preconditioned CG method
disp('TRUNCATED PCG')
s = TruncatedPCG(A,b,T,'P',P);
[uPCG,output] = s.solve();

%% Rank one approximation of the solution computed via an ALS
disp('RANK ONE APPROXIMATION')
s = RankOneALSLinearSolver(A,b);
[uR1,output] = s.solve();

%% Greedy approximation of the solution of Ax=b
disp('GREEDY RANK ONE APPROXIMATION')
s = GreedyLinearSolver(A,b);
[ugreedy,output] = s.solve();

%% Greedy approximation of the solution of Ax=b by minimizing
% \|Ax-b\|^2
disp('GREEDY RANK ONE APPROXIMATION, RESIDUAL MINIMIZATION')
locS = RankOneALSLinearSolver(A'*A,A'*b,'disp',false);
s = GreedyLinearSolver(A,b,'minimizeResidual',true,'localSolver',locS);
[ugreedy,output] = s.solve();

%% Greedy approximation of the solution of Ax=b but computes
% \|Ax-b\|/\|b\| every 5 iterations
disp(['GREEDY RANK ONE APPROXIMATION, COMPU§TATION OF THE RESIDUAL ' ...
    'EVERY 5 ITERATIONS'])
s = GreedyLinearSolver(A,b,'checkResidual',5);
[u5,output] = s.solve();