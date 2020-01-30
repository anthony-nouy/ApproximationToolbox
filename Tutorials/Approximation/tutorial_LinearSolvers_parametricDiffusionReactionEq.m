% Solve - \Delta u + \xi u = 1 on [0,L]^2 with u = 0 on the boundary using a finite difference method, for \xi \in \Xi = \{\xi_1,\hdots,\xi_Q\} a collection of parameters.

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

d = 2; % Nb of dimensions of the problem, here we consider (x,xi)

%% Spatial dimension
L = 1;
N = 10; % number of dof per dimension
N2 = N^2;
h = L/(N+1);

Ax = (1/h^2)*gallery('poisson',N);
Bx = speye(N2);
fx = ones(N2,1);

%% Parametric dimension
% Xi is randomly distributed in [xim,xiM].
Q = 200;

xim = 0;
xiM = 50;
Xi = xim + (xiM - xim) * rand(Q,1);

Axi = speye(Q);
Bxi = spdiags(Xi,0,Q,Q);
fxi = ones(Q,1);

%% Tensor operator and RHS
As = TSpaceOperators({{Ax;Bx};{Axi;Bxi}});
Ac = DiagonalTensor([1;1],d);
A = TuckerLikeTensor(Ac,As);

fs = TSpaceVectors({fx;fxi});
fc = DiagonalTensor(1,d);
f = TuckerLikeTensor(fc,fs);

%% Mean-based preconditioner
xi = (xim+xiM)/2;
Pxi = ImplicitMatrix.luInverse(Ax + xi*Bx);
P = TuckerLikeTensor.eye([N2;Q]);
P.space.spaces{1}{1} = Pxi;

%% Truncator
t = Truncator('tolerance',1e-10);

%% Solution using the truncated Richardson algorithm
s = TruncatedRichardsonIterations(A,f,t,'P',P,'maxIterations',100);
uRichardson = s.solve();

%% Solution using a greedy algorithm
s = GreedyLinearSolver(A,f,'maxIterations',20,'tolerance',1e-13,'stagnation',1e-13);
uGreedy = s.solve();

%% Solution using the ALS with a greedy initial guess
s = GreedyLinearSolver(A,f,'maxIterations',2);
u0 = s.solve();
s = TuckerLikeTensorALSLinearSolver(A, f, 'maxIterations', 10, 'x0', u0);
uals = s.solve();

%% Post-processing of a sample
X = linspace(0,L,N+2);
[X,Y] = meshgrid(X,X);

evalSamp = zeros(Q,1);
q = randi(Q);
evalSamp(q) = 1;

uref = (Ax+Xi(q)*Bx)\fx;

%%
uSamp = timesVector(uGreedy,evalSamp,2);
uSamp = full(uSamp);

disp('Relative error w.r.t. the reference solution:')
disp(norm(uSamp.data-uref)/norm(uref))

usamp = zeros(N+2);
usamp(2:N+1,2:N+1) = reshape(uSamp.data,N,N); % Add the values of u on boundaries

figure(1)
clf
surf(X,Y,usamp)