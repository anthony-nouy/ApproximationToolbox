% Tree-based tensor approximation of multivariate probability density functions using least-squares

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

clearvars, clc, close all

%% Function to approximate: multivariate gaussian distribution
d = 6; % Dimension

% Covariance matrix
S = [2 0 0.5 1 0 0.5 ; ...
    0 1 0 0 0.5 0 ; ...
    0.5 0 2 0 0 1 ; ...
    1 0 0 3 0 0 ; ...
    0 0.5 0 0 1 0 ; ...
    0.5 0 1 0 0 2];

% Reference measure
XI = RandomVector(arrayfun(@(x) UniformRandomVariable(-x,x),5*diag(S),'UniformOutput',false));

% Density, defined with respect to the reference measure
u = @(x) mvnpdf(x,zeros(1,d),S)*prod(2*5*diag(S));

%% Samples generation
N = 1e5; % Training sample size
NTest = 1e4; % Test sample size
NXi = 1e4; % Size of the sample used to compute the L2 error

x = mvnrnd(zeros(d,1),S,N);
xTest = mvnrnd(zeros(d,1),S,NTest);
xi = random(XI,NXi);

%% Approximation bases
p = 20; % Polynomial degree in each dimension
% Orthonormal bases in each dimension, with respect to the reference measure
polynomials = orthonormalPolynomials(XI);
h = cellfun(@(x) PolynomialFunctionalBasis(x,0:p),polynomials,'UniformOutput',false);
H = FunctionalBases(h);

%% Tree-based tensor learning parameters
tree = DimensionTree.linear(d);
isActiveNode = true(1,tree.nbNodes);
s = TreeBasedTensorLearning(tree,isActiveNode,DensityL2LossFunction);

randomize = true; % Random permutation of the leaves
if randomize
    s.tree.dim2ind = s.tree.dim2ind(randperm(d));
    s.tree = updateDimsFromLeaves(s.tree);
end

s.bases = H;
s.rank = 1;
s.initializationType = 'canonical';

s.display = true;
s.alternatingMinimizationParameters.display = false;

s.linearModelLearning.errorEstimation = true;
s.linearModelLearning.basisAdaptation = true;

s.testError = true;
s.testErrorData = xTest;

% In density estimation, the error is the risk, which is negative
s.tolerance.onError = -Inf;

s.rankAdaptation = true;
s.rankAdaptationOptions.maxIterations = 10;

s.treeAdaptation = true;
% For the tree adaptation in density estimation, a tolerance must be provided
s.treeAdaptationOptions.tolerance = 1e-6;

s.alternatingMinimizationParameters.maxIterations = 50;
s.alternatingMinimizationParameters.stagnation = 1e-6;

s.rankAdaptationOptions.earlyStopping = true;
% earlyStoppingFactor < 1 because we the risk is negative
s.rankAdaptationOptions.earlyStoppingFactor = 0.1;

%% Density estimation
[~, output] = s.solve([],x);
[risk,I] = min(output.testErrorIterations);
f = output.iterates{I}; % Model selection based on the risk estimation
err = norm(u(xi)-f(xi))/norm(u(xi)); % L2 relative error

fprintf('\nRisk leave-one-out estimation:       %d\n',output.errorIterations(I))
fprintf('Risk estimation using a test sample: %d\n',risk)
fprintf('L2 relative error estimation:         %d\n',err)