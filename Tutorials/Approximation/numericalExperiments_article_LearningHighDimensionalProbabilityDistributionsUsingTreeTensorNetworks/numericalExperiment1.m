% First numerical experiment of the article
% Grelier, E., Nouy, A., & Lebrun, R. (2019). Learning high-dimensional probability distributions using tree tensor networks. arXiv preprint arXiv:1912.07913.

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

%% Density to approximate
d = 6;
S = [2 0 0.5 1 0 0.5 ; 0 1 0 0 0.5 0 ; 0.5 0 2 0 0 1 ; 1 0 0 3 0 0 ; 0 0.5 0 0 1 0 ; 0.5 0 1 0 0 2];
u = @(x) mvnpdf(x,zeros(1,d),S)*prod(2*5*sqrt(diag(S))) ./ (mvncdf(5*sqrt(diag(S).'),zeros(1,d),S) - mvncdf(-5*sqrt(diag(S).'),zeros(1,d),S)) .* (all(x >= -5*sqrt(diag(S).'),2) & all(x <= 5*sqrt(diag(S).'),2));
XI = RandomVector(arrayfun(@(x) UniformRandomVariable(-x,x),5*sqrt(diag(S)),'UniformOutput',false));

%% Approximation basis
pdegree = 50;
h = cellfun(@(x) PolynomialFunctionalBasis(x,0:pdegree),orthonormalPolynomials(XI),'UniformOutput',false);
H = FunctionalBases(h);

%% Sample generation via rejection sampling
N = 1e5;
NTest = 1e5;
x = [];
while length(x) < N
    s = mvnrnd(zeros(1,d),S,N);
    x = [x ; s(all(s >= -5*sqrt(diag(S).'),2) & all(s <= 5*sqrt(diag(S).'),2),:)];
end
x = x(1:N,:);

xTest = [];
while length(xTest) < NTest
    s = mvnrnd(zeros(1,d),S,NTest);
    xTest = [xTest ; s(all(s >= -5*sqrt(diag(S).'),2) & all(s <= 5*sqrt(diag(S).'),2),:)];
end
xTest = xTest(1:NTest,:);

xi = random(XI,1e6);

%% Starting tree
tree = DimensionTree.linear(d);
tree.dim2ind = tree.dim2ind(randperm(d));
tree = updateDimsFromLeaves(tree);
isActiveNode = true(1,tree.nbNodes);

%% Computation of the approximation
s = TreeBasedTensorLearning(tree,isActiveNode,DensityL2LossFunction);
s.bases = H;

s.initializationType = 'canonical';

s.display = true;
s.alternatingMinimizationParameters.display = false;

s.linearModelLearning.basisAdaptation = true;

s.testError = true;
s.testErrorData = xTest;

s.tolerance.onError = -Inf;

s.rankAdaptation = true;
s.rankAdaptationOptions.maxIterations = 20;

s.treeAdaptation = true;
s.treeAdaptationOptions.tolerance = 1e-6;

s.alternatingMinimizationParameters.maxIterations = 50;
s.alternatingMinimizationParameters.stagnation = 1e-6;

s.rankAdaptationOptions.earlyStopping = true;
s.rankAdaptationOptions.earlyStoppingFactor = 1-1e-3;

tic
warning off
[f, output] = s.solve([],x);
warning on
if s.rankAdaptation % Model selection
    [~,i] = min(output.testErrorIterations);
    f = output.iterates{i};
    output.error = output.errorIterations(i);
    output.testError = output.testErrorIterations(i);
end
toc

%% Displays
errtest = norm(u(xi)-f(xi))/norm(u(xi));
fprintf('Ranks: [%s  ]\n',sprintf('  %i',f.tensor.ranks));
fprintf('Loo risk = %d\n',output.error);
fprintf('Test risk = %d\n',output.testError);
fprintf('Test error = %d\n',errtest);
fprintf('Storage complexity = %i\n\n',storage(f.tensor));

figure
subplot(1,2,1)
plotDims(f.tensor.tree)
title('Dimensions associated to the leaf nodes')
subplot(1,2,2)
plotWithLabelsAtNodes(f.tensor.tree,representationRank(f.tensor))
title('Representation ranks')