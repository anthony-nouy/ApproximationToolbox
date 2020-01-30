% Third numerical experiment of the article
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
load data3 P

d = 8;
g = graph(diag(ones(d-1,1),1) + diag(ones(d-1,1),-1));
tensors = repmat({FullTensor(P)},1,d-1);
u = GraphTensor(g, tensors, d, size(P,1)*ones(1,d));
u = full(u);
u.data = u.data ./ size(P,1);
u.data = u.data .* prod(sqrt(size(P,1)*ones(1,d)));
XI = RandomVector(DiscreteRandomVariable((1:size(P,1)).'),d);

%% Approximation basis
H = FunctionalBases.duplicate(OrthonormalDeltaFunctionalBasis((1:size(P,1)).'),d);

%% Sample generation via rejection sampling
N = 1e5;
NTest = 1e5;

x = zeros(N,d);
x(:,1) = randi(size(P,1),N,1); % Initial test
for i = 2:d
    dist = P(x(:,i-1),:);
    cumdist = cumsum(dist,2);
    r = rand(N,1);
    x(:,i) = size(P,1) + 1 - sum(cumdist>r,2);
end

xTest = zeros(NTest,d);
xTest(:,1) = randi(size(P,1),NTest,1); % Initial test
for i = 2:d
    dist = P(xTest(:,i-1),:);
    cumdist = cumsum(dist,2);
    r = rand(NTest,1);
    xTest(:,i) = size(P,1) + 1 - sum(cumdist>r,2);
end

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
errtest = norm(u-full(f.tensor))/norm(u);
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