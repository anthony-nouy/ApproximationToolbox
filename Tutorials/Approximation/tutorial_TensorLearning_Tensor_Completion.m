% Tensor completion using tree-based tensor formats

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

%% Generation of the tensor to recover
sz = [5, 8, 4, 3, 6]; % Size of the tensor
d = length(sz); % Order of the tensor
T = DimensionTree.linear(d);
ranks = [1, 3, 3, 4, 3, 6, 3, 2, 4];
FTBT = TreeBasedTensor.randn(T, ranks, sz);
F = full(FTBT);

%% Samples generation
p = 0.25; % Proportion of known entries of the tensor

N = storage(F);
n = round(p*N);
loc = randperm(N, n).'; % Random selection of n entries
y = F.data(loc);
yTest = F.data(:);

fprintf('Tensor to recover with %i%% of its entries known (%i entries):\n ', p*100, n)
disp(FTBT)

%% Features creation: matrices containing 1 if the relative entry is known
indices = MultiIndices.ind2sub(sz, loc);
H = arrayfun(@(s) sparse(n, s), sz, 'UniformOutput', false);
for i = 1:d
    ind = sub2ind(size(H{i}), (1:n).', indices.array(:, i));
    H{i}(ind) = 1*sqrt(sz(i)); % To obtain orthonormal bases
end

indicesTest = MultiIndices.ind2sub(sz, 1:N);
HTest = arrayfun(@(s) sparse(N, s), sz, 'UniformOutput', false);
for i = 1:d
    ind = sub2ind(size(HTest{i}), (1:N).', indicesTest.array(:, i));
    HTest{i}(ind) = 1*sqrt(sz(i)); % To obtain orthonormal bases
end

%% Tree-based tensor format
% Tensor format
% 1 - Random tree and active nodes
% 2 - Tensor-Train
% 3 - Hierarchial Tensor-Train
% 4 - Binary tree

c = 4;
switch c
    case 1
        fprintf('Random tree with active nodes\n\n')
        arity = [2, 4];
        tree = DimensionTree.random(d, arity);
        isActiveNode = true(1,tree.nbNodes);
        s = TreeBasedTensorLearning(tree,isActiveNode, SquareLossFunction);
    case 2
        fprintf('Tensor-train format\n\n')
        s = TreeBasedTensorLearning.TensorTrain(d, SquareLossFunction);
    case 3
        fprintf('Tensor Train Tucker format\n\n')
        s = TreeBasedTensorLearning.TensorTrainTucker(d, SquareLossFunction);
        isActiveNode = s.isActiveNode;
        tree = s.tree;
    case 4
        fprintf('Binary tree\n\n')
        tree = DimensionTree.balanced(d);
        isActiveNode = true(1, tree.nbNodes);
        s = TreeBasedTensorLearning(tree, isActiveNode, SquareLossFunction);
    otherwise
        error('Not implemented.')
end

%% Initial guess: known entries in a rank-1 tree-based tensor
guess = FullTensor.zeros(sz);
guess.data(loc) = y / sqrt(prod(sz));
tr = Truncator('tolerance', eps, 'maxRank', 1);
guess = tr.hsvd(guess, tree);

%% Computation of the approximation
s.basesEval = H;
s.trainingData = {[], y};

s.tolerance.onStagnation = 1e-15;
s.tolerance.onError = 1e-15;

s.initializationType = 'initialGuess';
s.initialGuess = guess;

s.testError = true;
s.testData = {[], yTest};
s.basesEvalTest = HTest;

s.rankAdaptation = true;
% High limit of rank adaptation iterations to try to recover the tensor
s.rankAdaptationOptions.maxIterations = 100;
s.rankAdaptationOptions.theta = 0.8;

s.treeAdaptation = true;
s.treeAdaptationOptions.maxIterations = 1e2;

s.errorEstimation = true;

s.alternatingMinimizationParameters.maxIterations = 30;
s.alternatingMinimizationParameters.stagnation = 1e-10;

s.display = true;
s.alternatingMinimizationParameters.display = false;

s.rankAdaptationOptions.earlyStopping = true;
s.rankAdaptationOptions.earlyStoppingFactor = 10;

s.modelSelection = true;
s.modelSelectionOptions.type = 'testError';

tic
[GTBT, output] = s.solve();
toc

G = full(GTBT.tensor)*sqrt(prod(sz)); % Obtained approximation

%% Displays
testError = norm(G.data(:) - F.data(:))/norm(F.data(:));
fprintf('\nRanks: [%s  ]\n', sprintf('  %i', GTBT.tensor.ranks))
fprintf('Test error = %d\n\n', testError);

figure
subplot(1,2,1)
plot(FTBT, FTBT.ranks)
title('Ranks of the tensor to recover')
subplot(1,2,2)
plot(GTBT.tensor, GTBT.tensor.ranks)
title('Ranks ot the approximation')