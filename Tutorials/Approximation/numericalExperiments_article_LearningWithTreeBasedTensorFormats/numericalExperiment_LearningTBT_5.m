% Fifth numerical experiment of the article
% Grelier, E., Nouy, A., & Chevreuil, M. (2018). Learning with tree-based 
% tensor formats. arXiv preprint arXiv:1811.04455.

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

%% Function to approximate
d = 8;
tree = DimensionTree.balanced(d);
X = RandomVector(UniformRandomVariable(),d);
fun = @(x1,x2) (2 + x1.*x2).^(2)/9;
fun = CompositionalModelFunction(tree,fun,X);
fun = UserDefinedFunction(fun,d);
fun.evaluationAtMultiplePoints = true;

%% Approximation basis
pdegree = 8;
h = cellfun(@(x) PolynomialFunctionalBasis(orthonormalPolynomials(x),0:pdegree),X.randomVariables,'UniformOutput',false);
H = FunctionalBases(h);

%% Training and test samples
n = 1e4;
x = random(X,n);
y = fun(x);
xTest = random(X,10000);
yTest = fun(xTest);

%% Starting tree
c = 1;
switch c
    case 1
        tree = DimensionTree.balanced(d);
    case 2
        tree = DimensionTree.linear(d);
    otherwise
        error('Not implemented.')
end
isActiveNode = true(1,tree.nbNodes);

%% Random shuffling of the dimensions associated to the leaves
randomize = true;
if randomize
    tree.dim2ind = tree.dim2ind(randperm(d));
    tree = updateDimsFromLeaves(tree);
end

%% Computation of the approximation
s = TreeBasedTensorLearning(tree,isActiveNode,SquareLossFunction);
s.bases = H;
s.trainingData = {x,y};

s.tolerance.onStagnation = 1e-15;
s.tolerance.onError = eps;

s.initializationType = 'canonical';

s.linearModelLearning.basisAdaptation = true;

s.testError = true;
s.testData = {xTest,yTest};

s.rankAdaptation = true;
s.rankAdaptationOptions.maxIterations = 30;

s.treeAdaptation = true;
s.treeAdaptationOptions.maxIterations = 1e2;

s.alternatingMinimizationParameters.maxIterations = 50;
s.alternatingMinimizationParameters.stagnation = 1e-10;

s.display = true;
s.alternatingMinimizationParameters.display = false;

s.rankAdaptationOptions.earlyStopping = true;
s.rankAdaptationOptions.earlyStoppingFactor = 10;

tic
warning off
[f, output] = s.solve();
warning on
if s.rankAdaptation % Model selection
    [~,i] = min(output.testErrorIterations);
    f = output.iterates{i};
    output.error = output.errorIterations(i);
    output.testError = output.testErrorIterations(i);
end
toc

%% Displays
errtest = norm(f(xTest) - yTest)/norm(yTest);
fprintf('Ranks: [%s  ]\n',sprintf('  %i',f.tensor.ranks))
fprintf('Loo error = %d\n',output.error);
fprintf('Test error = %d\n',errtest);
fprintf('Storage complexity = %i\n\n',storage(f.tensor));

figure
subplot(1,2,1)
plotDims(f.tensor.tree)
title('Dimensions associated to the leaf nodes')
subplot(1,2,2)
plotWithLabelsAtNodes(f.tensor.tree,representationRank(f.tensor))
title('Representation ranks')