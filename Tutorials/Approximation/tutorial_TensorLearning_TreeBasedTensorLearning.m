% Tree-based tensor approximation of multivariate functions using least-squares

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
choice = 2;
switch choice
    case 1
        d = 5;
        fun = vectorize('1/(10+x1+.5*x2)^2');
        X = RandomVector(NormalRandomVariable(),d);
    case 2
        [fun,X] = multivariateFunctionsBenchmarks('borehole');
        d = numel(X);
end

fun = UserDefinedFunction(fun,d);
fun.evaluationAtMultiplePoints = true;

%% Approximation basis
pdegree = 8;
orthonormalBases = true;
if orthonormalBases
    h = cellfun(@(x) PolynomialFunctionalBasis(orthonormalPolynomials(x),0:pdegree),X.randomVariables,'UniformOutput',false);
else
    h = cellfun(@(x) PolynomialFunctionalBasis(CanonicalPolynomials(x),0:pdegree),X.randomVariables,'UniformOutput',false);
end
H = FunctionalBases(h);

%% Training and test samples
x = random(X,100);
y = fun(x);
xTest = random(X,10000);
yTest = fun(xTest);

%% Tree-based tensor format
% Tensor format
% 1 - Random tree and active nodes
% 2 - Tensor-Train
% 3 - Hierarchial Tensor-Train
% 4 - Binary tree

c = 1;
switch c
    case 1
        fprintf('Random tree with active nodes\n\n')
        arity = [2 4];
        tree = DimensionTree.random(d,arity);
        isActiveNode = true(1,tree.nbNodes);
        s = TreeBasedTensorLearning(tree,isActiveNode,SquareLossFunction);
    case 2
        fprintf('Tensor-train format\n\n')
        s = TreeBasedTensorLearning.TensorTrain(d,SquareLossFunction);
    case 3
        fprintf('Tensor Train Tucker format\n\n')
        s = TreeBasedTensorLearning.TensorTrainTucker(d,SquareLossFunction);
        isActiveNode = s.isActiveNode;
    case 4
        fprintf('Binary tree\n\n')
        tree = DimensionTree.balanced(d);
        isActiveNode = true(1,tree.nbNodes);
        s = TreeBasedTensorLearning(tree,isActiveNode,SquareLossFunction);
    otherwise
        error('Not implemented.')
end

%% Random shuffling of the dimensions associated to the leaves
randomize = true;
if randomize
    s.tree.dim2ind = s.tree.dim2ind(randperm(d));
    s.tree = updateDimsFromLeaves(s.tree);
end

%% Computation of the approximation
s.bases = H;
% s.basesAdaptationPath = adaptationPath(H);
s.basesEval = H.eval(x);
s.trainingData = {[], y};

s.tolerance.onStagnation = 1e-6;
s.tolerance.onError = 1e-6;

s.initializationType = 'canonical';

s.linearModelLearning.basisAdaptation = true;
s.linearModelLearning.regularization = false;
s.linearModelLearning.errorEstimation = true;

s.testError = true;
s.testData = {xTest, yTest};
% s.basesEvalTest = H.eval(xTest);

s.rankAdaptation = true;
s.rankAdaptationOptions.maxIterations = 20;
s.rankAdaptationOptions.theta = 0.8;

s.treeAdaptation = true;
s.treeAdaptationOptions.maxIterations = 1e2;
% s.treeAdaptationOptions.forceRankAdaptation = true;

s.alternatingMinimizationParameters.maxIterations = 50;
s.alternatingMinimizationParameters.stagnation = 1e-10;

s.display = true;
s.alternatingMinimizationParameters.display = false;

s.rankAdaptationOptions.earlyStopping = true;
s.rankAdaptationOptions.earlyStoppingFactor = 10;

tic
[f, output] = s.solve();
if s.rankAdaptation && isfield(output, 'testErrorIterations')
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
fprintf('Test error = %d\n\n',errtest);

figure
subplot(1,3,1)
plot(f.tensor)
title('Active nodes')
subplot(1,3,2)
plotDims(f.tensor.tree)
title('Dimensions associated to the leaf nodes')
subplot(1,3,3)
plotWithLabelsAtNodes(f.tensor.tree,representationRank(f.tensor))
title('Representation ranks')