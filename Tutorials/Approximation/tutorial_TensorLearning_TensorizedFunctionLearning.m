% Tree-based tensor approximation of tensorized functions using least-squares

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

%% Function to approximate: identification of a function f(x) with a function g(i1,...,id,y)
% See also tutorial_TensorizedFunction

r = 4; % Resolution
b = 5; % Scaling factor
d = r+1;

X = UniformRandomVariable(0,1);
Y = UniformRandomVariable(0,1);

choice = 1;
switch choice
    case 1
        fun = @(x) sin(10*pi*(2*x+0.5))./(4*x+1) + (2*x-0.5).^4;
    case 2
        fun = @(x) (sin(4*pi*x) + 0.2*cos(16*pi*x)).*(x<0.5) + (2*x-1).*(x>=0.5);
    otherwise
        error('Bad function choice.')
end

t = Tensorizer(b,r,1,X,Y);
tensorizedfun = t.tensorize(fun);
tensorizedfun.f.evaluationAtMultiplePoints = true;

%% Approximation basis
pdegree = 5;
h = PolynomialFunctionalBasis(orthonormalPolynomials(Y),0:pdegree);
H = tensorizedFunctionFunctionalBases(t,h);

%% Training and test samples
x = random(X,200);
y = fun(x);
x = t.map(x); % Identification of x with (i_1,...,i_d,y)

xTest = random(X,1000);
yTest = fun(xTest);
xTest = t.map(xTest);

%% Tree-based tensor format
% Tensor format
% 1 - Random tree and active nodes
% 2 - Tensor-Train
% 3 - Hierarchial Tensor-Train
% 4 - Binary tree

c = 3;
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
s.trainingData = {x,y};

s.tolerance.onStagnation = 1e-6;
s.tolerance.onError = 1e-10;

s.initializationType = 'canonical';

s.linearModelLearning.basisAdaptation = true;
s.linearModelLearning.regularization = false;
s.linearModelLearning.errorEstimation = true;

s.testError = true;
s.testData = {xTest,yTest};

s.rankAdaptation = true;
s.rankAdaptationOptions.maxIterations = 50;

s.treeAdaptation = true;
s.treeAdaptationOptions.maxIterations = 1e2;

s.alternatingMinimizationParameters.maxIterations = 50;
s.alternatingMinimizationParameters.stagnation = 1e-10;

s.display = true;
s.alternatingMinimizationParameters.display = false;

s.rankAdaptationOptions.earlyStopping = true;
s.rankAdaptationOptions.earlyStoppingFactor = 10;

s.modelSelection = true;
s.modelSelectionOptions.type = 'testError';

tic
warning off
[f, output] = s.solve();
warning on
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

figure
xlin = linspace(0,1,1e3);
plot(xlin,fun(xlin(:)))
hold all
plot(xlin,f(t.map(xlin(:))))
legend('True function','Approximation')