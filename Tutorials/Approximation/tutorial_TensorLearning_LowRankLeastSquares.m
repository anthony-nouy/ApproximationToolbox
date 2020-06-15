% Low rank approximation of multivariate functions using least-squares

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
polynomials = orthonormalPolynomials(X);
h = cellfun(@(x) PolynomialFunctionalBasis(x,0:pdegree),polynomials,'UniformOutput',false);
H = FunctionalBases(h);

%% Training and test samples
x = random(X,1000);
y = fun(x);
xTest = random(X,10000);
yTest = fun(xTest);

%% Canonical format
s = CanonicalTensorLearning(d,SquareLossFunction);
s.rankAdaptation = true;
s.initializationType = 'mean';
s.tolerance.onError = 1e-6;
s.alternatingMinimizationParameters.stagnation = 1e-8;
s.alternatingMinimizationParameters.maxIterations = 100;
s.linearModelLearning.regularization = false;
s.linearModelLearning.basisAdaptation = true;
s.bases = H;
s.trainingData = {x,y};
s.display = true;
s.alternatingMinimizationParameters.display = false;
s.testError = true;
s.testData = {xTest,yTest};
s.alternatingMinimizationParameters.oneByOneFactor = false;
s.alternatingMinimizationParameters.innerLoops = 2;
s.alternatingMinimizationParameters.random = false;
s.rankAdaptationOptions.maxIterations = 20;
s.modelSelection = true;
s.modelSelectionOptions.type = 'testError';

[f,outputCanonical] = s.solve();

fxtest = f(xTest);
errtest = norm(yTest-fxtest)/norm(yTest);
fprintf('\nCanonical rank = %d, test error = %d\n\n',numel(f.tensor.core.data),errtest)

%% Tensor train format
s = TensorTrainTensorLearning(d,SquareLossFunction);
s.display = true;
s.rankAdaptation = true;
s.tolerance.onError = 1e-6;
s.alternatingMinimizationParameters.display = false;
s.alternatingMinimizationParameters.stagnation = 1e-8;
s.alternatingMinimizationParameters.maxIterations = 100;
s.linearModelLearning.regularization = false;
s.linearModelLearning.basisAdaptation = true;
s.bases = H;
s.trainingData = {x,y};
s.testError = true;
s.testData = {xTest,yTest};
s.basesEvalTest = H.eval(xTest);
s.rank = 1;
s.rankAdaptationOptions.truncateBefore = true;
s.rankAdaptationOptions.rankOneCorrection = true;
s.rankAdaptationOptions.maxIterations = 20;
s.treeAdaptation = true;
s.treeAdaptationOptions.maxIterations = 100;
s.treeAdaptationOptions.tolerance = 1e-8;
s.modelSelection = true;
s.modelSelectionOptions.type = 'testError';

[f,outputTensorTrain] = s.solve();

if ~isa(f,'FunctionalTensor')
    f = FunctionalTensor(f,H);
end

if isfield(outputTensorTrain,'treeAdaptationIterations')
    perm = outputTensorTrain.treeAdaptationIterations{outputTensorTrain.selectedModelNumber};
else
    perm = 1:f.tensor.order;
end

fxtest = f(xTest(:,perm));
errtest = norm(yTest-fxtest)/norm(yTest);
fprintf('\nTest error = %d\n',errtest);
fprintf('TT ranks = [ %s ]\n',num2str(f.tensor.ranks))
