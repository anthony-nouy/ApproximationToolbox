% Mapping
%
% Tutorial file that shows how to use the variable mapping to change
% the dimension of a problem by performing a mapping from Rd to Rm, with m
% usually much lower than d. In this example, the mapping is known, and the
% objective is to approximate the cosine of a linear combination of d
% variables, by transforming the high dimension d problem to a dimension 1
% problem. The three possible ways of writing a mapping are presented.

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

d = 100; % Problem dimension
X = RandomVector(NormalRandomVariable(0,1/sqrt(d)),d);
omega = rand(d,1); % Vector used for the linear combination
N = 50; % Number of training points
Ntest = 1e3; % Number of test points
p = 10; % Maximum degree of the approximation basis

% Function to approximate: cos(omega^T x), the cosine of a linear
% combination of variables, which is a univariate function cos(z) with z =
% omega^T x
fun = @(x) cos(x*omega);
fun = UserDefinedFunction(fun,d);
fun.evaluationAtMultiplePoints = true;

% Training and test samples
x = X.random(N);
y = fun.eval(x);
xTest = X.random(Ntest);
yTest = fun.eval(xTest);

% Solver
ls = LinearModelLearningSquareLoss;
ls.regularization = false;
ls.basisAdaptation = false;

% Approximation basis
P = HermitePolynomials;
h = PolynomialFunctionalBasis(P,0:p);

%% Approximation with a ComponentWise mapping
mapping = Mapping('c',d,1,{@(x) x*omega});
H = FunctionalBasesWithMappedVariables({h},mapping,X);

% Computation of the approximation coefficients
A = H.eval(x);
ls.basisEval = A{1};
ls.trainingData = {[], y};
a = ls.solve();

% Evaluation of the approximation on the test points
Psi = FunctionalBasesWithMappedVariables({SubFunctionalBasis(h,a)},mapping,X);
yApprox = cell2mat(Psi.eval(xTest).');

fprintf('Test error with ComponentWise mapping:  %d\n',norm(yApprox - yTest)/norm(yTest))

%% Approximation with a MatrixForm mapping
mapping = Mapping('m',d,1,omega.',0);
H = FunctionalBasesWithMappedVariables({h},mapping,X);

% Computation of the approximation coefficients
A = H.eval(x);
ls.basisEval = A{1};
ls.trainingData = {[], y};
a = ls.solve();

% Evaluation of the approximation on the test points
Psi = FunctionalBasesWithMappedVariables({SubFunctionalBasis(h,a)},mapping,X);
yApprox = cell2mat(Psi.eval(xTest).');

fprintf('Test error with MatrixForm mapping:     %d\n',norm(yApprox - yTest)/norm(yTest))

%% Approximation with a Function mapping
H.mapping = Mapping('f',d,1,@(x) x*omega);
H = FunctionalBasesWithMappedVariables({h},mapping,X);

% Computation of the approximation coefficients
A = H.eval(x); 
ls.basisEval = A{1};
ls.trainingData = {[], y};
a = ls.solve();

% Evaluation of the approximation on the test points
Psi = FunctionalBasesWithMappedVariables({SubFunctionalBasis(h,a)},mapping,X);
yApprox = cell2mat(Psi.eval(xTest).');

fprintf('Test error with Function mapping:       %d\n',norm(yApprox - yTest)/norm(yTest))

%% Random sampling of the FunctionalBasesWithMappedVariables
% Performed by providing the measure of the input variables
H = FunctionalBasesWithMappedVariables({h},mapping,X);
[HRand, xRand] = random(H,100);