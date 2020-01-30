% Least-squares projection with sparse tensor product functional basis

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

%% UserDefinedFunction
d = 2;
fun = vectorize('x1+x2^2*x1');
fun = UserDefinedFunction(fun,d);
X1 = UniformRandomVariable(-1,1);
X = RandomVector(X1,2);

%% Sparse tensor functional basis
p = 1;
m = 3;
h = PolynomialFunctionalBasis(CanonicalPolynomials(X1),0:5);
H = FunctionalBases.duplicate(h,d);
I = MultiIndices.withBoundedNorm(d,p,m);
Psi = SparseTensorProductFunctionalBasis(H,I);

figure(1)
plot(I)
title('Multi-Indices')

%% Least-squares projection
ls = LinearModelLearningSquareLoss;
ls.regularization = false;
ls.basisAdaptation = false;

structuredGrid = false;
switch structuredGrid
    case true
        g = SparseTensorGrid(linspace(-1,1,m+1),I+1,d);
        x = array(g);
    case false
        n = ceil(2*cardinal(Psi));
        x = random(X,n);
end
y = fun(x);
A = Psi.eval(x);
f = ls.solve(y,A);
f = FunctionalBasisArray(f,Psi);

fprintf('\n Least-Squares projection\n')
xtest = random(X,100);
ftest = f(xtest);
ytest = fun(xtest);
fprintf('\tError on training set = %d\n',norm(f(x)-y)/norm(y))
fprintf('\tError on test set     = %d\n',norm(ytest-ftest)/norm(ytest))

%% Adaptive sparse approximation using least-squares
d = 4;
p = 15;
fun = vectorize('1/(10+x1+.5*x2)^2');
fun = UserDefinedFunction(fun,d);
% fun = vectorize('[1/(10+x1+.5*x2)^2,x1+100*x2^2]');
% fun = UserDefinedFunction(fun,d,2);
fun.evaluationAtMultiplePoints = true;
h = PolynomialFunctionalBasis(HermitePolynomials(),0:p);
H = FunctionalBases.duplicate(h,d);
rv = getRandomVector(H);
% rv = RandomVector(NormalRandomVariable,d);

s = AdaptiveSparseTensorAlgorithm();
% s.nbSamples = 1;
% s.addSamplesFactor = 0.1;
s.tol = 1e-4;
s.tolStagnation = 5e-2;
% s.tolOverfit = 1.1;
% s.bulkParameter = 0.5;
% s.adaptiveSampling = true;
% s.adaptationRule = 'reducedmargin';
s.display = true;
s.displayIterations = true;
s.maxIndex = p;

ls = LinearModelLearningSquareLoss();
ls.regularization = true;
% ls.regularizationType = 'l1';
ls.errorEstimation = true;
% ls.errorEstimationType = 'leaveout';
% ls.errorEstimationOptions.correction = true;
% ls.solver = 'qr';

rng('default')
t = tic;
[f,err,~,y] = s.leastSquares(fun,H,ls,rv);
time = toc(t);

fprintf('Parametric dimension = %d\n',ndims(f.basis))% fprintf('parametric dimension = %d\n',numel(rv))
fprintf('Basis dimension = %d\n',cardinal(f.basis))
fprintf('Order = [ %s ]\n',num2str(max(f.basis.indices.array)))
% fprintf('Multi-index set = \n')
% disp(f.basis.indices.array)
fprintf('Nb samples = %d\n',size(y,1))
fprintf('CV error = %d\n',norm(err))
fprintf('Elapsed time = %f s\n',time);

figure(3)
clf
dim = 1:3;
plotMultiIndices(f,'dim',dim,'legend',false)

Ntest = 1000;
xtest = randn(Ntest,d);
ytest = fun(xtest);
fxtest = f(xtest);
errtest = norm(fxtest-ytest)/norm(ytest);
fprintf('Test error = %d\n',errtest)