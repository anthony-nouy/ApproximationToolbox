% Conditional expectations

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

%% Approximation of a function f(X)
d = 6;
fun = UserDefinedFunction(vectorize('x1 + x1*x2 +x1*x3^2 + x4^3 + x5 + x6'),d);
fun.evaluationAtMultiplePoints = true;

pdegree = 3;
X = RandomVector(NormalRandomVariable(0,1),d);
h = PolynomialFunctionalBasis(HermitePolynomials(),0:pdegree);
bases = FunctionalBases.duplicate(h,d);
g = FullTensorGrid(num2cell(random(X,1000),1));
H = FullTensorProductFunctionalBasis(bases);
f = H.tensorProductInterpolation(fun,g);

xtest = random(X,1000);
fxtest = f(xtest);
ytest  = fun(xtest);
err = norm(ytest-fxtest)/norm(ytest);
fprintf('Mean squared error = %d\n',err)

%% Conditional expectations
% E(f | X1) = 2*X1
dims = 1;
fce = conditionalExpectation(f,dims);
funce = UserDefinedFunction(vectorize('2*x1'),length(dims));
x = randn(10,length(dims));
err=norm(funce(x)-fce(x))/norm(funce(x));
fprintf('Dims %s : error = %d\n',num2str(dims),err)

% E(f | X1,X2) = 2*X1 + X1*X2
dims = [1,2];
fce = conditionalExpectation(f,dims);
funce = UserDefinedFunction(vectorize('2*x1+x1*x2'),length(dims));
x = randn(10,length(dims));
err=norm(funce(x)-fce(x))/norm(funce(x));
fprintf('Dims %s : error = %d\n',num2str(dims),err)

% E(f | X1,X3) = X1 + X1*X3^2
dims = [1,3];
fce = conditionalExpectation(f,dims);
funce = UserDefinedFunction(vectorize('x1+x1*x2^2'),length(dims));
x = randn(10,length(dims));
err=norm(funce(x)-fce(x))/norm(funce(x));
fprintf('Dims %s : error = %d\n',num2str(dims),err)

% E(f | X1,X4) = 2*X1 + X4^3
dims = [1,4];
fce = conditionalExpectation(f,dims);
funce = UserDefinedFunction(vectorize('2*x1+x2^3'),length(dims));
x = randn(10,length(dims));
err=norm(funce(x)-fce(x))/norm(funce(x));
fprintf('Dims %s : error = %d\n',num2str(dims),err)

% E(f | X1,X3,X4) = X1 + X1*X2^2 + X4^3
dims = [1,3,4];
fce = conditionalExpectation(f,dims);
funce = UserDefinedFunction(vectorize('x1+x1*x2^2+x3^3'),length(dims));
x = randn(10,length(dims));
err=norm(funce(x)-fce(x))/norm(funce(x));
fprintf('Dims %s : error = %d\n',num2str(dims),err)