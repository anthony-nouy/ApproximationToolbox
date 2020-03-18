% FunctionalBasis, Projection, Interpolation, Least-Squares Approximation

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
warning('off','MATLAB:fplot:NotVectorized') % For display purposes

%% Interpolation on a polynomial space using Chebychev Points

% Function to approximate
fun = @(x) cos(10*x);
fun = UserDefinedFunction(fun,1);
fun.evaluationAtMultiplePoints = true;
rv = UniformRandomVariable(-1,1);

% Interpolation basis and points
p = 30;
h = PolynomialFunctionalBasis(orthonormalPolynomials(rv),0:p);
xcp = chebychevPoints(cardinal(h),[-1,1]);

% Interpolation of the function
f = h.interpolate(fun,xcp);

% Displays and error
fprintf('\nInterpolation on a polynomial space using Chebychev Points\n')
figure(1); clf;
fplot(@(x) [fun(x),f(x)],[-1,1])
legend('True function','Interpolation')
N = 100;
errL2 = testError(f,fun,N,rv);
fprintf('Mean squared error = %d\n',errL2)

%% Interpolation on a polynomial space using magic Points

% Function to approximate
fun = @(x) cos(10*x);
fun = UserDefinedFunction(fun,1);
fun.evaluationAtMultiplePoints = true;
rv = UniformRandomVariable(-1,1);

% Interpolation basis and points
p = 30;
h = PolynomialFunctionalBasis(orthonormalPolynomials(rv),0:p);
xmp = magicPoints(h,random(rv,10000));

% Interpolation of the function
f = h.interpolate(fun,xmp);

% Displays and error
fprintf('\nInterpolation on a polynomial space using magic Points\n')
figure(1); clf;
fplot(@(x) [fun(x),f(x)],[-1,1])
legend('True function','Interpolation')
N = 100;
errL2 = testError(f,fun,N,rv);
fprintf('Mean squared error = %d\n',errL2)

%% Projection on polynomial space through quadrature

% Function to approximate
fun = @(x) x.^2/2;
fun = UserDefinedFunction(fun,1);
fun.evaluationAtMultiplePoints = true;
rv = NormalRandomVariable;

% Integration rule
I = gaussIntegrationRule(rv,5);

% Approximation basis
p = 3;
h = PolynomialFunctionalBasis(orthonormalPolynomials(rv),0:p);

% Computation of the projection
f = h.projection(fun,I);

% Displays and error
fprintf('\nProjection on polynomial space through quadrature\n')
N = 100;
errL2 = testError(f,fun,N,rv);
fprintf('Mean squared error = %d\n',errL2)

%% Interpolation on UserDefinedFunctionalBasis

% Function to approximate
fun = @(x) exp(-(x-1/2).^2);
fun = UserDefinedFunction(fun,1);
fun.evaluationAtMultiplePoints = true;
rv = UniformRandomVariable(0,1);

% Fourier Basis
n = 15;
h = cell(1,2*n+1);
h{1} = @(x) ones(numel(x),1);
for i=1:n
    h{2*i} = @(x) sqrt(2)*cos(2*pi*i*x);
    h{2*i+1} = @(x) sqrt(2)*sin(2*pi*i*x);
end
h = UserDefinedFunctionalBasis(h,rv);

% Display of the Fourier basis
figure(2);clf;
fplot(@(x) h.eval(x),[0,1])
title('Fourier basis')

% Computation of the interpolation
xmp = magicPoints(h,random(rv,10000));
f = h.interpolate(fun,xmp);

% Displays and error
fprintf('\nInterpolation on UserDefinedFunctionalBasis\n')
figure(1); clf;
fplot(@(x) [fun(x),f(x)],[0,1])
legend('True function','Interpolation')
N = 100;
errL2 = testError(f,fun,N,rv);
fprintf('Mean squared error = %d\n',errL2)

%% Interpolation with a radial basis

% Function to approximate
fun = @(x) exp(-x.^2);
fun = UserDefinedFunction(fun,1);
fun.evaluationAtMultiplePoints = true;
rv = UniformRandomVariable(-1,1);

% Radial basis
n = 20;
x = linspace(-1,1,n);
s = 10/n;
k = @(x,y) exp(-(x-y).^2/s^2);
h = cell(1,n);
for i=1:n
    h{i} = @(y) k(y,x(i));
end
h = UserDefinedFunctionalBasis(h,rv);

% Computation of the interpolation
xmp = magicPoints(h,random(rv,100000));
f = h.interpolate(fun,xmp);

% Displays and error
fprintf('\nInterpolation with a radial basis\n')
figure(1);clf;
fplot(@(x) [fun(x),f(x)],[-1,1])
legend('True function','Interpolation')
N = 100;
errL2 = testError(f,fun,N,rv);
fprintf('Mean squared error = %d\n',errL2)

%% Projection on polynomial space through quadrature

% Function to approximate
fun = @(x) x.^2/2;
fun = UserDefinedFunction(fun,1);
fun.evaluationAtMultiplePoints = true;
rv = NormalRandomVariable;

% Integration rule
I = gaussIntegrationRule(rv,5);

% Approximation basis
p = 3;
h = PolynomialFunctionalBasis(HermitePolynomials(),0:p);

% Computation of the approximation
f = h.projection(fun,I);

% Derivative and second derivative of f through projection
df = h.projection(@(x) f.evalDerivative(1,x),I);
ddf = h.projection(@(x) df.evalDerivative(1,x),I);

% Displays and error
fprintf('\nProjection on polynomial space through quadrature\n')
N = 100;
errL2 = testError(f,fun,N,rv);
fprintf('Mean squared error = %d\n',errL2)
figure(2);clf;
fplot(@(x) [f(x),df(x),ddf(x)],[-1,1]);
legend('f','df','ddf')

%% Least-squares approximation

% Function to approximate
fun = @(x) exp(-(x-1/2).^2);
fun = UserDefinedFunction(fun,1);
fun.evaluationAtMultiplePoints = true;
rv = UniformRandomVariable(0,1);

% Approximation basis: Hermite polynomials of maximal degree p
p = 10;
h = PolynomialFunctionalBasis(HermitePolynomials(),0:p);

% Solver
ls = LinearModelLearningSquareLoss;
ls.regularization = false;
ls.basisAdaptation = false;

% Training sample
x = random(rv,100);
y = fun(x);

% Computation of the approximation
ls.basis = h;
ls.trainingData = {x,y};
f = ls.solve();

% Displays and error
fprintf('\nLeast-squares approximation\n')
figure(1);clf;
fplot(@(x) [fun(x),f(x)],[-1,1])
legend('True function','Approximation')
xTest = random(rv,100);
fTest = f(xTest);
yTest = fun(xTest);
fprintf('Mean squared error = %d\n',norm(yTest-fTest)/norm(yTest))

warning('on','MATLAB:fplot:NotVectorized') % For display purposes