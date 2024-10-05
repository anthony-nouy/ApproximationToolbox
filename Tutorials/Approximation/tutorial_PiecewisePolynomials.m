% PiecewisePolynomialFunctionalBasis

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

clc, clearvars, close all

%% Piecewise polynomial basis with given points and degrees

points = [0,.2,1];
p = [1,2];
H = PiecewisePolynomialFunctionalBasis(points, p);
xplot = linspace(0,1,1000)';
figure(1)
plot(xplot, H.eval(xplot))

figure(2)
plot(xplot, H.evalDerivative(1,xplot))

%% Piecewise polynomial basis with constant degree and mesh size
h = 2^(-2);
p = 1;
H = PiecewisePolynomialFunctionalBasis.hp(0, 1, h, p);
xplot = linspace(0,1,1000)';
figure(1)
clf
plot(xplot, H.eval(xplot))

%% Piecewise polynomial basis with constant degree and given number of elements
n = 2;
p = 4;
H = PiecewisePolynomialFunctionalBasis.np(0, 1, n, p);
xplot = linspace(0,1,1000)';
figure(1)
clf
plot(xplot, H.eval(xplot))


%% Singularity adapted Piecewise polynomial basis
h = 2^-2;
H = PiecewisePolynomialFunctionalBasis.singularityhpAdapted(0, 1, [0], h);
xplot = linspace(0,1,1000)';
figure(1)
clf
plot(xplot, H.eval(xplot))

%% Interpolation of a function 

H = PiecewisePolynomialFunctionalBasis.np(0, 1, 4, 3);
f = UserDefinedFunction('cos(4*pi*x1)',1)    ;
g = H.interpolationPoints();
If = H.interpolate(f,g);
xplot = linspace(0,1,1000)';
figure(1)
clf
plot(xplot, If.eval(xplot))
hold on
plot(xplot, f.eval(xplot));
legend('If','f')

X = UniformRandomVariable(0, 1);
ERR_L2 = f.testError(If, 1000, X);
fprintf('Mean squared error = %2.5e\n', ERR_L2)

g = H.magicPoints();
If = H.interpolate(f,g);
ERR_L2 = f.testError(If, 1000, X);
fprintf('Mean squared error (magic points) = %2.5e\n', ERR_L2)

%% Quadrature
H = PiecewisePolynomialFunctionalBasis.np(0, 1, 4, 2);
f = UserDefinedFunction('exp(x1)',1)    ;
g = H.gaussIntegrationRule(10);
I = g.integrate(f);
Iex = exp(1)-1;
fprintf('Integration error = %2.5e\n', abs(Iex-I)/abs(Iex))


%% Approximation of a bivariate function with singularity
d=2;
X = RandomVector(UniformRandomVariable(0,1),2);
f = vectorize('1/(x1+x2)^.25');
f = UserDefinedFunction(f,d);
f.measure = X;


singularityAdapted = true;

if ~singularityAdapted
    h = 2^(-5);
    p = 5;
    bases = PiecewisePolynomialFunctionalBasis.hp(0,1,h,p);
else
    h = 2^(-20); % mesh size near singularities
    bases = PiecewisePolynomialFunctionalBasis.singularityhpAdapted(0,1,0,h);
end
bases = FunctionalBases.duplicate(bases,d);


H = FullTensorProductFunctionalBasis(bases);
[~,g] = interpolationPoints(H);
g = FullTensorGrid(g);
If = H.tensorProductInterpolation(f,g);

[errL2,errLinf] = testError(f,If,100000);
fprintf('L2 error = %d\n',errL2)
fprintf('Linfty error = %d\n',errLinf)
fprintf('storage = %d\n',storage(If))

figure(1)
xg=array(g);
plot(xg(:,1),xg(:,2),'.')
title('interpolation points')

figure(2)
subplot(1,2,1)
surf(f,200,'edgecolor','none')
ax = axis;
ca=caxis;
title('The function')
subplot(1,2,2)
surf(If,200,'edgecolor','none')
axis(ax);
caxis(ca);
title('Its interpolation')
