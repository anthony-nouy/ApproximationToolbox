% Multivariate functions, Tensor Grids, Projection

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

%% Scalar-valued UserDefinedFunction
d = 3;
fun = vectorize('x1+x2+x3^4');
fun = UserDefinedFunction(fun,d);
fun.evaluationAtMultiplePoints = true;
x = rand(4,d);
fun.eval(x)
fun(x)

%% Vector-valued UserDefinedFunction
f = vectorize('[x1,100*x2]');
f = UserDefinedFunction(f,d,2);
f.evaluationAtMultiplePoints = false;
f(x)
f.evaluationAtMultiplePoints = true;
f(x)

f = vectorize('[x1,100*x2]');
f = UserDefinedFunction(f,d,[1 2]);
f.evaluationAtMultiplePoints = false;
f(x)
f.evaluationAtMultiplePoints = true;
f(x)

%% Evaluation on a FullTensorGrid
g = FullTensorGrid(linspace(-1,1,100)',d);
fx = fun.evalOnTensorGrid(g);

figure(1)
clf
plotGrid(g,'x')

%% Evaluation on a SparseTensorGrid
I = MultiIndices.withBoundedNorm(d,1,5)+1;
g = SparseTensorGrid(linspace(0,1,6)',I,d);
fx = fun.evalOnTensorGrid(g);

figure(1)
clf
plotGrid(g,'x')

%% Functional Bases
h = PolynomialFunctionalBasis(CanonicalPolynomials(),0:4);
H = FunctionalBases.duplicate(h,d);
grid = FullTensorGrid(linspace(-1,1,10)',d);
x = array(grid);
Hx = H.eval(x);

%% Sparse tensor functional basis
d = 2;
p = 1;
m = 4;
h = PolynomialFunctionalBasis(CanonicalPolynomials(),0:4);
H = FunctionalBases.duplicate(h,d);
I = MultiIndices.withBoundedNorm(d,p,m);
Psi = SparseTensorProductFunctionalBasis(H,I);

figure(1)
plot(I)
title('Multi-Indices')

finegrid = FullTensorGrid((-1:.1:1)',d);
x = array(finegrid);
Psix = Psi.eval(x);

figure(2)
clf
loc = sub2ind(I+1,[m+1,m+1]);
for i=1:cardinal(I)
    subplot(m+1,m+1,loc(i))
    plot(finegrid,Psix(:,i),'edgecolor','none')
    title([mat2str(I.array(i,:))])
end

%% Projection on polynomial space through quadrature
d = 3;
p = 3;
fun = vectorize('x1+x2^2+x3^3');
fun = UserDefinedFunction(fun,d);
fun.evaluationAtMultiplePoints = true;
v = NormalRandomVariable;
rv = RandomVector(v,d);
I = gaussIntegrationRule(v,5);
I = I.tensorize(d);
u = fun.evalOnTensorGrid(I.points);

h = PolynomialFunctionalBasis(HermitePolynomials(),0:p);
H = FunctionalBases.duplicate(h,d);
H = FullTensorProductFunctionalBasis(H);

f = H.projection(fun,I);

Ntest = 100;
xtest = random(rv,Ntest);
errtest = norm(f(xtest) - fun(xtest))/norm(fun(xtest));
fprintf('Test error = %d\n',errtest)