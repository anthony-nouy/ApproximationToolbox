% Tensorizer, TensorizedFucntion

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

%% Identification of a function f(x) with a function g(i1,...,id,y)
% x is identified with (i_1,....,i_d,y) through a Tensorizer

d = 6; % Resolution
b = 3; % Scaling factor

X = UniformRandomVariable(0,1);
Y = UniformRandomVariable(0,1);

t = Tensorizer(b,d,1,X,Y);
fun = @(x) sqrt(x);
tensorizedfun = t.tensorize(fun);
tensorizedfun.f.evaluationAtMultiplePoints = true;

pdegree=4;
p = PolynomialFunctionalBasis(orthonormalPolynomials(Y),0:pdegree);
bases = tensorizedFunctionFunctionalBases(t,p);

H = FullTensorProductFunctionalBasis(bases);
grids = magicPoints(bases,random(bases.measure,100));
G = FullTensorGrid(grids);

funinterp = H.tensorProductInterpolation(tensorizedfun,G);
tr = Truncator();
tr.tolerance = 1e-9;
funinterp.tensor = tr.ttsvd(funinterp.tensor);
tensorizedfuninterp = TensorizedFunction(funinterp,t);

xtest = random(X,1000);
fxtest = tensorizedfuninterp(xtest);
ytest  = fun(xtest);
errL2 = norm(ytest-fxtest)/norm(ytest);
fprintf('Mean squared error = %d\n',errL2)

%% Identification of a bivariate function f(x1,x2) with a function g(i1,...,id,j1,...,jd,y1,y2)
% x1 and x2 are identified with (i_1,....,i_d,y1) and (j_1,....,j_d,y2) through a Tensorizer

dim = 2;
d = 6; % Resolution
b = 2; % Scaling factor

t = Tensorizer(b,d,dim);
X = t.X;
Y = t.Y;
t.orderingType = 1;
fun = UserDefinedFunction(vectorize('1./(1+x1+x2)'),dim);
fun.evaluationAtMultiplePoints = true;
tensorizedfun = t.tensorize(fun);

pdegree = 1;
p = cellfun(@(Y) PolynomialFunctionalBasis(orthonormalPolynomials(Y),0:pdegree),Y.randomVariables,'uniformoutput',false);
p = FunctionalBases(p);

bases = tensorizedFunctionFunctionalBases(t,p);
H = FullTensorProductFunctionalBasis(bases);
grids = magicPoints(bases,random(bases.measure,100));
G = FullTensorGrid(grids);

funinterp = H.tensorProductInterpolation(tensorizedfun,G);
tr = Truncator();
tr.tolerance = 1e-9;
funinterp.tensor = tr.ttsvd(funinterp.tensor);
tensorizedfuninterp = TensorizedFunction(funinterp,t);
fprintf('Representation ranks = %s\n',num2str(representationRank(funinterp.tensor)))

xtest = random(X,1000);
fxtest = tensorizedfuninterp(xtest);
ytest  = fun(xtest);
errL2 = norm(ytest-fxtest)/norm(ytest);
fprintf('Mean squared error = %d\n',errL2)