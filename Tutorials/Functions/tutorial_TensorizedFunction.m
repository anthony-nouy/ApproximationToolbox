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

L = 10; % Resolution
b = 3; % Scaling factor

t = Tensorizer(b,L,1);
t.orderingType = 1;
fun = @(x) sqrt(x);
tensorizedfun = t.tensorize(fun);
tensorizedfun.f.evaluationAtMultiplePoints = true;

pdegree=3; % polynomial degree (with respect to y variable)
bases = tensorizedFunctionFunctionalBases(t,pdegree);
funinterp = bases.tensorProductInterpolation(tensorizedfun);
tensorizedfuninterp = TensorizedFunction(funinterp,t);

xtest = rand(1000,1);
fxtest = tensorizedfuninterp(xtest);
ytest  = fun(xtest);
errL2 = norm(ytest-fxtest)/norm(ytest);
fprintf('Mean squared error for the interpolation = %d\n',errL2)

tr = Truncator();
for k=1:9
    tr.tolerance = 10^(-k); 
    tensorizedfuntt = tensorizedfuninterp;
    tensorizedfuntt.f.tensor = tr.ttsvd(tensorizedfuninterp.f.tensor);
    fprintf('Representation ranks = %s\n',num2str(representationRank(tensorizedfuntt.f.tensor)))
    fprintf('Storage complexity = %s\n',num2str(storage(tensorizedfuntt.f.tensor)))
    
    xtest = rand(1000,1);
    fxtest = tensorizedfuntt(xtest);
    ytest  = fun(xtest);
    errL2 = norm(ytest-fxtest)/norm(ytest);
    fprintf('Mean squared error = %d\n',errL2)
end

%% Identification of a bivariate function f(x1,x2) with a function g(i1,j1,...,id,jd,y1,y2)
% x1 and x2 are identified with (i_1,....,i_d,y1) and (j_1,....,j_d,y2) through a Tensorizer

dim = 2;
L = 8; % Resolution
b = 2; % Scaling factor

t = Tensorizer(b,L,dim);
t.orderingType = 2; % ordering of variables 
fun = UserDefinedFunction(vectorize('1./(1+x1+x2)'),dim);
fun = UserDefinedFunction(vectorize('x1<x2'),dim);
fun.evaluationAtMultiplePoints = true;
tensorizedfun = t.tensorize(fun);
tensorizedfun.f.evaluationAtMultiplePoints = true;

bases = tensorizedFunctionFunctionalBases(t,pdegree);
funinterp = bases.tensorProductInterpolation(tensorizedfun);
tensorizedfuninterp = TensorizedFunction(funinterp,t);

xtest = rand(100,dim);
fxtest = tensorizedfuninterp(xtest);
ytest  = fun(xtest);
errL2 = norm(ytest-fxtest)/norm(ytest);
fprintf('Mean squared error for the interpolation = %d\n',errL2)

tr = Truncator();
for k=1:9
    tr.tolerance = 10^(-k); 
    tensorizedfuntt = tensorizedfuninterp;
    tensorizedfuntt.f.tensor = tr.ttsvd(tensorizedfuninterp.f.tensor);
    fprintf('Representation ranks = %s\n',num2str(representationRank(tensorizedfuntt.f.tensor)))
    fprintf('Storage complexity = %s\n',num2str(storage(tensorizedfuntt.f.tensor)))
    
    xtest = rand(1000,dim);
    fxtest = tensorizedfuntt(xtest);
    ytest  = fun(xtest);
    errL2 = norm(ytest-fxtest)/norm(ytest);
    fprintf('Mean squared error = %d\n',errL2)
end
