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


%% Identification of a bivariate function f(x1,x2) with a function g(i1,j1,...,id,jd,y1,y2)
% x1 and x2 are identified with (i_1,....,i_d,y1) and (j_1,....,j_d,y2) through a Tensorizer

dim = 2;
L = 12; % Resolution
b = 2; % Scaling factor

t = Tensorizer(b,L,dim);
t.orderingType = 2; % ordering of variables 
fun = UserDefinedFunction(vectorize('1./(1+x1+x2)'),dim);
fun.evaluationAtMultiplePoints = true;
tensorizedfun = t.tensorize(fun);
tensorizedfun.f.evaluationAtMultiplePoints = true;
pdegree = 2;

bases = tensorizedFunctionFunctionalBases(t,pdegree);

%% Approximation in Tree based format
fprintf('--- Approximation in Tree based format ---- \n')
FPCA = FunctionalTensorPrincipalComponentAnalysis();
FPCA.PCASamplingFactor = 20;
FPCA.bases = bases;
FPCA.display = false;
tree = DimensionTree.balanced(tensorizedfun.f.dim);

fprintf('\nPrescribed tolerance\n')
tol = 1e-12;
FPCA.tol = tol;
TPCA.maxRank = inf;
[f,output] = FPCA.TBApproximation(tensorizedfun,tree);
tensorizedfuntb = TensorizedFunction(f,t);

fprintf('Number of evaluations = %d\n',output.numberOfEvaluations);
fprintf('Storage = %d\n',storage(f));
fprintf('Ranks = \n[%s]\n',num2str(f.tensor.ranks));

xtest = rand(1000,dim);
fxtest = tensorizedfuntb(xtest);
ytest  = tensorizedfun(xtest);
err = norm(ytest-fxtest)/norm(ytest);
fprintf('Mean squared error = %d\n',err)
