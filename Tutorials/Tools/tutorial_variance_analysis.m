% Variance analysis

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
d = 5;
fun = UserDefinedFunction(vectorize('x1 + x1*x2 + x3^3 + x5^2'),d);
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
ytest = fun(xtest);
err = norm(ytest-fxtest)/norm(ytest);
fprintf('Mean squared error = %d\n',err)

%% First order Sobol indices
S1 = SensitivityAnalysis.sobolIndices(f,(1:d)',d)

%% First order sensitivty indices
Ss1 = SensitivityAnalysis.sensitivityIndices(f,(1:d)',d)

%% Total sobol indices of order 1
St1 = SensitivityAnalysis.totalSobolIndices(f,(1:d)',d)

%% Closed Sobol indices (all)
Sc = SensitivityAnalysis.closedSobolIndices(f,powerSet(d))

%% Sobol Indices (all)
S = SensitivityAnalysis.sobolIndices(f,powerSet(d))
sum(S)

%% Total Sobol Indices (all)
St = SensitivityAnalysis.totalSobolIndices(f,powerSet(d))

%% Shapley indices
Sh = SensitivityAnalysis.shapleyIndices(f,(1:d)',d)
sum(Sh)