% FunctionalBasis, Projection, Interpolation

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

%% Definitions
d = 3;
p = 10;
f = vectorize('1+x1+(2+x1)/(2+x2)+.04*(x3-x3^3)');

f = UserDefinedFunction(f,d);
f.evaluationAtMultiplePoints = true;
v = UniformRandomVariable(-1,1);
rv = RandomVector(v,d);
basis = PolynomialFunctionalBasis(LegendrePolynomials(),0:p);
bases = FunctionalBases.duplicate(basis,d);

%% Sparse tensor product functional basis
Ind = MultiIndices.withBoundedNorm(d,1,p);
H = SparseTensorProductFunctionalBasis(bases,Ind);

%% Interpolation on a magic grid
g = magicPoints(H,random(rv,1000));
If = H.interpolate(f,g);

xtest = random(rv,1000);
errtest = norm(f(xtest)-If(xtest))/norm(f(xtest));
fprintf('Test error = %d\n',errtest)

%% Interpolation on a structured magic grid
g = num2cell(random(rv,1000),1);
g = magicPoints(bases,g);
g = SparseTensorGrid(g,H.indices+1);
If = H.interpolate(f,array(g));

xtest = random(rv,1000);
errtest = norm(f(xtest)-If(xtest))/norm(f(xtest));
fprintf('Test error = %d\n',errtest)

%% Interpolation on a structured magic grid (alternative)
g = num2cell(random(rv,1000),1);
If = H.tensorProductInterpolation(f,g);
xtest = random(rv,100);
errtest = norm(f(xtest)-If(xtest))/norm(f(xtest));
fprintf('Test error = %d\n',errtest)

%% Adaptive interpolation
alg = AdaptiveSparseTensorAlgorithm();
alg.tol = 1e-7;
alg.adaptationRule = 'reducedMargin';
alg.maxIndex = [];
g = num2cell(random(rv,10000),1);
f.store = true; % if true, store the evaluations of the functions and avoid several evaluations at the same point
[If,output] = alg.interpolate(f,bases,g);
xtest = random(rv,100);
errtest = norm(f(xtest)-If(xtest))/norm(f(xtest));
fprintf('Number of evaluations = %d\n',output.numberOfEvaluations)
fprintf('Test error = %d\n',errtest)