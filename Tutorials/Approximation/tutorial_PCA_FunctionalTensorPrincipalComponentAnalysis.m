% Functional PCA for low rank approximation of multivariate functions

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

% Choice of the function to approximate
choice = 1;
switch choice
    case 1
        fprintf('Henoneiles\n')
        d = 5;
        [fun,X] = multivariateFunctionsBenchmarks('henonheiles',d);
        fun = UserDefinedFunction(fun,d);
        fun.evaluationAtMultiplePoints = true;
        pdegree = 4;
        bases = cellfun(@(x) PolynomialFunctionalBasis(x,0:pdegree), orthonormalPolynomials(X), 'UniformOutput',false);
        fun.store = true;
        fun.measure = X;
    case 2
        fprintf('Anisotropic function\n')
        d = 6;
        X = RandomVector(UniformRandomVariable(-1,1),d);
        fun = vectorize('1/(10 + 2*x1 + x3 + 2*x4 - x5)^2');
        fun = UserDefinedFunction(fun,d);
        fun.evaluationAtMultiplePoints = true;
        pdegree = 13;
        bases = cellfun(@(x) PolynomialFunctionalBasis(x,0:pdegree), orthonormalPolynomials(X), 'UniformOutput',false);
        fun.store = true;
        fun.measure = X;
    case 3
        d = 20;
        fprintf('Sinus of a sum\n')
        fun = vectorize('sin(x1+x2+x3+x4+x5+x6+x7+x8+x9+x10)');
        fun = UserDefinedFunction(fun,d);
        fun.evaluationAtMultiplePoints = true;
        X = RandomVector(UniformRandomVariable(-1,1),d);
        pdegree = 17;
        bases = cellfun(@(x) PolynomialFunctionalBasis(x,0:pdegree), orthonormalPolynomials(X), 'UniformOutput',false);
        fun.store = true;
        fun.measure = X;
    case 4
        fprintf('Composition of function\n')
        d = 10;
        X = RandomVector(UniformRandomVariable(-1,1),d);
        tree = DimensionTree.balanced(d);
        % fun = @(x1,x2) 0.5.*exp(-abs(x1+x2).^2);
        % fun = @(x1,x2) cos(x1 + x2);
        % fun = @(x1,x2)1/9.*(2 + x1.*x2).^(-2);
        fun = @(x1,x2) (1 + x1.^2+x2.^2).^(-1);
        fun = PoggioModelFunction(tree,fun,X);
        pdegree = 15;
        bases = cellfun(@(x) PolynomialFunctionalBasis(x,0:pdegree), orthonormalPolynomials(X), 'UniformOutput',false);
        fun.store = true;
        fun.measure = X;
    case 5
        fprintf('Borehole function\n')
        [fun,X] = multivariateFunctionsBenchmarks('borehole');
        d = numel(X);
        fun = UserDefinedFunction(fun,d);
        fun.evaluationAtMultiplePoints = true;
        pdegree = 14;
        bases = cellfun(@(x) PolynomialFunctionalBasis(x,0:pdegree), orthonormalPolynomials(X), 'UniformOutput',false);
        fun.store = true;
        fun.measure = X;
    case 6
        d = 20;
        fun = @(x) 1./(1+sum(abs(x).^2,2));
        fun = UserDefinedFunction(fun,d);
        fun.evaluationAtMultiplePoints = true;
        
        pdegree = 20;
        h = PolynomialFunctionalBasis(LegendrePolynomials(),0:pdegree);
        bases = FunctionalBases.duplicate(h,d);
        X = RandomVector(UniformRandomVariable(-1,1),d);
        
    case 7
        
        r = 4; % Resolution
        b = 5; % Scaling factor
        d = r+1;
        
        X = UniformRandomVariable(0,1);
        Y = UniformRandomVariable(0,1);
        ifun = @(x) sqrt(x);
        t = Tensorizer(b,r,1,X,Y);
        fun = t.tensorize(ifun);
        fun.f.evaluationAtMultiplePoints = true;
        pdegree = 5;
        h = PolynomialFunctionalBasis(orthonormalPolynomials(Y),0:pdegree);
        bases = tensorizedFunctionFunctionalBases(t,h);
        X = randomVector(bases.measure);
        
        
    case 8
        d = 5;
        funp = vectorize('cos(x1)+1./(1+x2.^2+x3^4)+x3');
        funp = UserDefinedFunction(funp,d);
        n = 100;
        grid=linspace(0,1,n)';
        fun = @(i) funp(reshape(grid(i),size(i)));
        fun = UserDefinedFunction(fun,d);
        fun.evaluationAtMultiplePoints = true;
        X = RandomVector(DiscreteRandomVariable((1:n)'),d);
        %basis=PolynomialFunctionalBasis(DiscretePolynomials(DiscreteRandomVariable((0:n-1)')),0:n-1);
        basis=DeltaFunctionalBasis((1:n)');
        bases = FunctionalBases.duplicate(basis,d);
end


%% HOPCA (PCA for each dimension, provides reduced spaces)
% PCA for each dimension to get principal subspaces
fprintf('--- Higher order PCA ---- \n')
tol = 1e-10;
FPCA = FunctionalTensorPrincipalComponentAnalysis();
FPCA.PCASamplingFactor = 1;
FPCA.tol = tol;
[subbases,outputs] = FPCA.hopca(fun,X,bases);
fprintf('Number of evaluations = [%s]\n',num2str(cellfun(@(x) x.numberOfEvaluations,outputs)));
fprintf('Ranks {1,...,d} = [%s]\n',num2str(cellfun(@(x) cardinal(x),subbases)));

%% Approximation in Tucker Format
% PCA for each dimension to get principal subspaces
% and interpolation on the tensor product of principal subspaces
fprintf('--- Approximation in Tucker format ---- \n')
tol = 1e-10;
%tol = [2 3 4 5 2 2 2 3 4 5];
FPCA = FunctionalTensorPrincipalComponentAnalysis();
FPCA.PCASamplingFactor = 1;
FPCA.sparse = false;
FPCA.tol = tol;
[f,output] = FPCA.TuckerApproximation(fun,X,bases);

fprintf('Number of evaluations = %d\n',output.numberOfEvaluations);
xtest = random(X,10000);
fxtest = f(xtest);
ytest  = fun(xtest);
fprintf('Error = %d\n',norm(ytest-fxtest)/norm(ytest))

%% Approximation in Tree based format
fprintf('--- Approximation in Tree based format ---- \n')
tol = 1e-3;
FPCA = FunctionalTensorPrincipalComponentAnalysis();
FPCA.projectionType = 'interpolation';
FPCA.PCASamplingFactor = 1;
FPCA.tol = tol;
tree = DimensionTree.balanced(d);
[f,output] = FPCA.TBApproximation(fun,X,bases,tree);

fprintf('Number of evaluations = %d\n',output.numberOfEvaluations);
fprintf('Storage = %d\n',storage(f));
fprintf('Ranks {1,1:2,1:3,...,1:d-1} = [%s]\n',num2str(f.tensor.ranks));

xtest = random(X,1000);
fxtest = f(xtest);
ytest  = fun(xtest);
err = norm(ytest-fxtest)/norm(ytest);
fprintf('Mean squared error = %d\n',err)


%% Approximation in Tensor Train Tucker format
fprintf('--- Approximation in Tensor Train Tucker format ---- \n')
tol = 1e-12;
FPCA = FunctionalTensorPrincipalComponentAnalysis();
FPCA.projectionType = 'interpolation';
FPCA.PCASamplingFactor = 3;
FPCA.tol = tol;
[f,output] = FPCA.TTTuckerApproximation(fun,X,bases);
fprintf('Number of evaluations = %d\n',output.numberOfEvaluations);
fprintf('Storage = %d\n',storage(f));
fprintf('Ranks {1,2,...,d} = [%s]\n',num2str([f.tensor.ranks(1),cellfun(@(x) cardinal(x), f.bases.bases(2:end) )']));
fprintf('Ranks {1:2,1:3,...,1:d-1} = [%s]\n',num2str(f.tensor.ranks(2:end)));

xtest = random(X,1000);
fxtest = f(xtest);
ytest  = fun(xtest);
err = norm(ytest-fxtest)/norm(ytest);
fprintf('Mean squared error = %d\n',err)

%% Approximation in Tensor Train format
fprintf('--- Approximation in Tensor Train format ---- \n')
tol = 1e-8;
FPCA = FunctionalTensorPrincipalComponentAnalysis();
FPCA.projectionType = 'interpolation';
FPCA.PCASamplingFactor = 3;
FPCA.tol = tol;
[f,output] = FPCA.TTApproximation(fun,X,bases);
fprintf('Number of evaluations = %d\n',output.numberOfEvaluations);
fprintf('Storage = %d\n',storage(f));
fprintf('Ranks {1,1:2,1:3,...,1:d-1} = [%s]\n',num2str(f.tensor.ranks));

xtest = random(X,1000);
fxtest = f(xtest);
ytest  = fun(xtest);
err = norm(ytest-fxtest)/norm(ytest);
fprintf('Mean squared error = %d\n',err)

