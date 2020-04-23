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
choice = 2;
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
        fprintf('Sinus of a sum\n')
        d = 20;
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
        fun = @(x1,x2) (1 + x1.^2+x2.^2).^(-1);
        fun = CompositionalModelFunction(tree,fun,X);
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
        fprintf('Tensorized function\n') 
        r = 15; % Resolution
        b = 2; % Scaling factor
        d = r+1;
        
        X = UniformRandomVariable(0,1);
        Y = UniformRandomVariable(0,1);
        ifun = @(x) 1./(1+x);
        t = Tensorizer(b,r,1,X,Y);
        fun = t.tensorize(ifun);
        fun.f.evaluationAtMultiplePoints = true;
        pdegree = 1;
        h = PolynomialFunctionalBasis(orthonormalPolynomials(Y),0:pdegree);
        bases = tensorizedFunctionFunctionalBases(t,h);
        X = randomVector(bases.measure);
        
        
    case 7
        fprintf('Algebraic tensor\n')
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
FPCA.bases = bases;
[subbases,outputs] = FPCA.hopca(fun);
fprintf('Number of evaluations = [%s]\n',num2str(cellfun(@(x) x.numberOfEvaluations,outputs)));
szs =cellfun(@(x) cardinal(x),subbases);
fprintf('Ranks {1,...,d} = [%s ]\n',num2str(szs));

%% Approximation in Tucker Format
% PCA for each dimension to get principal subspaces
% and interpolation on the tensor product of principal subspaces
fprintf('--- Approximation in Tucker format ---- \n')

%tol = [2 3 4 5 2 2 2 3 4 5];
FPCA = FunctionalTensorPrincipalComponentAnalysis();
FPCA.PCASamplingFactor = 1;
FPCA.PCAAdaptiveSampling = 1;
FPCA.bases = bases;


% prescribed ranks
fprintf('\nPrescribed ranks\n')
FPCA.tol = inf;
FPCA.maxRank = randi(4,1,d);
[f,output] = FPCA.TuckerApproximation(fun);

fprintf('Number of evaluations = %d\n',output.numberOfEvaluations);
fprintf('Storage = %d\n',storage(f));
fprintf('Prescribed Tucker-rank =\n [%s]\n',num2str(FPCA.maxRank));
fprintf('Tucker-rank =\n[%s]\n',num2str(f.tensor.ranks(2:end-1)));

xtest = random(X,10000);
fxtest = f(xtest);
ytest  = fun(xtest);
fprintf('Error = %d\n',norm(ytest-fxtest)/norm(ytest))


% prescribed tolerance
fprintf('\nPrescribed tolerance\n')
tol = 1e-10;
FPCA.tol = tol;
FPCA.maxRank=inf;
[f,output] = FPCA.TuckerApproximation(fun);

fprintf('Number of evaluations = %d\n',output.numberOfEvaluations);
fprintf('Storage = %d\n',storage(f));
fprintf('Tucker rank =\n[%s]\n',num2str(f.tensor.ranks(2:end)));

xtest = random(X,10000);
fxtest = f(xtest);
ytest  = fun(xtest);
fprintf('Error = %d\n',norm(ytest-fxtest)/norm(ytest))



%% Approximation in Tree based format
fprintf('--- Approximation in Tree based format ---- \n')
FPCA = FunctionalTensorPrincipalComponentAnalysis();
FPCA.PCASamplingFactor = 20;
FPCA.bases = bases;
tree = DimensionTree.balanced(d);

% prescribed ranks
fprintf('\nPrescribed ranks\n')
FPCA.tol = inf;
FPCA.maxRank = randi(8,1,tree.nbNodes); FPCA.maxRank(tree.root)=1;
[f,output] = FPCA.TBApproximation(fun,tree);

fprintf('Number of evaluations = %d\n',output.numberOfEvaluations);
fprintf('Storage = %d\n',storage(f));
fprintf('Prescribed ranks = \n [%s]\n',num2str(FPCA.maxRank));
fprintf('Ranks = \n [%s]\n',num2str(f.tensor.ranks));

xtest = random(X,1000);
fxtest = f(xtest);
ytest  = fun(xtest);
err = norm(ytest-fxtest)/norm(ytest);
fprintf('Mean squared error = %d\n',err)

% prescribed tolerance
fprintf('\nPrescribed tolerance\n')
tol = 1e-10;
FPCA.tol = tol;
TPCA.maxRank = inf;
[f,output] = FPCA.TBApproximation(fun,tree);

fprintf('Number of evaluations = %d\n',output.numberOfEvaluations);
fprintf('Storage = %d\n',storage(f));
fprintf('Ranks = \n[%s]\n',num2str(f.tensor.ranks));

xtest = random(X,1000);
fxtest = f(xtest);
ytest  = fun(xtest);
err = norm(ytest-fxtest)/norm(ytest);
fprintf('Mean squared error = %d\n',err)


%% Approximation in Tensor Train format
fprintf('--- Approximation in Tensor Train format ---- \n')
FPCA = FunctionalTensorPrincipalComponentAnalysis();
FPCA.projectionType = 'interpolation';
FPCA.PCASamplingFactor = 3;
FPCA.PCAAdaptiveSampling = 1;
FPCA.bases = bases;

% prescribed ranks
fprintf('\nPrescribed ranks\n')
FPCA.tol = inf;
FPCA.maxRank = randi(8,1,d-1); 
[f,output] = FPCA.TTApproximation(fun);
fprintf('Number of evaluations = %d\n',output.numberOfEvaluations);
fprintf('Storage = %d\n',storage(f));
fprintf('Prescribed TT-ranks = \n [%s]\n',num2str(FPCA.maxRank));
ttranks = flip(f.tensor.ranks(f.tensor.isActiveNode));
fprintf('TT-ranks = \n [%s]\n',num2str(ttranks(1:end-1)));


xtest = random(X,1000);
fxtest = f(xtest);
ytest  = fun(xtest);
err = norm(ytest-fxtest)/norm(ytest);
fprintf('Mean squared error = %d\n',err)

% prescribed tolerance
fprintf('\nPrescribed tolerance\n')
tol = 1e-4;
FPCA.tol = tol;
FPCA.maxRank = inf;
[f,output] = FPCA.TTApproximation(fun);
fprintf('Number of evaluations = %d\n',output.numberOfEvaluations);
fprintf('Storage = %d\n',storage(f));
ttranks = flip(f.tensor.ranks(f.tensor.isActiveNode));
fprintf('TT-ranks = \n [%s]\n',num2str(ttranks(1:end-1)));

xtest = random(X,1000);
fxtest = f(xtest);
ytest  = fun(xtest);
err = norm(ytest-fxtest)/norm(ytest);
fprintf('Mean squared error = %d\n',err)
