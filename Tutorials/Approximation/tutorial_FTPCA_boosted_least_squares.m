clearvars; clc; close all

% Choice of the function to approximate
choice = 4;
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
        pdegree = 7;
        bases = cellfun(@(x) PolynomialFunctionalBasis(x,0:pdegree), orthonormalPolynomials(X), 'UniformOutput',false);
        fun.store = true;
        fun.measure = X;
    case 3
        fprintf('Sinus of a sum\n')
        d = 10;
        fun = vectorize('sin(x1+x2+x3+x4+x5+x6+x7+x8+x9+x10)');
        fun = UserDefinedFunction(fun,d);
        fun.evaluationAtMultiplePoints = true;
        X = RandomVector(UniformRandomVariable(-1,1),d);
        pdegree = 10;
        bases = cellfun(@(x) PolynomialFunctionalBasis(x,0:pdegree), orthonormalPolynomials(X), 'UniformOutput',false);
        fun.store = true;
        fun.measure = X;
    case 4
        fprintf('Composition of functions\n')
        d = 6;
        X = RandomVector(UniformRandomVariable(-1,1),d);
        tree = DimensionTree.balanced(d);
        fun = @(x1,x2) (1 + x1.^2+x2.^2).^(-1);
        fun = CompositionalModelFunction(tree,fun,X);
        pdegree = 4;
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

tol = 1e-4;
tree = DimensionTree.balanced(d);

%% Approximation in Tree-Based format using interpolation
FTPCA = FunctionalTensorPCAInterpolation();
VPCA = VectorPCA();
VPCA.samplingFactor = 1;
VPCA.adaptiveSampling = true;
VPCA.testError = false;
FTPCA.tol = tol;
FTPCA.basisAdaptation = false; 
FTPCA.vpca = VPCA;
fprintf('---Approximation in Tree-based format using interpolation ---\n')


isActiveNode = TreeBasedTensor.randomIsActiveNode(tree);
isActiveNode(:)=true;
[f,output] = FTPCA.TBApproximation(fun,X,bases,tree,isActiveNode);
fprintf('number of evaluations = %d\n',output.numberOfEvaluations);
fprintf('storage = %d\n',storage(f));
fprintf('ranks = [%s]\n',num2str(f.tensor.ranks));
xtest = random(X,1000);
fxtest = f(xtest);
ytest  = fun(xtest);
err = norm(ytest-fxtest)/norm(ytest);
errinf = norm(ytest-fxtest,'inf')/norm(ytest,'inf');
fprintf('mean squared error = %d\n',err)



%% Approximation in Tree-Based format using Boosted Least-Squares
tic
FTPCA = FunctionalTensorPCALeastSquares();
FTPCA.optimalSampling = true;
FTPCA.optimalSamplingBoosted = true;
FTPCA.samplingMethod = 0;
VPCA = VectorPCA();
VPCA.samplingFactor = 1;
VPCA.adaptiveSampling = true;
VPCA.testError = false;
FTPCA.tol = tol;
FTPCA.basisAdaptation = false;
FTPCA.vpca = VPCA;
fprintf('---Approximation in Tree-based format using BLS with slice---')

isActiveNode = TreeBasedTensor.randomIsActiveNode(tree);
isActiveNode(:)=true;
[f,output] = FTPCA.TBApproximation(fun,X,bases,tree,isActiveNode);
fprintf('number of evaluations = %d\n',output.numberOfEvaluations);
fprintf('storage = %d\n',storage(f));
fprintf('ranks = [%s]\n',num2str(f.tensor.ranks));

xtest = random(X,1000);
fxtest = f(xtest);
ytest  = fun(xtest);
err = norm(ytest-fxtest)/norm(ytest);
errinf = norm(ytest-fxtest,'inf')/norm(ytest,'inf');
fprintf('mean squared error = %d\n',err)

toc

%% Approximation in Tree-Based format using Boosted Least-Squares
tic
FTPCA = FunctionalTensorPCALeastSquares();
FTPCA.optimalSampling = true;
FTPCA.optimalSamplingBoosted = true;
FTPCA.samplingMethod = 1;
VPCA = VectorPCA();
VPCA.samplingFactor = 1;
VPCA.adaptiveSampling = true;
VPCA.testError = false;
FTPCA.tol = tol;
FTPCA.basisAdaptation = false;
FTPCA.vpca = VPCA;
fprintf('---Approximation in Tree-based format using BLS with discrete ---')

isActiveNode = TreeBasedTensor.randomIsActiveNode(tree);
isActiveNode(:)=true;
[f,output] = FTPCA.TBApproximation(fun,X,bases,tree,isActiveNode);
fprintf('number of evaluations = %d\n',output.numberOfEvaluations);
fprintf('storage = %d\n',storage(f));
fprintf('ranks = [%s]\n',num2str(f.tensor.ranks));

xtest = random(X,1000);
fxtest = f(xtest);
ytest  = fun(xtest);
err = norm(ytest-fxtest)/norm(ytest);
errinf = norm(ytest-fxtest,'inf')/norm(ytest,'inf');
fprintf('mean squared error = %d\n',err)
toc
