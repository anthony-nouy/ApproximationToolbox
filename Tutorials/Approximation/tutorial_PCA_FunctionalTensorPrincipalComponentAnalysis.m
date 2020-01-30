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

%%
d = 5;
fun = vectorize('cos(x1)+1./(1+x2.^2+x3^4)+x3');
fun = @(x) 1./(1+sum(abs(x).^2,2));
fun = UserDefinedFunction(fun,d);
fun.evaluationAtMultiplePoints = true;

pdegree = 20;
h = PolynomialFunctionalBasis(LegendrePolynomials(),0:pdegree);
bases = FunctionalBases.duplicate(h,d);
X = RandomVector(UniformRandomVariable(-1,1),d);

%% HOPCA (PCA for each dimension, provides reduced spaces)
% PCA for each dimension to get principal subspaces
fprintf('--- Higher order PCA ---- \n')
tol = 1e-8;
FPCA = FunctionalTensorPrincipalComponentAnalysis();
FPCA.PCASamplingFactor = 1;
FPCA.tol = tol;
[subbases,outputs] = FPCA.hopca(fun,X,bases);
fprintf('Number of evaluations = [%s]\n',num2str(cellfun(@(x) x.numberOfEvaluations,outputs)));
fprintf('Nanks {1,...,d} = [%s]\n',num2str(cellfun(@(x) cardinal(x),subbases)));

%% Approximation in Tucker Format
% PCA for each dimension to get principal subspaces
% and interpolation on the tensor product of principal subspaces
fprintf('--- Approximation in Tucker format ---- \n')
tol = 1e-8;
tol = [2 3 4 5 2 2 2 3 4 5];
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

%% Approximation in Tensor Train Tucker format
fprintf('--- Approximation in Tensor Train Tucker format ---- \n')
tol = 1e-12;
tol = 4;
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
tol = 1e-3;
tol =10;
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

%% Approximation in Tree based format
fprintf('--- Approximation in Tree based format ---- \n')
tol = 1e-4;
tol = 6;
FPCA = FunctionalTensorPrincipalComponentAnalysis();
FPCA.projectionType = 'interpolation';
FPCA.PCASamplingFactor = 5;
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

%% Approximation in Tree-Based format using interpolation
tree = DimensionTree.random(d,[2,2]);
FPCA = FunctionalTensorPrincipalComponentAnalysis();
FPCA.projectionType = 'interpolation';
FPCA.tol = tol;
fprintf('---Approximation in Tree-based format using interpolation ---')

figure(1)
subplot(1,2,1)
plotDimsNodesIndices(tree)
H1 = title('Approximation in Tree-based format using interpolation');
set(H1,'fontsize',16)
axis off
storenbsamples = [];
storeprec = [];
for kk=1:10
    isActiveNode = TreeBasedTensor.randomIsActiveNode(tree);
    isActiveNode(:)=true;
    [f,output] = FPCA.TBApproximation(fun,X,bases,tree,isActiveNode);
    fprintf('Number of evaluations = %d\n',output.numberOfEvaluations);
    fprintf('Storage = %d\n',storage(f));
    fprintf('Ranks = [%s]\n',num2str(f.tensor.ranks));
    
    figure(1)
    subplot(1,2,2)
    plot(f.tensor,representationRank(f.tensor));
    axis off
    xtest = random(X,1000);
    fxtest = f(xtest);
    ytest  = fun(xtest);
    err = norm(ytest-fxtest)/norm(ytest);
    errinf = norm(ytest-fxtest,'inf')/norm(ytest,'inf');
    fprintf('Mean squared error = %d\n',err)
    
    storenbsamples(end+1) = output.numberOfEvaluations;
    storeprec(end+1) = err;
end
storenbsamples = sort(storenbsamples);
storelog10prec = ceil(sort(log10(storeprec))*10)/10;
figure(1)
H2 = title(['N \in [' num2str(storenbsamples(2)) ',' ...
    num2str(storenbsamples(end-1)) '] , log_{10}(error) \in [' ...
    num2str(storelog10prec(2)) ',' num2str(storelog10prec(end-1)) ']']);
set(H2,'fontsize',16)