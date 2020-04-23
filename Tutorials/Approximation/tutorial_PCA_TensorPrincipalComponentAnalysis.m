% PCA for low rank approximation of tensors
%
% See the following article:  
% Anthony Nouy. Higher-order principal component analysis for the approximation of
% tensors in tree-based low-rank formats. Numerische Mathematik, 141(3):743--789, Mar 2019.


% Copyright (c) 2020, Loic Giraldi, Erwan Grelier, Anthony Nouy
% 
% This file is part of the Matlab Toolbox ApproximationToolbox.
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

%% Creating a function providing the entries of a tensor 
d = 5;
n=10;
sz = repmat(n,1,d);
fun = @(i) cos(i(:,1)/sz(1))+1./(1+(i(:,2)/sz(2)).^2+(i(:,3)/sz(3)).^4)+i(:,3)/(3);

fun = UserDefinedFunction(fun,d);
fun.evaluationAtMultiplePoints = true;
X = randomMultiIndices(sz);
tol = 1e-8;

%% HOPCA (PCA for each dimension, provides reduced spaces)
fprintf('--- Higher order PCA ---- \n')
TPCA = TensorPrincipalComponentAnalysis();
TPCA.PCASamplingFactor = 1;
TPCA.PCAAdaptiveSampling = 0;
% prescribed tolerance
fprintf('\nPrescribed tolerance\n')
TPCA.tol = tol;
[subbases,outputs] = TPCA.hopca(fun,sz);
fprintf('Number of evaluations \n= [%s]\n',num2str(cellfun(@(x) x.numberOfEvaluations,outputs)));
fprintf('Ranks {1,...,d} \n= [%s]\n',num2str(cellfun(@(x) size(x,2),subbases)));

% prescribed ranks
fprintf('\nPrescribed ranks\n')
TPCA.tol = inf;
TPCA.maxRank = randi(4,1,d);
[subbases,outputs] = TPCA.hopca(fun,sz);
fprintf('Number of evaluations \n= [%s]\n',num2str(cellfun(@(x) x.numberOfEvaluations,outputs)));
fprintf('Ranks {1,...,d} \n= [%s]\n',num2str(cellfun(@(x) size(x,2),subbases)));

%% Approximation in Tucker Format

fprintf('--- Approximation in Tucker format ---- \n')
TPCA = TensorPrincipalComponentAnalysis();
TPCA.PCASamplingFactor = 1;
TPCA.PCAAdaptiveSampling = 0;
% prescribed ranks
fprintf('\nPrescribed ranks\n')
TPCA.tol = inf;
TPCA.maxRank = randi(4,1,d);

[f,output] = TPCA.TuckerApproximation(fun,sz);

fprintf('Number of evaluations = %d\n',output.numberOfEvaluations);
fprintf('Prescribed ranks = [%s]\n',num2str(TPCA.maxRank));
fprintf('Ranks = [%s]\n',num2str(f.ranks(f.tree.dim2ind)));
xtest = random(X,10000);
fxtest = f.evalAtIndices(xtest);
ytest  = fun(xtest);
fprintf('Error = %d\n',norm(ytest-fxtest)/norm(ytest))


% prescribed tolerance
fprintf('\nPrescribed tolerance\n')
TPCA.tol = tol;
TPCA.maxRank = inf;
[f,output] = TPCA.TuckerApproximation(fun,sz);

fprintf('Number of evaluations = %d\n',output.numberOfEvaluations);
fprintf('Ranks = [%s]\n',num2str(f.ranks(f.tree.dim2ind)));
xtest = random(X,10000);
fxtest = f.evalAtIndices(xtest);
ytest  = fun(xtest);
fprintf('Error = %d\n',norm(ytest-fxtest)/norm(ytest))


%% Approximation in Tree based format
fprintf('--- Approximation in Tree based format ---- \n')
TPCA = TensorPrincipalComponentAnalysis();
TPCA.PCASamplingFactor = 1;
TPCA.PCAAdaptiveSampling = 0;
tree = DimensionTree.balanced(d);

% prescribed ranks
fprintf('\nPrescribed ranks\n')
TPCA.tol = inf;
TPCA.maxRank = randi(8,1,tree.nbNodes); TPCA.maxRank(tree.root)=1;
[f,output] = TPCA.TBApproximation(fun,sz,tree);

fprintf('Number of evaluations = %d\n',output.numberOfEvaluations);
fprintf('Storage = %d\n',storage(f));
fprintf('Prescribed ranks = \n [%s]\n',num2str(TPCA.maxRank));
fprintf('Ranks = \n [%s]\n',num2str(f.ranks));

xtest = random(X,1000);
fxtest = f.evalAtIndices(xtest);
ytest  = fun(xtest);
err = norm(ytest-fxtest)/norm(ytest);
fprintf('Mean squared error = %d\n',err)

% prescribed tolerance
fprintf('\nPrescribed tolerance\n')
TPCA.tol = tol;
TPCA.maxRank = inf;
[f,output] = TPCA.TBApproximation(fun,sz,tree);

fprintf('Number of evaluations = %d\n',output.numberOfEvaluations);
fprintf('Storage = %d\n',storage(f));
fprintf('Ranks = [%s]\n',num2str(f.ranks));

xtest = random(X,1000);
fxtest = f.evalAtIndices(xtest);
ytest  = fun(xtest);
err = norm(ytest-fxtest)/norm(ytest);
fprintf('Mean squared error = %d\n',err)

% prescribed tolerance (adaptive sampling)
fprintf('\nPrescribed tolerance\n')
TPCA.tol = tol;
TPCA.PCAAdaptiveSampling = 1;
TPCA.PCASamplingFactor = 1.2;
TPCA.maxRank = inf;
[f,output] = TPCA.TBApproximation(fun,sz,tree);

fprintf('Number of evaluations = %d\n',output.numberOfEvaluations);
fprintf('Storage = %d\n',storage(f));
fprintf('Ranks = [%s]\n',num2str(f.ranks));

xtest = random(X,1000);
fxtest = f.evalAtIndices(xtest);
ytest  = fun(xtest);
err = norm(ytest-fxtest)/norm(ytest);
fprintf('Mean squared error = %d\n',err)



%% Approximation in Tensor Train format
fprintf('--- Approximation in Tensor Train format ---- \n')
TPCA = TensorPrincipalComponentAnalysis();
TPCA.PCASamplingFactor = 1;
TPCA.PCAAdaptiveSampling = 0;
% prescribed tolerance
fprintf('\nPrescribed tolerance\n')
TPCA.tol = tol;
[f,output] = TPCA.TTApproximation(fun,sz);
fprintf('Number of evaluations = %d\n',output.numberOfEvaluations);
fprintf('Storage = %d\n',storage(f));
ttranks = flip(f.ranks(f.isActiveNode));
fprintf('TT-ranks = \n [%s]\n',num2str(ttranks(1:end-1)));

xtest = random(X,1000);
fxtest = f.evalAtIndices(xtest);
ytest  = fun(xtest);
err = norm(ytest-fxtest)/norm(ytest);
fprintf('Mean squared error = %d\n',err)

% prescribed ranks
fprintf('\nPrescribed ranks\n')
TPCA.tol = inf;
TPCA.maxRank = randi(8,1,d-1); 
[f,output] = TPCA.TTApproximation(fun,sz);
fprintf('Number of evaluations = %d\n',output.numberOfEvaluations);
fprintf('Storage = %d\n',storage(f));
fprintf('Prescribed TT-ranks = \n [%s]\n',num2str(TPCA.maxRank));
ttranks = flip(f.ranks(f.isActiveNode));
fprintf('TT-ranks = \n [%s]\n',num2str(ttranks(1:end-1)));

xtest = random(X,1000);
fxtest = f.evalAtIndices(xtest);
ytest  = fun(xtest);
err = norm(ytest-fxtest)/norm(ytest);
fprintf('Mean squared error = %d\n',err)
