% Tutorial for Tree Based tensor format

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

%% Linear dimension tree
d = 5;
T=DimensionTree.linear(d);
figure(1)
subplot(1,2,1)
plot(T)
title('Nodes indices')
subplot(1,2,2)
plotDims(T)
title('Nodes dimensions')

%% Random dimension tree
d = 20;
rangOfArity = [2,3];
T = DimensionTree.random(d,rangOfArity);
figure(2)
subplot(1,2,1)
plot(T)
title('Nodes indices')
subplot(1,2,2)
plotDims(T)
title('Nodes dimensions')

%% TreeBasedTensor: random generation
d = 20;
rangOfArity = [2,3];
T = DimensionTree.random(d,rangOfArity);
x = TreeBasedTensor.rand(T,'random','random','random')

figure(3)
subplot(1,3,1)
plot(x)
title('Active leaves')
subplot(1,3,2)
plotWithLabelsAtNodes(x.tree,cellfun(@size,x.tensors,'uniformoutput',false))
title('Tensors sizes')
subplot(1,3,3)
plotWithLabelsAtNodes(x.tree,representationRank(x))
title('Representation ranks')

%% Tensor train Tucker format (or Extended Tensor Train format)
d = 10;
T = DimensionTree.linear(d);
ranks = [1,randi(2,1,T.nbNodes-1)];
sz = 3*ones(1,d);
x = TreeBasedTensor.rand(T,ranks,sz);

figure(21)
subplot(1,2,1);plot(x);title('Active nodes')
subplot(1,2,2);plotDims(T);title('Nodes dimensions')

% With a permutation of dimensions
rperm = randperm(d);
T = DimensionTree.linear(rperm);
x = TreeBasedTensor.rand(T,ranks,sz);

figure(22)
subplot(1,2,1);plot(x);title('Active nodes')
subplot(1,2,2);plotDims(T);title('Nodes dimensions')


%% Tensor train format
d = 10;
T = DimensionTree.linear(d);
ranks = [1,randi(2,1,T.nbNodes-1)];
sz = 3*ones(1,d);
activeNodes = ~T.isLeaf;activeNodes(T.dim2ind(1))=true;
x = TreeBasedTensor.rand(T,ranks,sz,activeNodes);

figure(21)
subplot(1,2,1);plot(x);title('Active nodes')
subplot(1,2,2);plotDims(T);title('Nodes dimensions')

% With a permutation of dimensions
rperm = randperm(d);
T = DimensionTree.linear(rperm);
x = TreeBasedTensor.rand(T,ranks,sz,activeNodes);

figure(22)
subplot(1,2,1);plot(x);title('Active nodes')
subplot(1,2,2);plotDims(T);title('Nodes dimensions')


%% Activate and inactivate nodes
d = 5;
tree = DimensionTree.balanced(d);
x1 = TreeBasedTensor.rand(tree,'random','random');
figure(21)
plot(x1)
title('Initial tensor')
x2 = x1.inactivateNodes([5,9]);
fprintf('check x1 - x2 = 0 : \n%d\n',norm(x1-x2)/norm(x1))
figure(22)
plot(x2)
title('Inactivate nodes [5,9]')
x3 = x2.activateNodes([5,9]);
fprintf('check x1 - x3 = 0 : \n%d\n',norm(x1-x3)/norm(x1))
figure(23)
plot(x3)
title('Reactivate nodes [5,9]')


%% Algebraic operations
d = 8;
tree = DimensionTree.linear(d);
sz = randi([3,10],1,d);
T1 = TreeBasedTensor.rand(tree, 'random',sz);
T2 = TreeBasedTensor.rand(tree, 'random',sz);
fprintf('ranks of T1   : %s\n', num2str(T1.ranks))
fprintf('ranks of T2   : %s\n', num2str(T2.ranks))
fprintf('\nAddition of tensors:\n--------------------\n')
Tplus = T1+T2;
fprintf('ranks of T1+T2: %s\n', num2str(Tplus.ranks))
fprintf('\nSubstraction of tensors:\n----------------------\n')
Tminus= T1-T2;
fprintf('ranks of T1-T2: %s\n', num2str(Tminus.ranks))
fprintf('\nHadamard product of tensors:\n-------------------------\n')
Ttimes = T1.*T2;
fprintf('ranks of T1*T2: %s\n', num2str(Ttimes.ranks))
fprintf('\nNorm of tensors:\n--------------------\n')
fprintf('norm of T1 = %d\n',T1.norm())
fprintf('norm of T1*T2 = %d\n',Ttimes.norm())

fprintf('\nSum of T1 along dimensions\n-------------------------\n')
fprintf('dimensions of T1        : %s\n', num2str(T1.sz))
T1sum = sum(T1,1);
fprintf('dimensions of sum(T1,1) : %s\n', num2str(T1sum.sz))
fprintf('dimensions of sum(T1,1) : %s\n', num2str(size(squeeze(T1sum))))
T1sum = sum(T1,[2,5,7]);
fprintf('dimensions of sum(T1,[2,5,7]) : %s\n', num2str(T1sum.sz))
fprintf('sum of all entries = %d\n',squeeze(sum(T1,1:T1.order)))


%% Changing root node of a tree based tensor

d = 5;
tree = DimensionTree.balanced(d);
sz = randi([4,8], 1, d);
ranks = randi([4,8], 1, tree.nbNodes);
T1 = TreeBasedTensor.rand(tree, ranks ,sz);
figure(10)
subplot(2,2,1)
plot(T1)
title('Nodes indices with root 1')
num2 = 2;
T2 = T1.changeRoot(num2) ;
subplot(2,2,2)
plot(T2)
title(['Nodes indices with root ' num2str(num2)])
num3 = 9;
T3 = T1.changeRoot(num3) ;
subplot(2,2,3)
plot(T3)
title(['Nodes indices with root ' num2str(num3)])
T3bis = T2.changeRoot(num3);
subplot(2,2,4)
plot(T3bis)
title(['Nodes indices with root ' num2str(num3) ,  ...
    ' from format with root ' , num2str(num2)])
norm(T3-T3bis)/(norm(T3)+norm(T3bis))*2


%% Plotting neural network associated with a tree tensor network
d=8;
tree = DimensionTree.random(d);
sz = 10*ones(1,d);
ranks = randi([3,8],1,tree.nbNodes);
ranks(tree.root)=1;
x = TreeBasedTensor.rand(tree,ranks,sz);
figure(31)
clf
subplot(1,2,1)
[N,E,H] = plotWithLabelsAtNodes(x.tree,x.ranks);
set(N,'markersize',12,'markerfacecolor','k')
set(H,'Interpreter','latex','fontsize',24);
set(E,'linewidth',1)
title('Dimension tree and ranks at nodes')
axis off

axis off
subplot(1,2,2)
output = x.plotNeuralNetwork(false,'linewidth',.5,'markersize',3,'markerfacecolor','k')   ;
axis off
title('Corresponding neural network')



