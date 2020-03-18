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

% TreeBasedTensor: random generation
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

%% Truncation of a TreeBasedTensor with prescribed rank
tr = Truncator();
tr.tolerance = 0;
tr.maxRank = randi(4);
x = TreeBasedTensor.rand(T,'random','random','random');
xr = tr.hsvd(x);
fprintf('Prescribed rank, error = %d\n',norm(x-xr)/norm(x));

%% Truncation of a TreeBasedTensor with prescribed relative precision
d = 10;
T = DimensionTree.random(d,[3,3]);
x = TreeBasedTensor.rand(T,'random','random');

tr = Truncator();
tr.tolerance = 1e-8;

tr.hsvdType = 1; % root to leaves truncation
xr1 = tr.hsvd(x);
err1 = norm((x)-(xr1))/norm((x));
fprintf('Root to leaves: prescribed tolerance = %.1d, error = %.1d\n',tr.tolerance,err1);

tr.hsvdType = 2; % leaves to root truncation
xr2 = tr.hsvd(x);
err2 = norm((x)-(xr2))/norm((x));
fprintf('Leaves to root: prescribed tolerance = %.1d, error = %.1d\n',tr.tolerance,err2);

figure(4)
clf
plot(xr2,representationRank(xr2));
title('Representation ranks')

%% Truncation of a FullTensor
d = 5;
rangOfArity = [2,3];
T = DimensionTree.random(d,rangOfArity);

sz = randi([8,10],1,d);
activeNodes=TreeBasedTensor.randomIsActiveNode(T);

A = FullTensor.randn(sz);
tr = Truncator();
tr.tolerance = 5e-1;
tr.maxRank = Inf;
Ar = tr.hsvd(A,T,activeNodes);
err = norm(A-full(Ar))/norm(A);
fprintf('Tolerance = %.1d, error = %.1d\n',tr.tolerance,err);

figure(10)
clf
plot(Ar,representationRank(Ar));
title('Representation ranks')

tr.tolerance = 1e-2;
Ar = tr.hsvd(A,T,activeNodes);
err = norm(A-full(Ar))/norm(A);
fprintf('Tolerance = %.1d, error = %.1d\n',tr.tolerance,err);

figure(11)
clf
plot(Ar,representationRank(Ar));
title('Representation ranks')

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
output = x.plotNeuralNetwork('linewidth',.5,'markersize',3,'markerfacecolor','k')   ;
axis off
title('Corresponding neural network')
