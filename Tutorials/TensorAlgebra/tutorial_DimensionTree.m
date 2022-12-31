% Tutorial for DimensionTree

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
plotDims(T,1:T.nbNodes)
title('Nodes dimensions at the leaves')

%% Random dimension tree
d=10;
rangOfArity = [2,3];
T = DimensionTree.random(d,rangOfArity);
figure(2)
clf
subplot(1,2,1)
plot(T)
title('Nodes indices')
subplot(1,2,2)
plotDims(T)
title('Nodes dimensions at the leaves')

%% Balanced dimension tree
d = 10;
T=DimensionTree.balanced(d);
figure(1)
subplot(1,2,1)
plot(T)
title('Nodes indices')
subplot(1,2,2)
plotDims(T)
title('Nodes dimensions at the leaves')

figure(2)
clf
subplot(1,2,1)
plot(T)
hold on
plotNodes(T,ascendants(T,4),'ob','markerfacecolor','b')
title('Ascendants of node 4')
subplot(1,2,2)
plot(T)
hold on
plotNodes(T,descendants(T,4),'ob','markerfacecolor','b')
title('Descendants of node 4')

%% Extracting a subtree
[ST,nod] = subDimensionTree(T,4);
figure(3)
clf
subplot(1,2,1)
plot(T)
hold on
plotNodes(T,nod,'ob','markerfacecolor','b')
subplot(1,2,2)
plot(ST)

%% script for plotting a dimension tree in latex with tikz
d=8;
T = DimensionTree.random(d);
tik = tikzDimensionTree(T,1:T.nbNodes,1)
% copy paste the value of tik in your latex document

%% More examples for plotting dimension trees
d=8;
T = DimensionTree.random(d);

% plotting nodes with same level at the same height or with matlab layout
figure(4)
clf
subplot(1,2,1)
T.plotOptions.levelAlignment = false;
plotWithLabelsAtNodes(T,T.level);
subplot(1,2,2)
T.plotOptions.levelAlignment = true;
plotWithLabelsAtNodes(T,T.level);

% different label types, changing fonts and sizes

figure(5)
clf
plotNodes(T,[],'ko','markersize',12,'markerfacecolor','k')
plotLabelsAtNodes(T,4,1,...
    'Interpreter','latex','fontsize',24);
plotLabelsAtNodes(T,{'A','B'},2:3,...
    'Interpreter','latex','fontsize',24);
hold on
plotEdges(T,[],'k-','linewidth',1)



%% Change the root of a tree
d = 8;
tree = DimensionTree.balanced(d);
figure(6)
subplot(3,2,1)
plot(tree)
title('Nodes indices before changing the root')
subplot(3,2,2)
plotDims(tree)
title('Nodes dimensions before changing the root')
newtree = tree.changeRoot(5);
subplot(3,2,3)
plot(newtree)
title('Nodes indices after changing the root to node 5')
subplot(3,2,4)
plotDims(tree)
title('Nodes dimensions after changing the root to node 5')
newtree=tree.changeRoot(11);
subplot(3,2,5)
plot(newtree)
title('Nodes indices after changing the root to leaf node 11')
subplot(3,2,6)
plotDims(newtree)
title('Nodes dimensions after changing the root to leaf node 11')

               
%% Add a child to a node
d = 8;
tree = DimensionTree.balanced(d);
figure(7)
subplot(2,2,1)
tree.plot()
title('Nodes indices before changing the root')
newtree = tree.addChild(5);
subplot(2,2,2)
newtree.plot()
title('Nodes indices after adding a child to node 5')
newtree = tree.addChild(11);
subplot(2,2,3)
newtree.plot()
title('Nodes indices after adding a child to leaf node 11')
newtree = tree.addChild(1);
subplot(2,2,4)
newtree.plot()
title('Nodes indices after adding a child to root node 1')


