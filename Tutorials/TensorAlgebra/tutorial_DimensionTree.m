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
plotDims(T)
title('Nodes dimensions')

%% Random dimension tree
d=20;
rangOfArity = [2,3];
T = DimensionTree.random(d,rangOfArity);
figure(2)
subplot(1,2,1)
plot(T)
title('Nodes indices')
subplot(1,2,2)
plotDims(T)
title('Nodes dimensions')

%% Balanced dimension tree
d = 10;
T=DimensionTree.balanced(d);
figure(1)
subplot(1,2,1)
plot(T)
title('Nodes indices')
subplot(1,2,2)
plotDims(T)
title('Nodes dimensions')

figure(2)
clf
subplot(1,2,1)
plot(T)
hold on
plotNodes(T,ascendants(T,4),'ob','markerfacecolor','b')
title('ascendants of node 4')
subplot(1,2,2)
plot(T)
hold on
plotNodes(T,descendants(T,4),'ob','markerfacecolor','b')
title('descendants of node 4')

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