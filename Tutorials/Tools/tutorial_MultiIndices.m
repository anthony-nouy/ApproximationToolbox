% MultiIndices

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

clearvars, clc, close all

%% Product set
I = MultiIndices.productSet({1:3,1:2});
figure(1)
clf
plot(I)

%% Set of indices with bounded p-norm
m = 6;
p = 1/2;
d = 2;
Ip = MultiIndices.withBoundedNorm(d,p,m);
figure(2)
clf
plot(Ip)
a = title(['$\mathcal{A} = \{\Vert \alpha \Vert_{' num2str(p) '} \le '  num2str(m) '\}$']);
set(a,'Interpreter','latex')

%% Set of indices bounded by an multi-index
p = [3,2];
Ip = MultiIndices.boundedBy(p);
figure(3)
clf
plot(Ip)
a = title(['$\mathcal{A} = \{\alpha :  \alpha \le ' mat2str(p) '\}$']);
set(a,'Interpreter','latex')

%% Operations on indices
I1 = MultiIndices.boundedBy([1,3]);
I2 = MultiIndices.boundedBy([3,1]);
% Adding two sets
I = addIndices(I1,I2);
I.array
figure(1)
clf
plot(I)

% Substracting a set
I = removeIndices(I1,I2);
I.array
figure(2)
clf
plot(I)

% Adding (or substracting) an integer
I = I+2;
figure(3)
clf
plot(I)

%% Maximal elements, margin, reduced margin
I1 = MultiIndices.boundedBy([1,3]);
I2 = MultiIndices.boundedBy([3,1]);
I = addIndices(I1,I2);
Imarg = getMargin(I);
Ired = getReducedMargin(I);
Imax = getMaximalIndices(I);

% Plotting maximal elements and margin
figure(1)
clf
plot(I,'margin',true,'maximal',true)
a = title('margin $\mathcal{M}(\Lambda)$ and maximal elements $\max(\Lambda)$');
set(a,'Interpreter','latex')

% Plotting reduced margin
figure(2)
clf
plot(I,'reducedMargin',true)
a = title('reduced margin $\mathcal{M}_r(\Lambda)$');
set(a,'Interpreter','latex')

%% Check whether a set is downward closed
d = 2;
I = MultiIndices.withBoundedNorm(d,1,4);
figure(1)
clf
plot(I)
title(['isDownwardClosed = ' num2str(isDownwardClosed(I))])

I = addIndices(I,MultiIndices([2,4]));
figure(2)
clf
plot(I)
title(['isDownwardClosed = ' num2str(isDownwardClosed(I))])

%% When indices represent subindices of an nd array of size sz
% Obtaining the position of multi-indices in the nd array
I = MultiIndices.boundedBy([3,4,2],1);
e = sub2ind(I,[5,5,5]);
% Creating a multi-index set associated with entries of a multi-array
I = MultiIndices.ind2sub([5,5,5],e);

%% Reduced margin of a sum of two MultiIndices
I1 = MultiIndices.boundedBy([3,5,7,4,3]);
I2 = MultiIndices.boundedBy([7,5,3,4,2]);
I = addIndices(I1,I2);
tic
Imarg = getReducedMargin(I);
Imarg.array
toc
figure(1)
clf
plot(I,'reducedmargin',1,'maximal',1)