% Tutorial for truncation of tensors using Tree Based tensor format

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

%% Truncation of a TreeBasedTensor with prescribed rank
d = 20;
rangOfArity = [2,3];
T = DimensionTree.random(d,rangOfArity);

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

