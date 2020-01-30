% Magic Points and Interpolation

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

%% Magic points associated with a Functional Basis (1D)
p1 = PolynomialFunctionalBasis(LegendrePolynomials(),0:20);
dom = domain(p1);
g1 = linspace(dom(1),dom(2),200)';
magicpoints = magicPoints(p1,g1);

figure(1)
clf
plot(g1,zeros(size(g1)),'k.','markersize',1)
hold on
plot(magicpoints,zeros(size(magicpoints)),'ro')

%% Magic points associated with a FullTensorProductFunctionalBasis
% Tensorization of 1D uniform grids for the selection of magic points in dimension d
d = 2;
p1 = PolynomialFunctionalBasis(LegendrePolynomials(),0:20);
bases = FunctionalBases.duplicate(p1,d);
p = FullTensorProductFunctionalBasis(bases);
g1 = linspace(dom(1),dom(2),30)';
g = array(FullTensorGrid(g1,d));
magicpoints = p.magicPoints(g);

if d==2
    figure(2)
    clf
    plot(g(:,1),g(:,2),'k.','markersize',1)
    hold on
    plot(magicpoints(:,1),magicpoints(:,2),'ro')
end

%% Magic points associated with a SparseTensorProductFunctionalBasis
% Selection of magic points in dimension d in
% - a tensorization of 1D uniform grids
% - or a tensorization of 1D magic points
tensorizationOfMagicPoints = true;
d = 2;
p1 = PolynomialFunctionalBasis(LegendrePolynomials(),0:20);
bases = FunctionalBases.duplicate(p1,d);
w = [1,2]; % Weights for anisotropic sparsity
Ind = MultiIndices.withBoundedWeightedNorm(2,1,cardinal(p1)-1,w);
p = SparseTensorProductFunctionalBasis(bases,Ind);
if ~tensorizationOfMagicPoints
    g1 = linspace(dom(1),dom(2),100)';
    g = array(FullTensorGrid(g1,d));
else
    g1 = linspace(dom(1),dom(2),1000)';
    m1 = magicPoints(p1,g1);
    g = array(FullTensorGrid(m1,d));
end
magicpoints = p.magicPoints(g);

if d==2
    figure(3)
    clf
    plot(g(:,1),g(:,2),'k.','markersize',1)
    hold on
    plot(magicpoints(:,1),magicpoints(:,2),'ro')
end

%% Interpolation of a function using magic points
d = 2;
f = @(x) cos(x(:,1))+ x(:,2).^6 + x(:,1).^2.*x(:,2);
If = p.interpolate(f,magicpoints);

rv = RandomVector(UniformRandomVariable,d);
Ntest = 1000;
xtest = random(rv,Ntest);
errtest = norm(If(xtest) - f(xtest))/norm(f(xtest));
fprintf('Test error = %d\n',errtest)
