% INTERPOLATIONPOINTSFEATUREMAP - 
%  

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

function xI = interpolationPointsFeatureMap(F,x)
% function xI = interpolationPointsFeatureMap(F,x)
% Constructs interpolation points for feature map or functional basis
% using a greedy algorithm
% 
% F : FunctionalBasis of cardinal m defined on R^d
% x : N-by-d array (interpolation points are selected as n rows of x)
% xI : m-by-d array (interpolation points)


F = F.eval(x);
I = greedyAlgorithml2(F.',[]);
xI = x(I,:);
