% Monte Carlo

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

%% Definitions
fun = UserDefinedFunction(@(x) exp(x),1);
fun.evaluationAtMultiplePoints = true;
X = RandomVector(UniformRandomVariable(0,1));
Y = FunctionOfRandomVector(fun,X);

%% Standard Monte-Carlo
MC = MonteCarlo();
MC.N0 = 200;
MC.std = 1e-2;
[I,output] = MC.mean(Y)

%% Control Variate
funapprox = UserDefinedFunction(@(x) 1+x+x.^2/2,1);
funapprox.evaluationAtMultiplePoints = true;
Z = FunctionOfRandomVector(funapprox,X);
[I,output] = MC.meanControlVariateFunctionOfRandomVector(Y,Z)