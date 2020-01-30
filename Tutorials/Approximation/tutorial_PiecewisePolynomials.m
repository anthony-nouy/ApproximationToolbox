% PiecewisePolynomialFunctionalBasis

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

clc, clearvars, close all

%%
d=2;
X = RandomVector(UniformRandomVariable(0,1),2);
f = vectorize('1/(x1+x2)^.25');
f = UserDefinedFunction(f,d);
f.measure = X;


singularityAdapted = true;

if ~singularityAdapted
    h = 2^(-5);
    p = 5;
    bases = PiecewisePolynomialFunctionalBasis.hp(0,1,h,p);
else
    h = 2^(-20); % mesh size near singularities
    bases = PiecewisePolynomialFunctionalBasis.singularityhpAdapted(0,1,0,h);
end
bases = FunctionalBases.duplicate(bases,d);


H = FullTensorProductFunctionalBasis(bases);
[~,g] = interpolationPoints(H);
g = FullTensorGrid(g);
If = H.tensorProductInterpolation(f,g);

[errL2,errLinf] = testError(f,If,100000);
fprintf('L2 error = %d\n',errL2)
fprintf('Linfty error = %d\n',errLinf)
fprintf('storage = %d\n',storage(If))

figure(1)
xg=array(g);
plot(xg(:,1),xg(:,2),'.')
title('interpolation points')

figure(2)
subplot(1,2,1)
surf(f,200,'edgecolor','none')
ax = axis;
ca=caxis;
title('The function')
subplot(1,2,2)
surf(If,200,'edgecolor','none')
axis(ax);
caxis(ca);
title('Its interpolation')
