% ProbabilityMeasure, UserDefinedProbabilityMeasure, ProbabilityMeasureWithRadonDerivative

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

%% UserDefinedProbabilityMeasure
pdffun = UserDefinedFunction(@(x) (1-x).^2.*3./8.*isIn([-1,1],x),1);
mu = UserDefinedProbabilityMeasure(1,'pdf',pdffun,'supp',[-1,1]);
figure(1)
clf
subplot(1,3,1)
x = linspace(-2,2,100)';
plot(x,mu.pdf(x));
title('PDF')

subplot(1,3,2)
plot(x,mu.cdf(x))
title('CDF')

subplot(1,3,3)
u = linspace(0,1,100);
plot(u,mu.icdf(u))
title('Quantile function')

%% MixtureProbabilityMeasure
X1 = NormalRandomVariable(0,1/2);
X2 = NormalRandomVariable(3,1/4);
mu = MixtureProbabilityMeasure({X1,X2},[1/3,2/3]);
x = linspace(-3,6,100)';

figure(2)
clf
subplot(1,2,1)
plot(x,mu.pdf(x));
hold on
r=random(mu,100);
plot(r,zeros(1,length(r)),'.')
title('PDF')

subplot(1,2,2)
plot(x,mu.cdf(x))
hold on
pdfPlot(DiscreteRandomVariable(r(:)))
ylim([0 1])
title('CDF')

figure(3)
clf
plot(x,mu.pdf(x));
r=random(mu,1000);
hold on
plot(r,zeros(1,length(r)),'.')

%% ProbabilityMeasureWithRadonDerivative
mu = UniformRandomVariable(-1,1);
w = UserDefinedFunction(@(x) (1-x).^2*3/4,1);
nu = ProbabilityMeasureWithRadonDerivative(mu,w);

x = linspace(-2,2,100)';
figure(4)
clf
subplot(1,2,1)
plot(x,nu.pdf(x))
title('PDF')

subplot(1,2,2)
plot(x,nu.cdf(x))
title('CDF')

%% ProbabilityMeasureWithRadonDerivative
mu = NormalRandomVariable(0,1);
s = truncatedSupport(mu);
p = 12;
h = PolynomialFunctionalBasis(orthonormalPolynomials(mu),0:p);
w = christoffel(h);
nu = ProbabilityMeasureWithRadonDerivative(mu,w);
figure(10)
clf
plot(nu,'pdf')
tic;r = random(nu,100);toc;
hold on
plot(r,zeros(1,length(r)),'.')

%% ProbabilityMeasureWithRadonDerivative
mu = UniformRandomVariable(0,1);
s = support(mu);
p = 12;
h = PolynomialFunctionalBasis(orthonormalPolynomials(mu),0:p);
nu = optimalSamplingMeasure(h);
figure(10)
clf
x = linspace(0,1,100)';
subplot(1,2,1)
plot(x,nu.pdf(x))
title('PDF')

subplot(1,2,2)
plot(x,nu.cdf(x))
title('CDF')