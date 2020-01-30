% optimalSamplingMeasure, ProbabilityMeasure, UserDefinedProbabilityMeasure, ProductMeasure, ProbabilityMeasureWithRadonDerivative

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
% UserDefinedProbabilityMeasure with a Normal distribution, slice sampling
dim = 1;
fun = @(x) pdf('Normal',x,0,1);
s = UserDefinedProbabilityMeasure(dim,'pdffun',fun);
n = 10000;
% Sampling using slice sampling
samples = random(s,n);
figure(1)
hold on
histogram(samples, 'normalization', 'pdf')
fplot(@(x)fun(x),[-4,4],'r')
legend('normal distribution')
title('UserDefinedProbabilitMeasure')
hold off

%% ProbabilityMeasureWithRadonDerivative
p = 10;
rv1 = NormalRandomVariable(0,1);
h1 = PolynomialFunctionalBasis(orthonormalPolynomials(rv1),0:p);
nu1 = optimalSamplingMeasure(h1);
% nu1 is a ProbabilityMeasureWithRadonDerivative
rv2 = UniformRandomVariable(-1,1);
h2 = PolynomialFunctionalBasis(orthonormalPolynomials(rv2),0:p);
nu2 = optimalSamplingMeasure(h2);
% nu2 is a ProbabilityMeasureWithRadonDerivative

% We check that we get similar results with the Canonical Basis which is
% not orthonormal
rv3 = NormalRandomVariable(0,1);
h3 = PolynomialFunctionalBasis(CanonicalPolynomials(rv3),0:p);
nu3 = optimalSamplingMeasure(h3);
% nu3 is a ProbabilityMeasureWithRadonDerivative
rv4 = UniformRandomVariable(-1,1);
h4 = PolynomialFunctionalBasis(CanonicalPolynomials(rv4),0:p);
nu4 = optimalSamplingMeasure(h4);

figure(2)
subplot(2,2,1)
hold on
fplot(@(x)pdf(rv1,x),[-10,10],'b')
fplot(@(x)pdf(nu1,x),[-10,10],'r')
legend('pdf of the normal distribution','pdf of the optimal sampling measure')
title('Probability Density Functions')
subplot(2,2,2)
hold on
fplot(@(x)pdf(rv2,x),[-1,1],'b')
fplot(@(x)pdf(nu2,x),[-1,1],'r')
legend('pdf of the uniform distribution','pdf of the optimal sampling measure')
title('Probability Density Functions')
subplot(2,2,3)
hold on
fplot(@(x)pdf(rv3,x),[-10,10],'b')
fplot(@(x)pdf(nu3,x),[-10,10],'r')
legend('pdf of the uniform distribution','pdf of the optimal sampling measure')
title('Probability Density Functions')
subplot(2,2,4)
hold on
fplot(@(x)pdf(rv4,x),[-1,1],'b')
fplot(@(x)pdf(nu4,x),[-1,1],'r')
legend('pdf of the uniform distribution','pdf of the optimal sampling measure')
title('Probability Density Functions')

%% Boosted Optimal Sampling Measure
p = 5;
rvG = NormalRandomVariable(0,1);
hG = PolynomialFunctionalBasis(orthonormalPolynomials(rvG),0:p);
nuG = optimalSamplingMeasure(hG);

% Resampling and conditioning
delta = Inf;
M = 10;
nbeval = 10;
nbrepet = 1000;
parfor i=1:nbrepet
    store = ones(1,M);
    samples = cell(1,M);
    A = cell(1,M);
    W = cell(1,M);
    G = cell(1,M);
    for k=1:M
        samples{k} = random(nuG, nbeval);
        A{k} = hG.eval(samples{k});
        W{k} = diag(sqrt(cardinal(hG)*1./sum(A{k}.^2,2)));
        G{k} = 1/nbeval*(W{k}^2*A{k})'*A{k};
        store(k) = norm(G{k}-eye(cardinal(hG)));
    end
    [val, ind]= min(store);
    xG(i,:) = samples{ind}';
    xGc(i,:) = xG(i,:);
    % Conditioning
    while val > delta
        xGc(i,:) = random(nuG, nbeval)';
        A = hG.eval(xGc(i,:));
        W = diag(sqrt(cardinal(hG)*1./sum(A.^2,2)));
        G = 1/nbeval*(W^2*A)'*A;
        val = norm(G-eye(cardinal(hG)));
    end
    % Subsampling
    % If delta = Inf, greedy algorithm runs until length(xG) = cardinal(hG)
    [xGsubs{i}, deltanew] = greedySubsampling(xGc(i,:)', hG, delta);
end
% Plots
xG = sort(xG,2);
xGc = sort(xGc,2);
xGsubs = cellfun(@sort, xGsubs,'UniformOutput', false);
if delta == Inf
    figure(3)
    subplot(3,1,1)
    hist(xG,100)
    title('Distributions of the x^{nbsamples} sampled from the optimal measure with resampling')
    subplot(3,1,2)
    hist(xGc,100)
    title('Distributions of the x^{nbsamples} sampled from the optimal measure with resampling and conditioning')
    xGsubs = cell2mat(xGsubs) ;
    subplot(3,1,3)
    hist(xGsubs',100)
    title('Distributions of the x^{m} sampled from the optimal measure with resampling, conditioning and subsampling')
else
    figure(3)
    subplot(2,1,1)
    hist(xG,100)
    title('Distributions of the x^{nbsamples} sampled from the optimal measure with resampling')
    subplot(2,1,2)
    hist(xGc,100)
    title('Distributions of the x^{nbsamples} sampled from the optimal measure with resampling and conditioning')
end

p = 5;
rvU = UniformRandomVariable(-1,1);
hU = PolynomialFunctionalBasis(orthonormalPolynomials(rvU),0:p);
nuU = optimalSamplingMeasure(hU);

% Resampling and conditioning
delta = Inf;
M = 10;
nbeval = 10;
nbrepet = 1000;
parfor i=1:nbrepet
    store = ones(1,M);
    samples = cell(1,M);
    A = cell(1,M);
    W = cell(1,M);
    G = cell(1,M);
    for k=1:M
        samples{k} = random(nuU, nbeval);
        A{k} = hU.eval(samples{k});
        W{k} = diag(sqrt(cardinal(hU)*1./sum(A{k}.^2,2)));
        G{k} = 1/nbeval*(W{k}^2*A{k})'*A{k};
        store(k) = norm(G{k}-eye(cardinal(hU)));
    end
    [val, ind]= min(store);
    xU(i,:) = samples{ind}';
    xUc(i,:) = xU(i,:);
    % Conditioning
    while val > delta
        xUc(i,:) = random(nuU, nbeval)';
        A = hU.eval(xUc(i,:));
        W = diag(sqrt(cardinal(hU)*1./sum(A.^2,2)));
        G = 1/nbeval*(W^2*A)'*A;
        val = norm(G-eye(cardinal(hU)));
    end
    % Subsampling
    % If delta = Inf, greedy algorithm runs until length(xG) = cardinal(hG)
    [xUsubs{i}, deltanew] = greedySubsampling(xUc(i,:)', hU, delta);
end
% Plots
xU = sort(xU,2);
xUc = sort(xUc,2);
xUsubs = cellfun(@sort, xUsubs,'UniformOutput', false);
if delta == Inf
    figure(4)
    subplot(3,1,1)
    hist(xU,100)
    title('Distributions of the x^{nbsamples} sampled from the optimal measure with resampling')
    subplot(3,1,2)
    hist(xUc,100)
    title('Distributions of the x^{nbsamples} sampled from the optimal measure with resampling and conditioning')
    xUsubs = cell2mat(xUsubs) ;
    subplot(3,1,3)
    hist(xUsubs',100)
    title('Distributions of the x^{m} sampled from the optimal measure with resampling, conditioning and subsampling')
else
    figure(4)
    subplot(2,1,1)
    hist(xU,100)
    title('Distributions of the x^{nbsamples} sampled from the optimal measure with resampling')
    subplot(2,1,2)
    hist(xUc,100)
    title('Distributions of the x^{nbsamples} sampled from the optimal measure with resampling and conditioning')
end