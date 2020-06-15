% Boosted least squares
% Numerical experiment of the article
% Haberstich, C., Nouy, A., & Perrin, G. (2019). Boosted optimal weighted 
% least-squares. arXiv preprint arXiv:1912.07075.

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
p = 10;

% Two examples
choice = 1;
switch choice
    case 1
        fun = @(x) exp(-(x-1).^2/4);
        RV = NormalRandomVariable(0,1);
        basis = PolynomialFunctionalBasis(HermitePolynomials(),0:p);
    case 2
        fun = @(x) (1./(1+5*x.^2));
        RV = UniformRandomVariable(-1,1);
        basis = PolynomialFunctionalBasis(LegendrePolynomials(),0:p);
end
fun = UserDefinedFunction(fun,1);
fun.evaluationAtMultiplePoints = true;


nbrepet = 10;
delta = 0.9;
eta = 0.01;
ddelta = -delta + (1+delta)*log(1+delta);


m = cardinal(basis);
nu = optimalSamplingMeasure(basis);
nbsamples = round(ddelta^(-1)*m*log(2*m/eta));


%% Guaranteed stability
% Comparaison of 3 methods OWLS (Optimal Weighted Least-Squares), c-BLS
% (M=1) (conditionned-Boosted Least Squares with 1 repetition) and
% c-BLS (M=100) (conditionned-Boosted Least Squares with 100 repetition).

% least squares solver
ls = LinearModelLearningSquareLoss();

% OWLS
storenbsamplesOWLS = zeros(1,nbrepet);
storeprecOWLS = zeros(1,nbrepet);

% c-BLS (M=1)
storenbsamplesBLS1 = zeros(1,nbrepet);
storeprecBLS1 = zeros(1,nbrepet);

% c-BLS (M=100)
storenbsamplesBLS2 = zeros(1,nbrepet);
storeprecBLS2 = zeros(1,nbrepet);

parfor k = 1:nbrepet
    xtest = random(RV,1000);
    ytest = fun(xtest);
    % fprintf('Optimal weighted least squares\n')
    xOWLS= random(nu,m);
    A = basis.eval(xOWLS);
    W = diag(sqrt(cardinal(basis)*1./sum(A.^2,2)));
    yOWLS = fun(xOWLS);
    lsloc = ls;
    lsloc.basisEval = W*A;
    lsloc.trainingData = {[], W*yOWLS};
    fd = lsloc.solve();
    fd = FunctionalBasisArray(fd,basis);
    fxtest = fd(xtest);
    errOWLS = norm(ytest-fxtest)/norm(ytest);
    storenbsamplesOWLS(k) = nbsamples;
    storeprecOWLS(k) = sqrt(errOWLS);
    % fprintf('c-BLS (M=1)\n')
    dist = Inf;
    % conditioning
    while dist >  delta
        xBLS1 = random(nu, nbsamples);
        ABLS1 = basis.eval(xBLS1);
        WBLS1 = diag(sqrt(m*1./sum(ABLS1.^2,2)));
        GBLS1= 1/nbsamples*(WBLS1^2*ABLS1)'*ABLS1;
        dist = norm(GBLS1-eye(m));
    end
    yBLS1 = fun(xBLS1);
    lsloc.basisEval = WBLS1*ABLS1;
    lsloc.trainingData = {[], WBLS1*yBLS1};
    fBLS1 = lsloc.solve();
    fBLS1 = FunctionalBasisArray(fBLS1,basis);
    fxtest = fBLS1(xtest);
    errBLS1 = norm(ytest-fxtest)/norm(ytest);
    storenbsamplesBLS1(k) = length(xBLS1);
    storeprecBLS1(k) = sqrt(errBLS1);
    % c-BLS (M=100)\n'
    M = 100; % Number of repetitions
    eta2 = eta.^(1/M); % Actualisation of eta with the repetitions
    nbsamples2 = round(ddelta^(-1)*m*log(2*m/eta2));
    store = ones(1,M);
    samples = cell(1,M);
    for kk=1:M
        xBLS2 = random(nu, nbsamples2);
        ABLS2 = basis.eval(xBLS2);
        WBLS2 = diag(sqrt(m*1./sum(ABLS2.^2,2)));
        GBLS2= 1/nbsamples2*(WBLS2^2*ABLS2)'*ABLS2;
        store(kk) = norm(GBLS2-eye(m));
        samples{kk} = xBLS2;
    end
    [mini, ind]= min(store);
    xBLS2 = samples{ind};
    ABLS2 = basis.eval(xBLS2);
    WBLS2 = diag(sqrt(cardinal(basis)*1./sum(ABLS2.^2,2)));
    % conditioning
    while mini >  delta
        xBLS2 = random(nu, nbsamples2);
        ABLS2 = basis.eval(xBLS2);
        WBLS2 = diag(sqrt(cardinal(basis)*1./sum(ABLS2.^2,2)));
        GBLS2= 1/nbsamples2*(WBLS2^2*ABLS2)'*ABLS2;
        mini = norm(GBLS2-eye(m));
    end
    yBLS2 = fun(xBLS2);
    lsloc.basisEval = WBLS2*ABLS2;
    lsloc.trainingData = {[], WBLS2*yBLS2};
    fBLS2 = lsloc.solve();
    fBLS2 = FunctionalBasisArray(fBLS2,basis);
    fxtest = fBLS2(xtest);
    errBLS2 = norm(ytest-fxtest)/norm(ytest);
    storenbsamplesBLS2(k) = length(xBLS2);
    storeprecBLS2(k) = sqrt(errBLS2);
end

fprintf('OWLS mean squared error = %d\n', mean(storeprecOWLS))
fprintf('OWLS mean samples = %d\n', mean(storenbsamplesOWLS))
fprintf('c-BLS (M=1) mean squared error = %d\n', mean(storeprecBLS1))
fprintf('c-BLS (M=1) mean samples = %d\n', mean(storenbsamplesBLS1))
fprintf('c-BLS (M=100) mean squared error = %d\n', mean(storeprecBLS2))
fprintf('c-BLS (M=100) mean samples = %d\n', mean(storenbsamplesBLS2))

%% Given cost

% Comparison of 5 methods I (classical Interpolation), 
% EI (Empirical Interpolation Method computed with magic points),
% OWLS (Optimal Weighted Least-Squares), BLS (M = 100), (Boosted Least
% Squares with Resampling) and s-BLS, (Boosed Least Squares with
% subsampling plus conditioning).

% least squares solver
ls = LinearModelLearningSquareLoss();

% classical Interpolation I

% EIM (with random points)
storenbsamplesEIM = zeros(1,nbrepet);
storeprecEIM = zeros(1,nbrepet);

% OWLS
storenbsamplesOWLS = zeros(1,nbrepet);
storeprecOWLS = zeros(1,nbrepet);

% BLS (M=100)
storenbsamplesBLS = zeros(1,nbrepet);
storeprecBLS = zeros(1,nbrepet);

% sBLS
storenbsamplesSBLS = zeros(1,nbrepet);
storeprecSBLS = zeros(1,nbrepet);


% fprintf('classical interpolation\n')
switch choice
    case 1
        xIc = gaussIntegrationRule(RV,m);
        Ic = basis.interpolate(fun, xIc.points);
    case 2
        xIc = chebyshevPoints(m);
        Ic = basis.interpolate(fun, xIc);
end
xtest = random(RV,1000);
fxtest1 = Ic(xtest);
ytest  = fun(xtest);
errIc = norm(ytest-fxtest1)/norm(ytest);

storenbsamplesI = m;
storeprecI = errIc;


for k = 1:nbrepet
    % fprintf('EIM (avec random points)\n')
    magicpoints = random(RV,10000);
    magicpoints = magicPoints(basis, magicpoints);
    EIM = basis.interpolate(fun, magicpoints);
    fxtest2 = EIM(xtest);
    ytest  = fun(xtest);
    errEIM = norm(ytest-fxtest2)/norm(ytest);
    storenbsamplesEIM(k) = m;
    storeprecEIM(k) = errEIM;
    % fprintf('Optimal weighted least squares\n')
    xOWLS= random(nu,m);
    A = basis.eval(xOWLS);
    W = diag(sqrt(cardinal(basis)*1./sum(A.^2,2)));
    yOWLS = fun(xOWLS);
    ls.basisEval = W*A;
    ls.trainingData = {[], W*yOWLS};
    fd = ls.solve();
    fd = FunctionalBasisArray(fd,basis);
    fxtest = fd(xtest);
    errOWLS = norm(ytest-fxtest)/norm(ytest);
    storenbsamplesOWLS(k) = m;
    storeprecOWLS(k) = errOWLS;
    % fprintf('Boosted weighted least squares (M=100) \n')
    M = 100;
    store = ones(1,M);
    samples = cell(1,M);
    for kk=1:M
        xBLS = random(nu, m);
        ABLS = basis.eval(xBLS);
        WBLS = diag(sqrt(m*1./sum(ABLS.^2,2)));
        GBLS= 1/m*(WBLS^2*ABLS)'*ABLS;
        store(kk) = norm(GBLS-eye(m));
        samples{kk} = xBLS;
    end
    [mini, ind]= min(store);
    xBLS = samples{ind};
    ABLS = basis.eval(xBLS);
    WBLS = diag(sqrt(m*1./sum(ABLS.^2,2)));
    yBLS = fun(xBLS);
    ls.basisEval = WBLS*ABLS;
    ls.trainingData = {[], WBLS*yBLS};
    fBLS = ls.solve();
    fBLS = FunctionalBasisArray(fBLS,basis);
    fxtest = fBLS(xtest);
    errBLS = norm(ytest-fxtest)/norm(ytest);
    storenbsamplesBLS(k) = length(xBLS);
    storeprecBLS(k) = errBLS;
    % fprintf('sBLS\n')
    % resampling
    M = 100; % Number of repetitions
    eta2 = eta.^(1/M); % Actualisation of eta with the repetitions
    nbsamples2 = round(ddelta^(-1)*m*log(2*m/eta2));
    store = ones(1,M);
    samples = cell(1,M);
    for kk=1:M
        xBLS2 = random(nu, nbsamples2);
        ABLS2 = basis.eval(xBLS2);
        WBLS2 = diag(sqrt(m*1./sum(ABLS2.^2,2)));
        GBLS2= 1/nbsamples2*(WBLS2^2*ABLS2)'*ABLS2;
        store(kk) = norm(GBLS2-eye(m));
        samples{kk} = xBLS2;
    end
    [mini, ind]= min(store);
    xBLS2 = samples{ind};
    ABLS2 = basis.eval(xBLS2);
    WBLS2 = diag(sqrt(m*1./sum(ABLS2.^2,2)));
    xSBLS = xBLS2;
    % conditioning
    dist = mini;
    while dist > delta
        xSBLS = random(nu, nbsamples);
        ASBLS = basis.eval(xSBLS);
        WSBLS = diag(sqrt(m*1./sum(ASBLS.^2,2)));
        GSBLS= 1/nbsamples*(WSBLS^2*ASBLS)'*ASBLS;
        dist = norm(GSBLS-eye(m));
    end
    % Greedy removal of samples
    [xSBLS, deltanew] = greedySubsampling(xSBLS, basis, delta);
    ASBLS = basis.eval(xSBLS);
    WSBLS = diag(sqrt(m*1./sum(ASBLS.^2,2)));
    ySBLS = fun(xSBLS);
    ls.basisEval = WSBLS*ASBLS;
    ls.trainingData = {[], WSBLS*ySBLS};
    fSBLS = ls.solve();
    fSBLS = FunctionalBasisArray(fSBLS,basis);
    fxtest = fSBLS(xtest);
    errSBLS = norm(ytest-fxtest)/norm(ytest);
    storenbsamplesSBLS(k) = length(xSBLS);
    storeprecSBLS(k) = errSBLS;
end


fprintf('classical Interpolation I error = %d\n', storeprecI)
fprintf('classical Interpolation I samples = %d\n', storenbsamplesI)
fprintf('EI mean squared error = %d\n', mean(storeprecEIM))
fprintf('EI (M=1) mean samples = %d\n', mean(storenbsamplesEIM))
fprintf('OWLS mean squared error = %d\n', mean(storeprecOWLS))
fprintf('OWLS mean samples = %d\n', mean(storenbsamplesOWLS))
fprintf('BLS mean squared error = %d\n', mean(storeprecBLS))
fprintf('BLS mean samples = %d\n', mean(storenbsamplesBLS))
fprintf('sBLS mean squared error = %d\n', mean(storeprecSBLS))
fprintf('sBLS mean samples = %d\n', mean(storenbsamplesSBLS))
%
fprintf('The s-BLS method provides better results than the OWLS and BLS (M=100) methods when n is chosen equal to m.\n')