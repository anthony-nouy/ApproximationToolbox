function [gridalpha, dist] = conditionnedResampling(basis, delta, epsilon, M, type)
m = cardinal(basis);
cdelta = -delta + (1+delta).*(log(1+delta));
nb = log(2*m/epsilon)/cdelta*m;
nbsamples = abs(ceil(nb));
nu = optimalSamplingMeasure(basis);
% gridalpha = zeros(1,nbsamples);
% Resampling
for kk=1:M
    gridalpha = randomSequential(nu, nbsamples, type);
    ABLS = basis.eval(gridalpha);
    WBLS = diag(sqrt(m*1./sum(ABLS.^2,2)));
    GBLS= 1/nbsamples*(WBLS^2*ABLS)'*ABLS;
    store(kk) = norm(GBLS-eye(m));
    samples{kk} = gridalpha;
end
[mini, ind]= min(store);
gridalpha = samples{ind};
% conditioning
dist = mini;
nbiter = 0;
while dist > delta && nbiter < 10
    nbiter = nbiter +1;
    gridalpha = randomSequential(nu, nbsamples, type);
    ASBLS = basis.eval(gridalpha);
    WSBLS = diag(sqrt(m*1./sum(ASBLS.^2,2)));
    GSBLS= 1/nbsamples*(WSBLS^2*ASBLS)'*ASBLS;
    dist = norm(GSBLS-eye(m));
    storeC(nbiter) = dist;
    samplesC{nbiter} = gridalpha;
end
if nbiter == 10
    % warning('Stability condition is not verified')
    [mini, ind]= min(storeC);
    gridalpha = samplesC{ind};
end