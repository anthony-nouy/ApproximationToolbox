% tutorial for B-Splines 

% This file is part of ApproximationToolbox.

clearvars; clc; close all
warning('off','MATLAB:fplot:NotVectorized') % For display purposes

%% BSplines 
knots = linspace(-1,1,10)'; % knots
s = 3 ; % degree
h = BSplinesFunctionalBasis(knots,s);

figure(1)
subplot(1,3,1)
fplot(@(x) h.eval(x) , [-1,1]);
title('BSplines')
subplot(1,3,2)
fplot(@(x) h.evalDerivative(1,x) , [-1,1]);
title('BSplines first derivative')
subplot(1,3,3)
fplot(@(x) h.evalDerivative(2,x) , [-1,1]);
title('BSplines second derivative')

%% Dilated BSplines
s = 0; % degree 
l = 2; % resolution
b = 2; % base of dilation
h = DilatedBSplinesFunctionalBasis.withLevelBoundedBy(s,b,l);

figure(2)
clf
for k=1:cardinal(h)
    subplot(1,cardinal(h),k)
    fplot(@(x) h.basis.eval(h.indices(k,:),x) , [0,1]);
    title(['index #' num2str(k)])
end


%% Dilated BSplines
s = 1; % degree 
l = 3; % resolution
b = 2; % base of dilation
h = DilatedBSplinesFunctionalBasis.withLevelBoundedBy(s,b,l);

figure(2)
subplot(1,3,1)
fplot(@(x) h.eval(x) , [0,1]);
title('BSplines')
subplot(1,3,2)
fplot(@(x) h.evalDerivative(1,x) , [0,1]);
title('BSplines first derivative')
subplot(1,3,3)
fplot(@(x) h.evalDerivative(2,x) , [0,1]);
title('BSplines second derivative')



%% Interpolation with Bsplines

% BSplines basis
supp = [-1.1,1.1];
knots = linspace(supp(1),supp(2),200)'; % knots
s = 4; % degree
h = BSplinesFunctionalBasis(knots,s);

% Function to approximate
fun = @(x) cos(10*x);
fun = UserDefinedFunction(fun,1);
fun.evaluationAtMultiplePoints = true;
mu = LebesgueMeasure(-1,1);
xD = linspace(supp(1),supp(2),1000)';
% Interpolation points
points = interpolationPointsFeatureMap(h,xD);

% Interpolation of the function
f = h.interpolate(fun,points);

% Displays and error
fprintf('\nInterpolation using empirical interpolation points\n')
figure(1); clf;
fplot(@(x) [fun(x),f(x)],[-1,1])
legend('True function','Interpolation')
N = 100;
errL2 = testError(f,fun,N,mu);
fprintf('Mean squared error = %d\n',errL2)


%% Interpolation with Dilated Bsplines

% BSplines basis
s = 0; % degree 
l = 2; % resolution
b = 2; % base of dilation
h = DilatedBSplinesFunctionalBasis.withBoundedLevel(s,b,l);
%h = PiecewisePolynomialFunctionalBasis(linspace(0,1,10),s);
figure(10)
clf
fplot(@(x) full(h.eval(x)) , [0,1]);

% Function to approximate
fun = @(x) cos(10*x);
fun = UserDefinedFunction(fun,1);
fun.evaluationAtMultiplePoints = true;
mu = LebesgueMeasure(0,1);

% Interpolation points
xD = linspace(0,1,2000)';
points = interpolationPointsFeatureMap(h,xD);
points = interpolationPoints(h,xD);
points = random(mu,cardinal(h));
points = linspace(.12,.9,cardinal(h))';
muD = discretize(mu,1000);
hD = h.orthonormalize(muD);
dpp = ProjectionDPP(hD);
[points,output] = random(dpp);


% Interpolation of the function
f = h.interpolate(fun,points);

% Displays and error
fprintf('\nInterpolation using empirical interpolation points\n')
figure(4); 
clf;
fplot(@(x) [fun(x),f(x)],[0,1])
legend('True function','Interpolation')
N = 100;
errL2 = testError(f,fun,N,mu);
fprintf('Mean squared error = %d\n',errL2)


