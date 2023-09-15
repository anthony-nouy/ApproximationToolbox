% Class FunctionalTensorPCALeastSquares

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

classdef FunctionalTensorPCALeastSquares < FunctionalTensorPCA
    
    properties
        optimalSampling = false
        optimalSamplingBoosted = false
        regressionTensorizedSampling = false
        regressionSamplingFactor = 1
        tol = 1e-10;
    end
    
    methods
        
        function FTPCA = FunctionalTensorPCALeastSquares()
            % s = FunctionalTensorPCALeastSquares()
        end
        
        function [projection, outputproj] = getalphaProjection(FTPCA,alpha, Valpha,gridalpha)
            ls = LinearModelLearningSquareLoss();
            ls.errorEstimation = true;
            if (numel(alpha) ==1 && FTPCA.basisAdaptation == true)
                alg = AdaptiveSparseTensorAlgorithm();
                alg.fullOutput = true;
                alg.maxIndex = [];
                alg.tol = 1e-9; % to be eventually changed
                alg.displayIterations = false;
                alg.adaptiveSampling = true;
                if FTPCA.optimalSampling == false
                    projection = @(f)alg.leastSquares(f,Valpha,ls);
                else
                    delta = 0.9;
                    eta = 0.01;
                    M = 10;
                    projection = @(f)alg.adaptBoostedLS(ls, f, Valpha, delta, eta, M,2);
                end
                % outputproj.numberOfEvaluations = size(gridalpha,1);
                if isa(Valpha , 'cell') || isa(Valpha , 'FunctionalBases')
                    Valpha = FullTensorProductFunctionalBasis(Valpha);
                end
            else
                if  FTPCA.optimalSampling == false
                    if isa(Valpha , 'cell') || isa(Valpha , 'FunctionalBases')
                        Valpha = FullTensorProductFunctionalBasis(Valpha);
                    end
                    if isa(Valpha , 'FullTensorProductFunctionalBasis')  && ~isa(gridalpha, 'double')
                        gridalpha = array(gridalpha);
                    end
                    W = eye(length(gridalpha));
                    A = Valpha.eval(gridalpha);
                    projection = @(f)wrappergetalphaProjection(ls, A, W, gridalpha, Valpha, f);
                else
                    if isa(gridalpha,'FullTensorGrid')
                        gridalpha = array(gridalpha);
                    end
                    if isa(Valpha , 'cell') || isa(Valpha , 'FunctionalBases')
                        Valpha = FullTensorProductFunctionalBasis(Valpha);
                    end
                    if isa(Valpha , 'FullTensorProductFunctionalBasis')  && ~isa(gridalpha, 'double')
                        gridalpha = array(gridalpha);
                    end
                    
                    W = diag(sqrt(cardinal(Valpha)*1./sum(Valpha.eval(gridalpha).^2,2)));
                    A = Valpha.eval(gridalpha);
                    projection = @(f)wrappergetalphaProjection(ls, A, W, gridalpha, Valpha, f);
                end
            end
            outputproj.numberOfEvaluations = size(gridalpha,1);
            outputproj.basis = Valpha;
        end
        
        function [gridalpha,nbsamples] = alphaGrid(FTPCA,alpha,X,basis,varargin)
            Xalpha = RandomVector(X.randomVariables(alpha));
            m = cardinal(basis);
            delta = 0.9;
            epsilon = 0.01;
            cdelta = -delta + (1+delta).*(log(1+delta));
            nb = log(2*m/epsilon)/cdelta*m;
            nbsamples = abs(ceil(nb*FTPCA.regressionSamplingFactor));
            if FTPCA.optimalSampling
                if FTPCA.optimalSamplingBoosted
                    % Resampling and conditionning
                    M = 10;
                    [gridalpha, dist] = conditionnedResampling(basis, delta, epsilon, M);
                    % Greedy removal of samples
                    [gridalpha, deltanew] = greedySubsamplingSmart(gridalpha, basis, delta);
                else
                    M = 10;
                    [gridalpha, dist] = conditionnedResampling(basis, delta, epsilon, M);
                     if dist > delta
                         warning('Bad sampling')
                     end
                end
            else
                gridalpha = random(Xalpha,nbsamples);
            end
            nbsamples = size(gridalpha,1);
        end
    end
end

function projection = wrappergetalphaProjection(ls, A, W, gridalpha, Valpha, f)
lsloc = ls;
lsloc.basisEval = W*A;
lsloc.trainingData = {[], W*f(gridalpha)};
lsloc.basis = Valpha;
[projection, output] = lsloc.solve();
end
