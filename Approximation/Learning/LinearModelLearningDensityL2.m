% Class LinearModelLearningDensityL2

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

classdef LinearModelLearningDensityL2 < LinearModelLearning
    
    properties
        isBasisOrthonormal = true
    end
    
    methods
        function s = LinearModelLearningDensityL2()
            % LINEARMODELLEARNINGDENSITYL2- Constructor for the class LinearModelLearningDensityL2
            %
            % s = LINEARMODELLEARNINGDENSITYL2()
            % s: LINEARMODELLEARNINGDENSITYL2
            %
            % s.basis: FunctionalBasis
            % s.trainingData: training dataset
            % s.basisEval: evaluations of the basis on the training dataset
            % s.testError: logical indicating if a test error is evaluated 
            % s.testData: test dataset
            % s.basisEvalTest: evaluations of the basis on the test dataset
            % s.regularization: regularization (true or false), false by default
            % s.regularizationOptions: parameters for the regularization
            % s.modelSelection: model selection (true or false), true by default
            % s.modelSelectionOptions: parameters for model selection, struct('stopIfErrorIncrease',false) by default
            % s.errorEstimation: cross-validation error estimation (true or false), false by default
            % s.errorEstimationType: type of cross-validation procedure, 'leaveout' by default
            % s.basisAdaptation: basis adaptation (true or false), false by default
            % s.basisAdaptationPath: regularization path for basis adaptation
            % s.options: other options, struct() by default
            
            s@LinearModelLearning(DensityL2LossFunction)
        end
        
        function [x,output] = solve(s)
            % SOLVE - Solution of the L2 minimization problem
            %
            % [x,output] = SOLVE(s)
            % s: LinearModelLearningDensityL2
            % x: P-by-n double containing the coefficients
            % output.error: 1-by-n double containing the leave-one-out cross-validation error estimate of the risk
            
            s = initialize(s);
            
            if s.basisAdaptation
                [x,output] = s.solveBasisAdaptation();
            elseif s.regularization
                [x,output] = s.solveRegularized();
            else
                [x,output] = s.solveStandard();
            end
            
            if s.testError
                fEval = s.basisEvalTest * x;
                output.testError = s.lossFunction.testError(fEval, s.testData, norm(x)^2);
            end
            
            if ~isempty(s.basis)
                x = FunctionalBasisArray(x, s.basis);
            end
        end
    end
    
    methods (Hidden)
        function [a,output] = solveStandard(s)
            % Hypothesis of orthonormal basis
            if ~s.isBasisOrthonormal
                error('Only implemented for orthonormal bases.')
            end
            
            A = s.basisEval;
            
            a = (mean(A,1).');
            if iscell(s.trainingData) && length(s.trainingData) == 2 && ...
                    ~isempty(s.trainingData{2})
                b = s.trainingData{2};
                a = a - b;
            else
                b = [];
            end
            
            output = [];
            if s.errorEstimation && strcmpi(s.errorEstimationType,'leaveout')
                N = size(A,1);
                if isempty(b)
                    output.error = (-N^2)/(1-N)^2*norm(a)^2 + (2*N-1)/(N*(N-1)^2)*sum(A(:).^2);
                else
                    output.error = (N^2 - 2*N)/(N-1)^2*norm(a)^2 + 1/(N-1)^2*norm(b)^2 - 2/(N-1)^2*sum(A*b) + (2*N-1)/(N*(N-1)^2)*sum(A(:).^2) - 2/(N-1)*sum(A*a) + 2*a.'*b;
                end
            end
        end
        
        function [a, output] = solveRegularized(s)
            aStandard = solveStandard(s);
            
            A = s.basisEval;
            N = size(A,1);
            
            if iscell(s.trainingData) && length(s.trainingData) == 2 && ...
                    ~isempty(s.trainingData{2})
                b = s.trainingData{2};
            else
                b = [];
            end
            
            AsquareSum = sum(A.^2,1);
            
            [~,list] = sort(abs(aStandard),'descend');
            
            if isfield(s.regularizationOptions,'includedCoefficients')
                includedCoefficients = s.regularizationOptions.includedCoefficients;
                list = setdiff(list,includedCoefficients,'stable');
                
                aIncludedCoefficients = aStandard;
                aIncludedCoefficients(setdiff(1:length(aIncludedCoefficients),includedCoefficients)) = 0;
                if isempty(b)
                    b = s.trainingData{2};
                    errIncludedCoefficients = (N^2 - 2*N)/(N-1)^2*norm(aIncludedCoefficients)^2 + 1/(N-1)^2*norm(b(includedCoefficients))^2 - ...
                        2/(N-1)^2*sum(A(:,includedCoefficients)*b(includedCoefficients)) + (2*N-1)/(N*(N-1)^2)*sum(AsquareSum(includedCoefficients)) - ...
                        2/(N-1)*sum(A*aIncludedCoefficients) + 2*aIncludedCoefficients.'*b;
                else
                    errIncludedCoefficients = (-N^2)/(1-N)^2*norm(aIncludedCoefficients)^2 + (2*N-1)/(N*(N-1)^2)*sum(AsquareSum(includedCoefficients));
                end
            end
            
            if isempty(list)
                a = aStandard;
                output.error = NaN;
                return
            end
            
            n = length(list);
            aList = repmat({zeros(size(aStandard))},1,n);
            err = zeros(1,n);
            for i = 1:n
                I = list(1:i);
                if isfield(s.regularizationOptions,'includedCoefficients')
                    I = [includedCoefficients' ; I];
                end
                ared = aStandard(I);
                
                if isempty(b)
                    err(i) = (-N^2)/(1-N)^2*norm(ared)^2 + (2*N-1)/(N*(N-1)^2)*sum(AsquareSum(I));
                else
                    err(i) = (N^2 - 2*N)/(N-1)^2*norm(ared)^2 + 1/(N-1)^2*norm(b(I))^2 - 2/(N-1)^2*sum(A(:,I)*b(I)) + ...
                        (2*N-1)/(N*(N-1)^2)*sum(AsquareSum(I)) - 2/(N-1)*sum(A(:,I)*ared) + 2*ared.'*b(I);
                end
                aList{i}(I) = ared;
            end
            
            if isfield(s.regularizationOptions,'includedCoefficients')
                aList = [{aIncludedCoefficients} , aList];
                err = [errIncludedCoefficients , err];
            end
            
            [output.error,ind] = min(err);
            a = aList{ind};
            
            solpath = [aList{:}];
            pattern = solpath ~= 0;
            
            output.errorPath = err;
            output.ind = ind;
            output.pattern = pattern(:,ind);
            output.patternPath = pattern;
            output.optimalSolution = a;
            output.solutionPath = solpath;
        end
        
        function [a, output] = solveBasisAdaptation(s)
            P = size(s.basisEval,2);
            if isempty(s.basisAdaptationPath)
                solpath = true(P,P);
                solpath = triu(solpath);
            else
                solpath = s.basisAdaptationPath;
            end
            
            [a,output] = s.selectOptimalPath(solpath);
            output.flag = 2;
        end
        
        function [a,output] = selectOptimalPath(s,solpath)
            [aStandard, output] = s.solveStandard();
            
            A = s.basisEval;
            if iscell(s.trainingData) && length(s.trainingData) == 2 && ...
                    ~isempty(s.trainingData{2})
                b = s.trainingData{2};
            else
                b = [];
            end
            
            AsquareSum = sum(A.^2,1);
            if ~isempty(b)
                Asum = sum(A,1);
            end
            
            pattern = (solpath~=0);
            rep = all(~pattern,1);
            pattern(:,rep) = [];
            
            [~,I,~] = unique(pattern','rows','first');
            I = sort(I);
            pattern = pattern(:,I);
            
            n = size(A,1);
            m = size(pattern,2);
            a = cell(1,m);
            a(:) = {zeros(size(pattern,1),1)};
            err = zeros(1,m);
            
            I = cellfun(@nnz,num2cell(pattern,1)) > size(A,1);
            err(:,I) = Inf;
            pattern(:,I) = [];
            
            for i=1:size(pattern,2)
                ind = pattern(:,i);
                ared = aStandard(ind);
                
                if isempty(b)
                    err(i) = (-n^2)/(1-n)^2*(ared.'*ared) + (2*n-1)/(n*(n-1)^2)*sum(AsquareSum(ind));
                else
                    bred = b(ind);
                    err(i) = (n^2 - 2*n)/(n-1)^2*norm(ared)^2 + 1/(n-1)^2*norm(bred)^2 - 2/(n-1)^2*Asum(ind)*bred + (2*n-1)/(n*(n-1)^2)*sum(AsquareSum(ind)) - 2/(n-1)*Asum(ind)*ared + 2*ared.'*bred;
                end
                a{i}(ind) = ared;
                
                if i>1 && s.modelSelectionOptions.stopIfErrorIncrease && err(i)>2*err(i-1)
                    warning('stopIfErrorIncrease')
                    err(i+1:end) = Inf;
                    break
                end
            end
            
            apath = [a{:}];
            [~,ind] = min(err);
            
            a = apath(:,ind);
            output.error = err(ind);
            output.errorPath = err;
            output.ind = ind;
            output.pattern = pattern(:,ind);
            output.patternPath = pattern;
            output.optimalSolution = a;
            output.solutionPath = apath;
        end
    end
end