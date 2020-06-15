% Class LinearModelLearningSquareLoss

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

classdef LinearModelLearningSquareLoss < LinearModelLearning
    
    properties
        linearSolver = 'qr'
        weights = [];
        sharedCoefficients = true
    end
    
    methods
        function s = LinearModelLearningSquareLoss(weights)
            % LINEARMODELLEARNINGSQUARELOSS - Constructor for the class LinearModelLearningSquareLoss
            %
            % s = LINEARMODELLEARNINGSQUARELOSS(weights)
            % weights: N-by-1 double
            % s: LINEARMODELLEARNINGSQUARELOSS
            %
            % s.basis: FunctionalBasis
            % s.trainingData: training dataset
            % s.basisEval: evaluations of the basis on the training dataset
            % s.testError: logical indicating if a test error is evaluated 
            % s.testData: test dataset
            % s.basisEvalTest: evaluations of the basis on the test dataset
            % s.sharedCoefficients: when approximating a vector-valued function, indicates if the coefficients of the approximation are common to all the outputs, true by default
            % s.regularization: regularization (true or false), false by default
            % s.regularizationType: type of regularization ('l0', 'l1' or 'l2'), 'l1' by default
            % s.regularizationOptions: parameters for regularization, struct('lambda',0) by default
            % s.modelSelection: model selection (true or false), true by default
            % s.modelSelectionOptions: parameters for model selection, struct('stopIfErrorIncrease',false) by default
            % s.errorEstimation: cross-validation error estimation (true or false), false by default
            % s.errorEstimationType: type of cross-validation procedure ('leaveout' or 'kfold'), 'leaveout' by default
            % s.errorEstimationOptions.correction: correction for cross-validation procedure (true or false), true by default
            % s.errorEstimationOptions.numberOfFolds: number of folds (only if s.errorEstimationType = 'kfold'), min(10,N) by default, where N is the number of samples
            % s.errorEstimationOptions.gramMatrix: Gram matrix
            % s.linearSolver: solver for OLS problem ('\' or 'qr'), 'qr' by default
            % s.basisAdaptation: basis adaptation (true or false), false by default
            % s.basisAdaptationPath: regularization path for basis adaptation
            % s.options: other options, struct() by default
            %
            % When approximating a vector-valued function, setting 
            % s.sharedCoefficients to false will independently compute 
            % size(y,2) sets of coefficients, whereas setting it to true
            % will compute one set of coefficients shared across all the 
            % outputs. In that case, s.basisEval should be a (n-by-N-by-D)
            % double, with n the size of the dataset, N the number of basis
            % functions and D the number of outputs.
            
            s@LinearModelLearning(SquareLossFunction)
            
            if nargin == 1
                s.weights = weights;
            end
        end
        
        function [x,output] = solve(s)
            % SOLVE - Solution (Ordinary or Regularized) of the Least-Squares problem and cross-validation procedure
            %
            % [x,output] = SOLVE(s)
            % s: LinearModelLearningSquareLoss
            % x: P-by-n double containing the coefficients
            % output.error: 1-by-n double containing the (corrected) relative (leave-one-out or k-fold) cross-validation error estimates
            
            s = initialize(s);
            
            A = s.basisEval;
            y = s.trainingData{2};
            
            if s.sharedCoefficients
                if size(A, 3) ~= size(y,2)
                    error('size(A, 3) should be equal to size(y,2).')
                end
                A = permute(A, [1,3,2]);
                A = reshape(A, [], size(A, 3));
                y = y(:);
                s.basisEval = A;
                s.trainingData{2} = y;
            end
            n = size(y,2);
            
            if ~isempty(s.weights)
                W = spdiags(sqrt(s.weights),0,size(A,1),size(A,1));
                A = W*A;
                y = W*y;
            end
            
            if ~s.basisAdaptation
                if ~s.regularization
                    [x,output] = s.solveOLS();
                else
                    if n==1
                        [x,output] = s.solveRegularizedLS();
                    else
                        x = zeros(size(A,2),n);
                        for i=1:n
                            s.trainingData{2} = y(:,i);
                            [x(:,i),output_tmp] = s.solveRegularizedLS();
                            output.error(i) = output_tmp.error;
                            output.outputs{i} = output_tmp;
                        end
                    end
                end
            else
                if n == 1
                    [x,output] = s.solveBasisAdaptation();
                else
                    x = zeros(size(A,2),n);
                    for i=1:n
                        s.trainingData{2} = y(:,i);
                        [x(:,i),output_tmp] = s.solveBasisAdaptation();
                        output.error(i) = output_tmp.error;
                        output.outputs{i} = output_tmp;
                    end
                end
            end
 
            if s.testError
                if n == 1
                    fEval = s.basisEvalTest * x;
                    output.testError = s.lossFunction.testError(fEval, s.testData);
                else
                    output.testError = zeros(1, n);
                    fEval = s.basisEvalTest * x;
                    for i = 1:n
                        testData = {s.testData{1}, s.testData{2}(:, i)};
                        output.testError(i) = s.lossFunction.testError(fEval(:,i), testData);
                    end
                end
            end
            
            if ~isempty(s.basis)
                x = FunctionalBasisArray(x, s.basis, n);
            end
        end
    end
    
    methods (Hidden)
        function [x,output] = solveOLS(s)
            % SOLVEOLS - Solution of the Ordinary Least-Squares (OLS) problem and cross-validation procedure
            %
            % [x,output] = SOLVEOLS(s)
            % s: LinearModelLearningSquareLoss
            % x: P-by-n double containing the coefficients
            % output.error: 1-by-n double containing the (corrected) relative (leave-one-out or k-fold) cross-validation error estimates
            % output.delta: N-by-n double containing the residuals
            
            A = s.basisEval;
            y = s.trainingData{2};
            
            switch s.linearSolver
                case '\'
                    B = A'*A;
                    x = B\(A'*y);
                case 'qr'
                    [qA,rA] = qr(A,0);
                    x = rA\(qA'*y);
            end
            if s.errorEstimation
                switch s.errorEstimationType
                    case 'residuals'
                        output.delta = y-A*x;
                        err = sum(output.delta.^2,1)/length(y);
                        err = err./(moment(y,2,1)+mean(y,1).^2);
                        output.error = err;
                    otherwise
                        secondMoment = moment(y,2,1)+mean(y,1).^2;
                        switch lower(s.linearSolver)
                            case '\'
                                [output.error,output.delta] = s.computeCVError(y,secondMoment,A,x,'\',inv(B));
                            case 'qr'
                                [output.error,output.delta] = s.computeCVError(y,secondMoment,A,x,'qr',qA,rA);
                        end
                end
            end
            output.flag = 2;
        end
        
        function [x,output] = solveRegularizedLS(s)
            % SOLVEREGULARIZEDLS - Solution of the Regularized Least-Squares problem and cross-validation procedure
            %
            % [x,output] = SOLVEREGULARIZEDLS(s)
            % s: LinearModelLearningSquareLoss
            % x: P-by-1 double containing the coefficients
            % output: struct
            %
            % See also selectOptimalPath
            
            A = s.basisEval;
            y = s.trainingData{2};
            
            P = size(A,2);
            param = s.regularizationOptions;
            switch s.regularizationType
                case 'l0'
                    D = sqrt(sum(A.^2,1));
                    A_OMP = A*spdiags(1./D(:),0,length(D),length(D));
                    % A_OMP = A./repmat(D,[N,1]);
                    [x,solpath] = mexOMP(full(y),full(A_OMP),param);
                    x = spdiags(D(:),0,length(D),length(D))\x;
                    x = full(x);
                case 'l1'
                    [x,solpath] = mexLasso(full(y),full(A),param);
                case 'l2'
                    x = mexRidgeRegression(full(y),full(A),zeros(P,1),param);
                otherwise
                    error(['Regularization technique ' s.regularizationType ' not implemented'])
            end
            
            if s.modelSelection && exist('solpath','var') && norm(solpath)~=0
                if isfield(s.options,'nonzeroblocks')
                    rep = true(1,size(solpath,2));
                    rep = rep & any(solpath,1);
                    for k=1:length(s.options.nonzeroblocks)
                        Ik = s.options.nonzeroblocks{k};
                        rep = rep & any(solpath(Ik,:),1);
                    end
                    solpath = solpath(:,rep);
                end
                
                [x,output] = s.selectOptimalPath(solpath);
            elseif s.modelSelection
                warning('solpath does not exist or is empty')
            end
            
            if s.errorEstimation && ~exist('output','var')
                switch s.errorEstimationType
                    case 'residuals'
                        output.delta = y-A*x;
                        err = sum(output.delta.^2,1)/length(y);
                        err = err./(moment(y,2,1)+mean(y,1).^2);
                        output.error = err;
                    otherwise
                        secondMoment = moment(y,2,1)+mean(y,1).^2;
                        I = find(x);
                        A = A(:,I);
                        switch lower(s.linearSolver)
                            case '\'
                                B = A'*A;
                                [output.error,output.delta] = s.computeCVError(y,secondMoment,A,x(I),'\',inv(B));
                            case 'qr'
                                [qA,rA] = qr(A,0);
                                [output.error,output.delta] = s.computeCVError(y,secondMoment,A,x(I),'qr',qA,rA);
                        end
                end
            end
            output.flag = 2;
        end
        
        function [x,output] = solveBasisAdaptation(s)
            % SOLVEBASISADAPTATION - Solution of the Least-Squares problem with working-set and cross-validation procedure
            %
            % [x,output] = SOLVEBASISADAPTATION(s)
            % s: LinearModelLearningSquareLoss
            % x: P-by-1 double containing the coefficients
            % output: struct
            %
            % See also selectOptimalPath
            
            P = size(s.basisEval,2);
            if isempty(s.basisAdaptationPath)
                solpath = true(P,P);
                solpath = triu(solpath);
            else
                solpath = s.basisAdaptationPath;
            end
            
            [x,output] = s.selectOptimalPath(solpath);
            output.flag = 2;
        end
        
        function [x,output] = selectOptimalPath(s,solpath)
            % SELECTOPTIMALPATH - Selection of a solution using (corrected) relative (leave-one-out or k-fold) cross-validation error
            %
            % [x,output] = SELECTOPTIMALPATH(s,solpath)
            % s: LinearModelLearningSquareLoss
            % solpath: P-by-m double whose columns give m potential solutions with different sparsity patterns
            % x: P-by-1 double containing the optimal solution
            % output.error: 1-by-1 double containing the double containing the (corrected) relative (leave-one-out or k-fold) cross-validation error estimate
            % output.errorPath: 1-by-(m-1) double containing the regularization path of (corrected) relative (leave-one-out or k-fold) cross-validation error estimates
            % output.ind: 1-by-1 double giving the index of the optimal pattern
            % output.pattern: P-by-1 boolean indicating the optimal sparsity pattern
            % output.patternPath: P-by-(m-1) double containing the regularization path of sparsity patterns
            % output.solutionPath: P-by-(m-1) double containing the regularization path of potential non-zero solutions with different sparsity patterns
            % output.solutionPathOLS: P-by-(m-1) double containing the regularization path of OLS solutions
            % output.optimalSolution: P-by-1 double containing the optimal solution x
            % output.delta: N-by-1 double containing the optimal residual
            % output.deltaPath: N-by-(m-1) double containing the regularization paths of the residuals

            A = s.basisEval;
            y = s.trainingData{2};
            
            pattern = (solpath~=0);
            rep = all(~pattern,1);
            pattern(:,rep) = [];
            solpath(:,rep) = [];
            
            [~,I,~] = unique(pattern','rows','first');
            I = sort(I);
            solpath = solpath(:,I);
            pattern = pattern(:,I);
            
            m = size(pattern,2);
            x = cell(1,m);
            x(:) = {zeros(size(pattern,1),1)};
            delta = cell(1,m);
            err = zeros(1,m);
            linearSolvertype = s.linearSolver;
            
            if isfield(s.errorEstimationOptions,'gramMatrix')
                gramMatrix = s.errorEstimationOptions.gramMatrix;
            end
            
            secondMoment = moment(y,2,1)+mean(y,1).^2;
            
            I = sum(pattern~=0,1) > size(A,1);
            err(:,I) = [];
            delta(I) = [];
            pattern(:,I) = [];
            solpath(:,I) = [];
            
            if isempty(solpath)
                [x,output] = s.solveOLS();
                return
            end
            
            for i=1:size(pattern,2)
                ind = pattern(:,i);
                
                Ared = A(:,ind);
                if isfield(s.errorEstimationOptions,'gramMatrix')
                    s.errorEstimationOptions.gramMatrix = gramMatrix(ind,ind);
                end
                
                if size(Ared,2)>size(Ared,1)
                    s.linearSolver = '\';
                else
                    s.linearSolver = linearSolvertype;
                end
                
                switch lower(s.linearSolver)
                    case 'qr'
                        [qAred,rAred] = qr(Ared,0);
                        xred = rAred\(qAred'*y);
                        [err(i),delta{i}] = s.computeCVError(y,secondMoment,Ared,xred,'qr',qAred,rAred);
                        
                    case '\'
                        Cred = inv(Ared'*Ared);
                        xred = Cred*(Ared'*y);
                        [err(i),delta{i}] = s.computeCVError(y,secondMoment,Ared,xred,'\',Cred);
                end
                x{i}(ind) = xred;
                
                if i>1 && s.modelSelectionOptions.stopIfErrorIncrease && err(i)>2*err(i-1)
                    warning('stopIfErrorIncrease')
                    err(i+1:end) = Inf;
                    delta(i+1:end) = {Inf(size(y))};
                    break
                end
            end
            
            xpath = [x{:}];
            deltapath = [delta{:}];
            
            [~,ind] = min(err);
            
            x = xpath(:,ind);
            output.error = err(ind);
            output.errorPath = err;
            output.ind = ind;
            output.pattern = pattern(:,ind);
            output.patternPath = pattern;
            output.solutionPath = solpath;
            output.optimalSolution = xpath(:,ind);
            output.solutionPathOLS = xpath;
            output.delta = deltapath(:,ind);
            output.deltaPath = deltapath;
        end
        
        function [err,delta] = computeCVError(s,y,secondMoment,A,x,linearSolvertype,varargin)
            % COMPUTECVERROR - Relative cross-validation error estimates and residuals
            %
            % [err,delta] = COMPUTECVERROR(s,y,A,x,linearSolvertype,qA,rA)
            % s: LinearModelLearningSquareLoss
            % A: N-by-P double containing the evaluations of basis functions
            % y: N-by-n double containing the evaluations of response vector
            % x: P-by-n double containing the coefficients
            % output.error: 1-by-n double containing the (corrected) relative (leave-one-out or k-fold) cross-validation error estimates
            % output.delta: N-by-n double containing the residuals
            %
            % [err,delta] = COMPUTECVERROR(s,y,A,u,'qr',qA,rA)
            % qA,rA: factors of the QR decomposition of A (optional)
            %
            % [err,delta] = COMPUTECVERROR(s,y,A,u,'\',C)
            % C: inv(A'*A) (optional)
            %
            % s.errorEstimationType: type of cross-validation procedure ('leaveout' or 'kfold'), 'leaveout' by default
            %
            % For s.errorEstimationType = 'leaveout'
            % Compute the relative leave-one-out cross-validation error for the coefficients matrix x
            % using the fast leave-one-out cross-validation procedure [Cawlet & Talbot 2004] based on the Bartlett matrix inversion formula (special case of the Sherman-Morrison-Woodbury formula)
            % if s.errorEstimationOptions.correction = true, compute the corrected relative leave-one-out cross-validation error for the coefficients matrix x
            %
            % For s.errorEstimationType = 'kfold'
            % Compute the relative k-fold cross-validation error for the coefficients matrix x
            % using the fast k-fold cross-validation procedure based on the Sherman-Morrison-Woodbury formula
            % s.errorEstimationOptions.numberOfFolds: number of folds (only for the k-fold cross-validation procedure), min(10,N) by default where N is the number of samples
            % if s.errorEstimationOptions.correction = true, compute the corrected relative k-fold cross-validation error for the coefficients matrix x
            
            N = size(y,1);
            n = size(y,2);
            P = size(A,2);
            err = zeros(1,n);
            delta = zeros(size(y));
            
            if strcmp(s.errorEstimationType,'kfold')
                if ~isfield(s.errorEstimationOptions,'numberOfFolds')
                    s.errorEstimationOptions.numberOfFolds = min(10,N);
                end
                if s.errorEstimationOptions.numberOfFolds == N
                    s.errorEstimationType = 'leaveout';
                end
            end
            
            cv = s.errorEstimationType;
            
            if nargin<=5 && P>N
                C = inv(A'*A);
                linearSolvertype = '\';
            elseif nargin<=5
                [qA,rA] = qr(A,0);
                linearSolvertype = 'qr';
            else
                switch lower(linearSolvertype)
                    case 'qr'
                        qA = varargin{1};
                        rA = varargin{2};
                    case '\'
                        C = varargin{1};
                end
            end
            
            % Compute the absolute cross-validation error (cross-validation error estimate of the mean-squared error),
            % also called mean predicted residual sum of squares (PRESS) or empirical mean-squared predicted residual
            if strcmp(cv,'leaveout') % if leave-one-out cross-validation
                % Create a random partition of nearly equal size for leave-one-out cross-validation on N observations
                % cvp = cvpartition(N,'leaveout');
                % Check whether there are enough samples when performing ordinary least-squares minimization
                % if any(cvp.TrainSize<P)
                if N-1 < P
                    % warning('Not enough samples for performing OLS on the training set')
                    err(:) = Inf;
                    delta(:) = Inf;
                    return
                end
                % Compute the predicted residuals using the Bartlett matrix inversion formula
                % (special case of the Sherman-Morrison-Woodbury formula)
                switch lower(linearSolvertype)
                    case '\'
                        T = sum(A'.*(C*A'),1)';
                    case 'qr'
                        T = sum(qA.^2,2);
                end
                delta = (y-A*x)./(1-T);
                % Compute the absolute cross-validation error
                err = sum(delta.^2,1)/N;
            elseif strcmp(cv,'kfold') % if k-fold cross-validation
                % Create a random partition of nearly equal size for k-fold cross-validation on N observations
                k = s.errorEstimationOptions.numberOfFolds;
                cvp = cvpartition(N,'kfold',min(k,N));
                % Check whether there are enough samples when performing ordinary least-squares minimization
                if any(cvp.TrainSize<P)
                    % warning('Not enough samples for performing OLS on the training set.')
                    err(:) = Inf;
                    delta(:) = Inf;
                    return
                end
                errors = cell(cvp.NumTestSets,1);
                deltas = cell(cvp.NumTestSets,1);
                switch lower(linearSolvertype)
                    case '\'
                        H = A*C*A';
                    case 'qr'
                        H = qA*qA';
                end
                parfor i=1:cvp.NumTestSets % for each fold i
                    teIdx = cvp.test(i);
                    Atest = A(teIdx,:);
                    ytest = y(teIdx,:);
                    Htest = H(teIdx,teIdx);
                    % Compute predicted residual for fold i using Sherman-Morrison-Woodbury formula
                    deltas{i} = linsolve(eye(cvp.TestSize(i))-Htest,ytest-Atest*x);
                    % Compute absolute cross-validation error for fold i
                    errors{i} = sum(deltas{i}.^2,1)/cvp.TestSize(i);
                end
                for i=1:cvp.NumTestSets
                    teIdx = cvp.test(i);
                    delta(teIdx) = deltas{i};
                end
                % Average over k = cvp.NumTestSets folds
                err = sum(cell2mat(errors),1)/cvp.NumTestSets;
            else
                error(['Cross-validation ' cv ' not implemented'])
            end
            
            % Compute the relative cross-validation error
            err = err./secondMoment; % err is divided by empirical second moment of y
            
            % Compute the corrected relative cross-validation error to reduce the sensitivity of the error estimate to overfitting
            % (the non-corrected cross-validation error estimate underpredicts the error in L^2-norm (generalization error))
            if strcmpi(linearSolvertype, 'qr') && rcond(rA) < eps
                s.errorEstimationOptions.correction = false;
            end
            if s.errorEstimationOptions.correction
                if N~=P
                    % corr = (N-1)/(N-P-1); % = (1-1/N)*(1-P/N-1/N)^(-1) % Adjusted Empirical Error (AEE) -> only accurate when N >> P (corr ~ 1+P/N when N->Inf)
                    % corr = (N+P)/(N-P); % = (1+P/N)*(1-P/N)^(-1) % Future Prediction Error (FPE) [Akaike, 1970] -> only accurate when N >> P (corr ~ 1+2*P/N when N->Inf)
                    % corr = max(0,(1-1*sqrt((P*(log(N/P)+1)-3)/N)))^(-1); % Uniform Convergence Bounds (UCB) [Cherkassky, Mulier, & Vapnik, 1997]
                    switch lower(linearSolvertype)
                        case 'qr'
                            invrA = inv(rA);
                            C = invrA*invrA.';
                    end
                    
                    if isfield(s.errorEstimationOptions,'gramMatrix')
                        % Direct Eigenvalue Estimator (DEE) in the case of
                        % non orthonormal bases [Chapelle, Vapnik & Bengio, 2002], [Blatman & Sudret, 2011]
                        % The Gram matrix must be provided. Independence between variables is assumed
                        corr = (N/(N-P))*(1+trace(C*s.errorEstimationOptions.gramMatrix)); % Correcting term
                    else
                        corr = (N/(N-P))*(1+trace(C)); % = (1-P/N)^(-1)*(1+trace(C)) % Direct Eigenvalue Estimator [Chapelle, Vapnik & Bengio, 2002], [Blatman & Sudret, 2011] -> accurate even when N is not >> P (corr ~ 1+2*P/N when N->Inf, as trace(C) ~ P/N when N->Inf)
                    end
                    
                    if corr > 0
                        err = err*corr;
                    end
                end
            end
            err = sqrt(err);
        end
    end
end