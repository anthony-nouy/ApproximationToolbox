% Class TensorLearning: learning with tensor formats

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

classdef TensorLearning < Learning
    
    properties
        % ORDER - Order of the tensor
        order
        % BASES - FunctionalBases
        bases
        % BASESEVAL - Cell of bases evaluations on the training sample
        basesEval
        % BASESEVALTEST - Cell of bases evaluations on the test sample
        basesEvalTest
        % ALGORITHM - Char specifying the choice of algorithm
        algorithm = 'standard'
        % INITIALIZATIONTYPE - Char specifying the type of initialization
        initializationType
        % INITIALGUESS - Initial guess for the learning algorithm
        initialGuess
        % TREEADAPTATION - Logical enabling or disabling the tree adaptation
        treeAdaptation = false
        % TREEADAPTATIONOPTIONS - Structure specifying the options for the tree adaptation
        treeAdaptationOptions = struct('tolerance',[],'maxIterations',100,'forceRankAdaptation',true)
        % RANKADAPTATION - Logical enabling or disabling the rank adaptation
        rankAdaptation = false
        % RANKADAPTATIONOPTIONS - Structure specifying the options for the rank adaptation
        rankAdaptationOptions = struct('maxIterations',10,'earlyStopping',false,'earlyStoppingFactor',10)
        % TOLERANCE - Structure specifying the options for the stopping criteria
        tolerance = struct('onError',1e-6,'onStagnation',1e-6)
        % LINEARMODELLEARNING - LinearModelLearning
        linearModelLearning
        % LINEARMODELLEARNINGPARAMETERS - Structure specifying the parameters for the linear model learning
        linearModelLearningParameters = struct('identicalForAllParameters',true)
        % ALTERNATINGMINIMIZATIONPARAMETERS - Structure specifying the options for the alternating minimization
        alternatingMinimizationParameters = struct('display',false,...
            'maxIterations',30,'stagnation',1e-6,'random',false)
        % BASESADAPTATIONPATH - Cell containing the adaptation paths
        basesAdaptationPath
        % STOREITERATES - Logical enabling or disabling the storage of the iterates
        storeIterates = true
        % RANK - 1-by-1 or 1-by-order integer
        rank = 1
        % OUTPUTDIMENSION - 1-by-1 integer indicating the number of outputs of the function to approximate
        outputDimension = 1
    end
    
    properties (Hidden)
        % NUMBEROFPARAMETERS - Integer
        numberOfParameters
        % EXPLORATIONSTRATEGY - 1-by-numberOfParameters integer: ordering for the optimization of each parameter
        explorationStrategy
        % WARNINGS - Structure
        warnings = struct('orthonormalityWarningDisplay', true, 'emptyBasesWarningDisplay', true);
    end
    
    methods
        function s = TensorLearning(varargin)
            % TENSORLEARNING - Constructor for the TensorLearning class
            %
            % s = TENSORLEARNING(loss)
            % loss: LossFunction
            % s: TENSORLEARNING
            
            s@Learning(varargin{:});
            s.linearModelLearning = linearModel(Learning(s.lossFunction));
        end
        
        function [f,output] = solve(s, varargin)
            % SOLVE - Solver for the learning problem with tensor formats
            %
            % [f,output] = SOLVE(s)
            % s: TensorLearning
            % f: FunctionalTensor
            % output: structure
            
            % If possible, deduce from trainingData the output dimension
            if iscell(s.trainingData) && length(s.trainingData) == 2 && ...
                    ~isempty(s.trainingData{2}) && isa(s.trainingData{2}, 'double')
                s.outputDimension = size(s.trainingData{2}, 2);
            end
            if s.outputDimension > 1 && (~isa(s, 'TreeBasedTensorLearning') || ...
                    ~isa(s.lossFunction, 'SquareLossFunction'))
                error(['Solver not implemented for vector-valued functions approximation, ', ...
                    'use TreeBasedTensorLearning with a SquareLossFunction instead.'])
            end
            
            if s.warnings.orthonormalityWarningDisplay && (isempty(s.bases) || ...
                    (~isempty(s.bases) && ~all(cellfun(@(x) x.isOrthonormal,s.bases.bases))))
                s.warnings.orthonormalityWarningDisplay = false;
                warning(['The implemented learning algorithms are designed ', ...
                    'for orthonormal bases. These algorithms work with ', ...
                    'non-orthonormal bases, but without some guarantees on their results.'])
            end
            
            % If no bases are provided, warn that the returned functions are
            % evaluated on the training data
            if isempty(s.bases) && s.warnings.emptyBasesWarningDisplay
                s.warnings.emptyBasesWarningDisplay = false;
                warning(['The returned functions are evaluated on the training data. ' ...
                    'To evaluate them at other points, assign to the FunctionalTensor a ' ...
                    'nonempty field bases and set the field evaluatedBases to false.'])
            end
            
            % If the test error cannot be computed, it is disabled
            if s.testError && ~isa(s.bases,'FunctionalBases') && isempty(s.basesEvalTest)
                warning('The test error cannot be computed.')
                s.testError = false;
            end
            
            % Assert if basis adaptation can be performed
            if isempty(s.basesAdaptationPath) && ~ismethod(s.bases, 'adaptationPath')
                if iscell(s.linearModelLearning) && ...
                        any(cellfun(@(x) x.basisAdaptation,s.linearModelLearning))
                    warning('Cannot perform basis adaptation, disabling it.')
                    s.linearModelLearning = cellfun(@(x) setfield(x, 'basisAdaptation', false), ...
                        s.linearModelLearning, 'UniformOutput', false);
                elseif ~iscell(s.linearModelLearning) && ...
                        s.linearModelLearning.basisAdaptation
                    warning('Cannot perform basis adaptation, disabling it.')
                    s.linearModelLearning.basisAdaptation = false;
                end
            end
            
            % Bases evaluation
            if ismethod(s.bases, 'eval')
                if ~isempty(s.trainingData) && isempty(s.basesEval)
                    if iscell(s.trainingData) && ~isempty(s.trainingData{1})
                        s.basesEval = s.bases.eval(s.trainingData{1});
                    elseif ~iscell(s.trainingData) && ~isempty(s.trainingData)
                        s.basesEval = s.bases.eval(s.trainingData);
                    else
                        error('Must provide input training data.')
                    end
                end
                
                if s.testError && ~isempty(s.testData) && isempty(s.basesEvalTest)
                    if iscell(s.testData) && ~isempty(s.testData{1})
                        s.basesEvalTest = s.bases.eval(s.testData{1});
                    elseif ~iscell(s.testData) && ~isempty(s.testData)
                        s.basesEvalTest = s.bases.eval(s.testData);
                    else
                        error('Must provide input test data.')
                    end
                end
            end
            s.basesEval = cellfun(@full,s.basesEval,'uniformoutput',false);
            if ~isempty(s.basesEvalTest)
                s.basesEvalTest = cellfun(@full,s.basesEvalTest,'uniformoutput',false);
            end
            
            if s.rankAdaptation
                if ~isfield(s.rankAdaptationOptions,'type')
                    [f,output] = solveAdaptation(s);
                elseif ischar(s.rankAdaptationOptions.type)
                    % Call the method corresponding to the asked rank adaptation option
                    str = lower(s.rankAdaptationOptions.type); str(1) = upper(str(1));
                    eval(['[f,output] = solve',str,'RankAdaptation(s,varargin{:});']);
                else
                    error('The rankAdaptationOptions property must be either empty or a string.')
                end
            elseif strcmpi(s.algorithm,'standard')
                [f,output] = solveStandard(s);
            else
                str = lower(s.algorithm); str(1) = upper(str(1));
                eval(['[f,output] = solve',str,'(s,varargin{:});']);
            end
        end 
    end
        
    methods (Hidden)
        function [f,output] = solveStandard(s)
            % SOLVESTANDARD - Solver for the learning problem with tensor formats using the standard algorithm (without adaptation)
            %
            % [f,output] = SOLVESTANDARD(s)
            % s: TensorLearning
            % f: FunctionalTensor
            % output: structure
            
            output.flag = 0;
            
            [s,f] = initialize(s); % Initialization
            f = FunctionalTensor(f,s.basesEval);
            
            % Replication of the LinearModelLearning objects
            if s.linearModelLearningParameters.identicalForAllParameters && length(s.linearModelLearning) == 1
                s.linearModelLearning = repmat({s.linearModelLearning},1,s.numberOfParameters);
            elseif length(s.linearModelLearning) ~= s.numberOfParameters
                error('Must provide numberOfParameters LinearModelLearning objects.')
            end
            
            % Working set paths
            if any(cellfun(@(x) x.basisAdaptation,s.linearModelLearning)) && ...
                    isempty(s.basesAdaptationPath)
                s.basesAdaptationPath = adaptationPath(s.bases);
            end
            
            if s.alternatingMinimizationParameters.maxIterations == 0
                return
            end
            
            % Alternating minimization loop
            for k = 1:s.alternatingMinimizationParameters.maxIterations
                [s,f] = preProcessing(s,f); % Pre-processing
                f0 = f;
                
                if s.alternatingMinimizationParameters.random
                    alphaList = randomizeExplorationStrategy(s); % Randomize the exploration strategy
                else
                    alphaList = s.explorationStrategy;
                end
                
                for alpha = alphaList
                    [s,A,b,f] = prepareAlternatingMinimizationSystem(s,f,alpha);
                    s.linearModelLearning{alpha}.trainingData = {[], b};
                    s.linearModelLearning{alpha}.basis = [];
                    s.linearModelLearning{alpha}.basisEval = A;
                    [C, outputLML] = s.linearModelLearning{alpha}.solve();
                    
                    if isempty(C(:)) || ~nnz(C(:)) || ~all(isfinite(C(:))) || any(isnan(C(:)))
                        warning('Empty, zero or NaN solution, returning to the previous iteration.')
                        output.flag = -2;
                        output.error = Inf;
                        break
                    end
                    
                    f = setParameter(s,f,alpha,C);
                end
                
                stagnation = stagnationCriterion(s,f,f0);
                output.stagnationIterations(k)=stagnation;
                
                if s.storeIterates
                    if isa(s.bases,'FunctionalBases')
                        output.iterates{k} = FunctionalTensor(f.tensor,s.bases);
                    else
                        output.iterates{k} = f;
                    end
                end
                
                if isfield(outputLML,'error')
                    output.error = outputLML.error;
                end
                
                if s.testError
                    fEvalTest = FunctionalTensor(f, s.basesEvalTest);
                    output.testError = s.lossFunction.testError(fEvalTest,s.testData);
                    output.testErrorIterations(k) = output.testError;
                end
                
                if s.alternatingMinimizationParameters.display
                    fprintf('\tAlt. min. iteration %i: stagnation = %.2d',k,stagnation)
                    if isfield(output,'error')
                        fprintf(', error = %.2d',output.error)
                    end
                    if s.testError
                        fprintf(', test error = %.2d',output.testError);
                    end
                    fprintf('\n')
                end
                
                if (k>1) && stagnation < s.alternatingMinimizationParameters.stagnation
                    output.flag = 1;
                    break
                end
            end
            
            if isa(s.bases,'FunctionalBases')
                f = FunctionalTensor(f.tensor,s.bases);
            end
            output.iter = k;
            
            if s.display
                if s.alternatingMinimizationParameters.display
                    fprintf('\n')
                end
                finalDisplay(s,f);
                if isfield(output,'error')
                    fprintf(', CV error = %.2d',output.error)
                end
                if isfield(output,'testError')
                    fprintf(', test error = %.2d',output.testError)
                end
                fprintf('\n')
            end
        end
        
        function [f,output] = solveAdaptation(s)
            % SOLVEADAPTATION - Solver for the learning problem with tensor formats using the adaptive algorithm
            %
            % [f,output] = SOLVEADAPTATION(s)
            % s: TensorLearning
            % f: FunctionalTensor
            % output: structure
            
            slocal = localSolver(s);
            slocal.display = false;
            
            flag = 0;
            treeAdapt = false;
            
            f = [];
            errors = zeros(1,s.rankAdaptationOptions.maxIterations);
            testErrors = zeros(1,s.rankAdaptationOptions.maxIterations);
            iterates = cell(1,s.rankAdaptationOptions.maxIterations);
            
            newRank = slocal.rank;
            enrichedNodes = [];
            
            for i = 1:s.rankAdaptationOptions.maxIterations
                slocal.bases = s.bases;
                slocal.basesEval = s.basesEval;
                slocal.basesEvalTest = s.basesEvalTest;
                slocal.trainingData = s.trainingData;
                slocal.testData = s.testData;
                slocal.rank = newRank;
                
                fOld = f;
                [f,outputLocal] = slocal.solve();
                if isfield(outputLocal,'error')
                    errors(i) = outputLocal.error;
                    if isinf(errors(i))
                        disp('Infinite error, returning the previous iterate.')
                        f = fOld;
                        i = i - 1;
                        flag = -2;
                        break
                    end
                end
                
                if s.testError
                    fEvalTest = FunctionalTensor(f, s.basesEvalTest);
                    testErrors(i) = s.lossFunction.testError(fEvalTest,s.testData);
                end
                
                if s.storeIterates
                    if isa(s.bases,'FunctionalBases')
                        iterates{i} = FunctionalTensor(f.tensor,s.bases);
                    else
                        iterates{i} = f;
                    end
                end
                
                if s.display
                    if s.alternatingMinimizationParameters.display
                        fprintf('\n')
                    end
                    fprintf('\nRank adaptation, iteration %i:\n',i)
                    adaptationDisplay(s,f,enrichedNodes);
                    
                    fprintf('\tStorage complexity = %i\n',storage(f.tensor))
                    
                    if errors(i) ~= 0
                        fprintf('\tError      = %.2d\n',errors(i))
                    end
                    if  testErrors(i) ~= 0
                        fprintf('\tTest Error = %.2d\n',testErrors(i))
                    end
                    
                    if s.alternatingMinimizationParameters.display
                        fprintf('\n')
                    end
                end
                
                if i == s.rankAdaptationOptions.maxIterations
                    break
                end
                
                if (s.testError && ...
                        testErrors(i) < s.tolerance.onError) || ...
                        (isfield(outputLocal,'error') && ...
                        errors(i) < s.tolerance.onError)
                    flag = 1;
                    break
                end
                
                if s.rankAdaptationOptions.earlyStopping && ...
                        i > 1 && ((s.testError && (isnan(testErrors(i)) || s.rankAdaptationOptions.earlyStoppingFactor*min(testErrors(1:i-1)) < testErrors(i))) || ...
                        (isfield(outputLocal,'error') && ( isnan(errors(i)) || s.rankAdaptationOptions.earlyStoppingFactor*min(errors(1:i-1)) < errors(i))))
                    fprintf('Early stopping')
                    if isfield(outputLocal,'error')
                        fprintf(', error = %d',errors(i))
                    end
                    if s.testError
                        fprintf(', test error = %d',testErrors(i))
                    end
                    fprintf('\n\n')
                    i = i-1;
                    f = fOld;
                    flag = -1;
                    break
                end
                
                adaptedTree = false;
                if slocal.treeAdaptation && i>1 && ...
                        (~s.treeAdaptationOptions.forceRankAdaptation || ~treeAdapt)
                    Cold = storage(f.tensor);
                    [s,f,output] = adaptTree(s,f,errors(i),[],output,i);
                    adaptedTree = output.adaptedTree;
                    if adaptedTree
                        if s.display
                            fprintf('\t\tStorage complexity before permutation = %i\n',Cold);
                            fprintf('\t\tStorage complexity after permutation  = %i\n',storage(f.tensor));
                        end
                        if s.testError
                            fEvalTest = FunctionalTensor(f, s.basesEvalTest);
                            testErrors(i) = s.lossFunction.testError(fEvalTest,s.testData);
                            if s.display
                                fprintf('\t\tTest error after permutation = %.2d\n',testErrors(i));
                            end
                        end
                        if s.alternatingMinimizationParameters.display
                            fprintf('\n')
                        end
                    end
                end
                
                if ~s.treeAdaptation || ~adaptedTree
                    if i>1 && ~treeAdapt
                        stagnation = stagnationCriterion(s,FunctionalTensor(f.tensor,s.basesEval),FunctionalTensor(fOld.tensor,s.basesEval));
                        if (stagnation < s.tolerance.onStagnation || isnan(stagnation))
                            break
                        end
                    end
                    treeAdapt = false;
                    [f,newRank,enrichedNodes,tensorForInitialization] = newRankSelection(s,f);
                    output.enrichedNodesIterations{i} = enrichedNodes;
                    slocal = initialGuessNewRank(s,slocal,tensorForInitialization,newRank);
                else
                    treeAdapt = true;
                    enrichedNodes = [];
                    newRank = f.tensor.ranks;
                    slocal.initializationType = 'initialguess';
                    slocal.initialGuess = f.tensor;
                end
            end
            
            if isa(s.bases,'FunctionalBases')
                f = FunctionalTensor(f.tensor,s.bases);
            end
            
            if s.storeIterates
                output.iterates = iterates(1:i);
            end
            output.flag = flag;
            if isfield(outputLocal,'error')
                output.errorIterations = errors(1:i);
                output.error = errors(i);
            end
            if s.testError
                output.testErrorIterations = testErrors(1:i);
                output.testError = testErrors(i);
            end
            if isfield(output, 'adaptedTree')
                output = rmfield(output, 'adaptedTree');
            end
        end
    end
    
    methods (Abstract)
        %% Standard solver methods
        
        % INITIALIZE - Initialization of the learning algorithm
        %
        % [s,f] = INITIALIZE(s)
        % s: TensorLearning
        % f: AlgebraicTensor
        [s,f] = initialize(s);
        
        % PREPROCESSING - Initialization of the alternating minimization algorithm
        %
        % [s,f] = PREPROCESSING(s,f)
        % s: TensorLearning
        % f: AlgebraicTensor
        [s,f] = preProcessing(s,f);
        
        % RANDOMIZEEXPLORATIONSTRATEGY - Randomization of the exploration strategy
        %
        % selmu = RANDOMIZEEXPLORATIONSTRATEGY(s)
        % s: TensorLearning
        % selmu: 1-by-s.numberOfParameters integer
        selmu = randomizeExplorationStrategy(s);
        
        % PREPAREALTERNATINGMINIMIZATIONSYSTEM - Preparation of the alternating minimization algorithm
        %
        % [s,A,b] = PREPAREALTERNATINGMINIMIZATIONSYSTEM(s,f,mu)
        % s: TensorLearning
        % f: FunctionalTensor
        % mu: 1-by-1 integer
        % A: n-by-numel(f.tensor.tensors{mu}) double
        % b: n-by-1 double
        [s,A,b] = prepareAlternatingMinimizationSystem(s,f,mu);
        
        % SETPARAMETER - Update of the parameter of the tensor
        %
        % f = SETPARAMETER(s,f,mu,a)
        % s: TensorLearning
        % f: FunctionalTensor
        % mu: 1-by-1 integer
        % a: numel(f.tensor.tensors{mu})-by-1 double
        f = setParameter(s,f,mu,a);
        
        % STAGNATIONCRITERION - Computation of the stagnation criterion
        %
        % stagnation = STAGNATIONCRITERION(s,f,f0)
        % Computes an indicator of the stagnation of the alternating minimization, using current and previous iterates f and f0
        % s: TensorLearning
        % f,f0: FunctionalTensor
        % stagnation: 1-by-1 double
        stagnation = stagnationCriterion(s,f,f0);
        
        % FINALDISPLAY - Display at the end of the computation
        %
        % FINALDISPLAY(s,f)
        % s: TensorLearning
        % f: FunctionalTensor
        finalDisplay(s,f);
        
        %% Rank adaptation solver methods
        
        % LOCALSOLVER - Extraction of the solver for the adaptive algorithm
        %
        % slocal = LOCALSOLVER(s)
        % s, slocal: TensorLearning
        slocal = localSolver(s);
        
        % NEWRANKSELECTION - Selection of a new rank in the adaptive algorithm
        %
        % [f,newRank,enrichedNodes,tensorForInitialization] = NEWRANKSELECTION(s,f)
        % s: TensorLearning
        % f: FunctionalTensor
        % newRank: 1-by-s.numberOfParameters integer
        % enrichedNodes: 1-by-N integer, with N the number of enriched nodes
        % tensorForInitialization: AlgebraicTensor
        [f,newRank,enrichedNodes,tensorForInitialization] = newRankSelection(s,f);
        
        % INITIALGUESSNEWRANK - Computation of the initial guess with the new selected rank
        %
        % slocal = INITIALGUESSNEWRANK(s,slocal,f,newRank)
        % s, slocal: TensorLearning
        % f: FunctionalTensor
        % newRank: 1-by-s.numberOfParameters integer
        slocal = initialGuessNewRank(s,slocal,f,newRank);
        
        % ADAPTATIONDISPLAY - Display during the adaptation
        %
        % ADAPTATIONDISPLAY(s,f,enrichedNodes)
        % s: TensorLearning
        % f: FunctionalTensor
        % enrichedNodes: 1-by-N integer, with N the number of enriched nodes
        adaptationDisplay(s,f,enrichedNodes);
    end
    
end