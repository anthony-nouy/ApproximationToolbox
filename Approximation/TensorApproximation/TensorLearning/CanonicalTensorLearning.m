% Class CanonicalTensorLearning: learning with canonical tensor formats

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

classdef CanonicalTensorLearning < TensorLearning
    
    properties
        % TRUNCATEINITIALIZATION - Logical: initialization truncation
        truncateInitialization = true
    end
    
    methods
        function s = CanonicalTensorLearning(d,varargin)
            % CANONICALTENSORLEARNING - Constructor for the class CanonicalTensorLearning
            %
            % s = CANONICALTENSORLEARNING(d,loss)
            % d: 1-by-1 integer (order)
            % loss: LossFunction
            % s: CANONICALTENSORLEARNING
            
            s@TensorLearning(varargin{:});
            s.order = d;
            s.initializationType = 'mean';
        end
        
        %% Standard solver methods
        
        function [s,f] = initialize(s)
            if s.treeAdaptation
                warning('treeAdaptation not defined for CanonicalTensorLearning.')
                s.treeAdaptation = false;
            end
            
            if isfield(s.alternatingMinimizationParameters,'oneByOneFactor') && s.alternatingMinimizationParameters.oneByOneFactor && s.rank ~= 1
                s.explorationStrategy = 1:(s.alternatingMinimizationParameters.innerLoops * s.rank * s.order + 1);
                s.numberOfParameters = length(s.explorationStrategy);
            else
                s.alternatingMinimizationParameters.oneByOneFactor = false;
                s.explorationStrategy = 1:s.order + 1;
                s.numberOfParameters = s.order + 1;
            end
            s.linearModelLearning(s.numberOfParameters+1:end) = [];
            
            sz = cellfun(@(x)size(x,2),s.basesEval)';
            switch lower(s.initializationType)
                case 'random'
                    f = CanonicalTensor.randn(s.rank,sz);
                case 'ones'
                    f = CanonicalTensor.ones(s.rank,sz);
                case 'initialguess'
                    f = s.initialGuess;
                case {'mean','meanrandomized'}
                    if ~iscell(s.trainingData) || ...
                            (iscell(s.trainingData) && length(s.trainingData) == 1)
                        error('Initialization type not implemented in unsupervised learning.')
                    end
                    if isa(s.bases,'FunctionalBases')
                        os = mean(s.bases);
                    else
                        os = cell(s.order,1);
                        for mu=1:s.order
                            os{mu} = full(mean(s.basesEval{mu},1)');
                        end
                    end
                    
                    if strcmp(s.initializationType,'meanrandomized')
                        for mu=1:s.order
                            os{mu} = os{mu} + 0.01*randn(size(os{mu}));
                        end
                    end
                    f = CanonicalTensor(TSpaceVectors(os),mean(s.trainingData{2}));
                case 'greedy'
                    sini = s;
                    sini.rankAdaptation = false;
                    sini.algorithm = 'greedy';
                    sini.initializationType = 'mean';
                    sini.alternatingMinimizationParameters.display = false;
                    sini.linearModelLearning.errorEstimation = false;
                    sini.testError = false;
                    sini.display = false;
                    [f,outputini] = sini.solve();
                    
                    if s.display && isfield(outputini,'error')
                        fprintf('Greedy initialization: rank = %d, error = %d\n',numel(f.core.data),outputini.error);
                    end
                otherwise
                    error('bad initializationType')
            end
            if isa(f,'FunctionalTensor')
                f = f.tensor;
            end
            
            if s.rank>numel(f.core.data)
                fx = evalDiag(timesMatrix(f,s.basesEval));
                sini = s;
                sini.algorithm = 'Standard';
                sini.rankAdaptation = false;
                sini.rank = s.rank-numel(f.core.data);
                sini.initializationType = 'greedy';
                sini.display = false;
                sini.alternatingMinimizationParameters.display = false;
                
                if isa(s.lossFunction,'SquareLossFunction')
                    R = s.trainingData{2} - fx;
                elseif isa(s.lossFunction,'DensityL2LossFunction')
                    R = f;
                end
                if ~iscell(sini.trainingData)
                    sini.trainingData = {sini.trainingData};
                end
                sini.trainingData{2} = R;
                
                fadd = sini.solve();
                if isa(fadd,'FunctionalTensor')
                    f = f+fadd.tensor;
                else
                    f = f+fadd;
                end
            end
            
            if s.truncateInitialization
                if s.order==2
                    Tr = Truncator();
                    f = Tr.truncate(f);
                    f = CanonicalTensor(f.space,f.core.data);
                    s.rank = length(f.core.data);
                end
            end
        end
        
        function [s,f] = preProcessing(s,f)
            
        end
        
        function selmu = randomizeExplorationStrategy(s)
            if ~isfield(s.alternatingMinimizationParameters,'oneByOneFactor') || ~s.alternatingMinimizationParameters.oneByOneFactor
                selmu = [randperm(s.numberOfParameters-1) s.numberOfParameters];
            else
                selmu = randperm(s.order);
                seli = randperm(s.rank);
                strategy = reshape(s.explorationStrategy(1:end-1),[],s.order);
                strategy(:,selmu) = strategy;
                strategy = reshape(reshape(strategy,1,[]),s.rank,[]);
                strategy(seli,:) = strategy;
                selmu = [reshape(strategy,1,[]) s.numberOfParameters];
            end
        end
        
        function [s,A,b,f] = prepareAlternatingMinimizationSystem(s,f,mu)
            if ~isa(s.lossFunction, 'SquareLossFunction')
                error('Method not implemented for this loss function.')
            end
            
            y = s.trainingData{2};
            N = size(s.basesEval{1},1);
            
            if mu ~= s.numberOfParameters
                if s.alternatingMinimizationParameters.oneByOneFactor
                    ind = mu;
                    mu = ceil(ind / s.alternatingMinimizationParameters.innerLoops / s.rank);
                    i = ind - s.rank * (ceil(ind / s.rank) - 1);
                    
                    fH = timesMatrix(f.tensor,s.basesEval);
                    fHmu = ones(N,s.rank);
                    nomu = 1:f.tensor.order;
                    nomu(mu)=[];
                    for nu=nomu
                        fHmu = fHmu.*fH.space.spaces{nu};
                    end
                    
                    B = fHmu.*fH.space.spaces{mu};
                    noi = 1:s.rank;
                    noi(i)=[];
                    anoi = fH.core.data(noi);
                    b = y-B(:,noi)*anoi;
                    
                    A = s.basesEval{mu}.*repmat(fHmu(:,i),1,f.tensor.sz(mu));
                    if s.linearModelLearning{mu}.basisAdaptation
                        s.linearModelLearning{mu}.basisAdaptationPath=s.basesAdaptationPath{mu};
                    end
                else
                    A = permute(parameterGradientEval(f,mu),[1 3 2]);
                    A = A.data(:,:);
                    
                    if s.linearModelLearning{mu}.basisAdaptation
                        s.linearModelLearning{mu}.basisAdaptationPath = repmat(s.basesAdaptationPath{mu},s.rank,1);
                    else
                        if s.rank>1
                            s.linearModelLearning{mu}.options.nonzeroblocks = cell(1,s.rank);
                            for kk=1:s.rank
                                s.linearModelLearning{mu}.options.nonzeroblocks{kk}=(kk-1)*f.tensor.sz(mu)+(1:f.tensor.sz(mu));
                            end
                        end
                    end
                    b = y;
                end
            else
                if s.alternatingMinimizationParameters.oneByOneFactor
                    mu = s.order + 1;
                end
                A = parameterGradientEval(f,mu);
                A = A.data(:,:);
                b = y;
                s.linearModelLearning{mu}.basisAdaptation = false;
            end
        end
        
        function f = setParameter(s,f,mu,a)
            if mu ~= s.numberOfParameters
                if ~s.alternatingMinimizationParameters.oneByOneFactor
                    a = reshape(a,[f.tensor.sz(mu),s.rank]);
                    norma = sqrt(sum(a.^2,1));
                    I = (norma~=0);
                    if ~all(I)
                        warning('degenerate case: one factor is zero')
                    end
                    a(:,I) = a(:,I)*diag(1./norma(I));
                    f.tensor.space.spaces{mu} = a;
                    f.tensor.core.data = norma(:);
                    
                    if length(f.tensor.space.spaces) == 2 && mu==1
                        [f.tensor.space.spaces{mu},~] = qr(f.tensor.space.spaces{mu},0);
                    end
                else
                    ind = mu;
                    mu = ceil(ind / s.alternatingMinimizationParameters.innerLoops / s.rank);
                    i = ind - s.rank * (ceil(ind / s.rank) - 1);
                    norma = norm(a);
                    a = a/norma;
                    f.tensor.space.spaces{mu}(:,i) = a;
                    f.tensor.core.data(i) = norma;
                end
            else
                f.tensor.core.data = a;
            end
        end
        
        function stagnation = stagnationCriterion(s,f,f0)
            nf = norm(f.tensor);
            nf0 = norm(f0.tensor);
            stagnation = 2*abs(nf-nf0)/(nf+nf0);
        end
        
        function finalDisplay(s,f)
            fprintf('Rank = %d',length(f.tensor.core.data));
        end
        
        %% Rank adaptation solver methods
        
        function slocal = localSolver(s)
            slocal = s;
            slocal.initializationType = 'mean';
            slocal.algorithm = 'standard';
            slocal.rankAdaptation = false;
            slocal.modelSelection = false;
        end
        
        function [f,newRank, enrichedNodes, tensorForInitialization] = newRankSelection(s,f)
            newRank = length(f.tensor.core.data) + 1;
            enrichedNodes = 1;
            tensorForInitialization = f;
        end
        
        function slocal = initialGuessNewRank(s,slocal,f,varargin)
            slocal.initializationType = 'initialGuess';
            slocal.initialGuess = f.tensor;
        end
        
        function adaptationDisplay(s,f,varargin)
            fprintf('\tRank = %d\n',length(f.tensor.core.data));
        end
        
        %% Greedy solver
        
        function [f,output] = solveGreedy(s)
            % SOLVEGREEDY - Greedy solver
            %
            % [f,output] = SOLVEGREEDY(s)
            % s: CanonicalTensorLearning
            % f: FunctionalTensor
            % output: structure
            
            if ~isa(s.lossFunction, 'SquareLossFunction')
                error('Method not implemented for this loss function.')
            end

            H = s.basesEval;
            if s.linearModelLearning.basisAdaptation && isempty(s.basesAdaptationPath)
                if ismethod(s.bases, 'adaptationPath')
                    s.basesAdaptationPath = adaptationPath(s.bases);
                else
                    warning('Cannot perform basis adaptation, disabling it.')
                    s.linearModelLearning = cellfun(@(x) setfield(x, 'basisAdaptation', false), ...
                        s.linearModelLearning, 'UniformOutput', false);
                end
            end
            
            y = s.trainingData{2};
            
            slocal = s;
            slocal.algorithm = 'standard';
            slocal.rank = 1;
            slocal.display = false;
            slocal.testError = false;
            slocal.modelSelection = false;
            
            f = CanonicalTensor.zeros(0,cellfun(@(x)size(x,2),H)');
            f0 = f;
            stagnation = zeros(1,s.rank);
            
            if iscell(slocal.linearModelLearning)
                lslocal = slocal.linearModelLearning{s.numberOfParameters};
            else
                lslocal = slocal.linearModelLearning;
            end
            
            iserror = 0;
            err = ones(1,s.rank);
            puregreedy = 0;
            if ~puregreedy
                if iscell(slocal.linearModelLearning)
                    slocal.linearModelLearning = cellfun(@(x) setfield(x,'errorEstimation',false),s.linearModelLearning,'UniformOutput',false);
                else
                    slocal.linearModelLearning.errorEstimation = false;
                end
            end
            if ~iscell(slocal.trainingData)
                slocal.trainingData = {slocal.trainingData};
            end
            for k = 1:s.rank
                slocal.trainingData{2} = y-evalDiag(timesMatrix(f,H)); % Residual
                [fadd,outputgreedy] = slocal.solve();
                if isa(fadd,'FunctionalTensor')
                    fadd = fadd.tensor;
                end
                f = f+fadd;
                
                if ~puregreedy
                    fH = timesMatrix(f,H);
                    A = ones(numel(y),numel(fH.core.data));
                    for nu=1:length(H)
                        A = A.*fH.space.spaces{nu};
                    end
                    lslocal.basisAdaptation=false;
                    lslocal.basis = [];
                    lslocal.basisEval = A;
                    lslocal.trainingData = {[], y};
                    [a,outputgreedy] = lslocal.solve();
                    f.core.data = a(:);
                    
                end
                stagnation(k) = 2*norm(f-f0)/(norm(f)+norm(f0));
                currentrank = length(f.core.data);
                output.sequence{k} = f;
                if isfield(outputgreedy,'error')
                    err(k) = outputgreedy.error;
                    iserror = 1;
                    if s.display
                        fprintf('Alternating minimization (greedy): rank = %d, error = %d, stagnation = %d\n',currentrank,err(k),stagnation(k))
                    end
                else
                    if s.display
                        fprintf('Alternating minimization (greedy): rank = %d, stagnation = %d\n',currentrank,stagnation(k))
                    end
                end
                
                if err(k)<s.tolerance.onError || stagnation(k)<s.tolerance.onStagnation || ...
                        (k>2 && err(k)> err(k-1) && err(k-1)> err(k-2))
                    break
                end
                
                f0 = f;
                if s.testError
                    fEvalTest = FunctionalTensor(f,s.basesEvalTest);
                    output.testError = s.lossFunction.testError(fEvalTest(),s.testData);
                    output.testErrorIterations(k) = output.testError;
                    if s.display
                        fprintf('Greedy: iteration #%d, test error = %d\n',k,output.testError)
                    end
                end
            end
            
            output.stagnation = stagnation(1:k);
            if iserror
                output.errors = err(1:k);
                [~,K] = min(output.errors);
                f = output.sequence{K};
                output.selectedIterate = K;
                output.error = output.errors(K);
            end
            
            if isa(s.bases,'FunctionalBases')
                f = FunctionalTensor(f,s.bases);
            end
        end
    end
end