% Class TensorTrainTensorLearning: learning with tensor-train formats

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

classdef TensorTrainTensorLearning < TensorLearning
    
    properties
        % MAXRANKVARIATION - Integer: maximum rank variation
        maxRankVariation = 100
    end
    
    methods
        function s = TensorTrainTensorLearning(d,varargin)
            % TENSORTRAINTENSORLEARNING - Constructor for the class TensorLearning
            %
            % s = TENSORTRAINTENSORLEARNING(d,loss)
            % d: 1-by-1 integer (order)
            % loss: LossFunction
            % s: TENSORTRAINTENSORLEARNING
            
            s@TensorLearning(varargin{:});
            s.order = d;
            s.numberOfParameters = d;
            s.initializationType = 'canonical';
            s.rankAdaptationOptions.rankOneCorrection = true;
            s.rankAdaptationOptions.theta = 0.8;
        end
        
        %% Standard solver methods
        function [s,f] = initialize(s)
            s.explorationStrategy = 1:s.order;
            if numel(s.rank)==1
                s.rank = repmat(s.rank,1,s.order-1);
            end
            switch lower(s.initializationType)
                case 'mean'
                    if ~iscell(s.trainingData) || ...
                            (iscell(s.trainingData) && length(s.trainingData) == 1)
                        error('Initialization type not implemented in unsupervised learning.')
                    end
                    os = mean(s.bases);
                    f = CanonicalTensor(TSpaceVectors(os),mean(s.trainingData{2}));
                    s.rank = ones(1,s.order-1);
                case 'canonical'
                    f = canonicalInitialization(s,max(s.rank));
                case 'initialguess'
                    f = s.initialGuess;
                case 'random'
                    sz = numel(s.bases);
                    f = TTTensor.randn(sz,s.rank);
                otherwise
                    error('Bad initialization.');
            end
            f = TTTensor(f);
            
            if any(f.ranks<s.rank) && ~strcmpi(s.initializationType,'canonical')
                r = max(s.rank-f.ranks);
                f = canonicalCorrection(s,f,r);
            end
            if ~all(f.ranks==s.rank) && all(f.ranks>=s.rank)
                trunc = Truncator('tolerance',0,'maxRank',s.rank);
                f = trunc.truncate(f);
            end
            if ~all(f.ranks==s.rank)
                warning('Initialization with undesired ranks.')
            end
        end
        
        function [s,f] = preProcessing(s,f)
            f.tensor = orth(f.tensor,1);
        end
        
        function selmu = randomizeExplorationStrategy(s)
            % Never randomize for the TT format, because of the orthonormalization procedure
            warning('Cannot randomize the exploration strategy, keeping it on default.')
            selmu = s.explorationStrategy;
        end
        
        function [s,A,b,f] = prepareAlternatingMinimizationSystem(s,f,mu)
            if ~isa(s.lossFunction, 'SquareLossFunction')
                error('Method not implemented for this loss function.')
            end
            
            g = parameterGradientEval(f,mu);
            A = g.data(:,:);
            b = s.trainingData{2};
            
            if s.linearModelLearning{mu}.basisAdaptation
                sz = f.tensor.cores{mu}.sz;
                s.linearModelLearning{mu}.basisAdaptationPath = createBasisAdaptationPath(s,s.basesAdaptationPath{mu},sz(1),sz(3));
            end
        end
        
        function f = setParameter(s,f,mu,a)
            f.tensor.cores{mu} = FullTensor(a,3,f.tensor.cores{mu}.sz);
            if mu<length(s.basesEval)
                [f.tensor.cores{mu},~] = orth(f.tensor.cores{mu},3);
            end
            f.tensor.isOrth=false;
        end
        
        function stagnation = stagnationCriterion(s,f,f0)
            stagnation = norm((f.tensor-f0.tensor))/(norm(f0.tensor)+norm(f.tensor));
        end
        
        function finalDisplay(s,f)
            fprintf('Ranks = [ %s ]',num2str(f.tensor.ranks));
        end
        
        function f = canonicalCorrection(s,f,r)
            % CANONICALCORRECTION - Rank-r canonical correction
            %
            % f = CANONICALCORRECTION(s,f,r)
            % s: TensorTrainTensorLearning
            % f: TTTensor
            % r: 1-by-1 integer
            
            fx = timesMatrix(f,s.basesEval);
            fx = evalDiag(fx);
            
            if isa(s.lossFunction,'SquareLossFunction')
                R = s.trainingData{2} - fx;
            elseif isa(s.lossFunction,'DensityL2LossFunction')
                R = f;
            end
            if ~iscell(s.trainingData)
                s.trainingData = {s.trainingData};
            end
            s.trainingData{2} = R;
            
            fadd = canonicalInitialization(s,r);
            f = f + TTTensor(fadd);
        end
        
        function f = canonicalInitialization(s,r)
            % CANONICALINITIALIZATION - Rank-r canonical initialization
            %
            % f = CANONICALINITIALIZATION(s,r)
            % s: TensorTrainTensorLearning
            % r: 1-by-1 integer
            % f: TTTensor
            
            C = CanonicalTensorLearning(s.order,s.lossFunction);
            if iscell(s.linearModelLearning)
                C.linearModelLearning = s.linearModelLearning{1};
            else
                C.linearModelLearning = s.linearModelLearning;
            end
            C.modelSelection = false;
            C.alternatingMinimizationParameters = s.alternatingMinimizationParameters;
            C.bases = s.bases;
            C.basesEval = s.basesEval;
            C.trainingData = s.trainingData;
            C.display = false;
            C.rankAdaptation = true;
            C.alternatingMinimizationParameters.display = false;
            C.initializationType = 'mean';
            C.alternatingMinimizationParameters.maxIterations = 10;
            C.rankAdaptationOptions.maxIterations = r;
            C.warnings = s.warnings;
            f = C.solve();
            if isa(f,'FunctionalTensor')
                f = f.tensor;
            end
        end
        
        function p = createBasisAdaptationPath(s,pmu,rleft,rright)
            % CREATEBASISADAPTATIONPATH - Creation of the basis adaptation path
            %
            % p = CREATEBASISADAPTATIONPATH(s,pmu,rleft,rright)
            % s: TensorTrainTensorLearning
            % pmu, p: logical matrix (adaptation paths)
            % rleft, rright: 1-by-1 integer
            
            p = permute(pmu,[3,1,4,2]);
            p = repmat(p,[rleft,1,rright,2]);
            p = reshape(p,[rleft*size(p,2)*rright,size(p,4)]);
        end
        
        %% Rank adaptation solver methods
        
        function slocal = localSolver(s)
            slocal = s;
            slocal.rankAdaptation = false;
            slocal.storeIterates = false;
            slocal.initializationType = 'canonical';
            slocal.modelSelection = false;
        end
        
        function [f,newRank,enrichedNodes,tensorForInitialization] = newRankSelection(s,f)
            if s.rankAdaptationOptions.rankOneCorrection
                stemp = s;
                stemp.initializationType = 'mean';
                tensorforselection = canonicalCorrection(stemp,f.tensor,1);
            else
                tensorforselection = f.tensor;
            end
            sv=singularValues(tensorforselection);
            svmin = cellfun(@min,sv(1:end-1));
            
            enrichedNodes = find(svmin>=s.rankAdaptationOptions.theta*max(svmin));
            newRank = f.tensor.ranks;
            newRank(enrichedNodes) = newRank(enrichedNodes)+1;
            tensorForInitialization = f;
        end
        
        function slocal = initialGuessNewRank(s,slocal,f,newRank)
            slocal.initializationType = 'initialguess';
            slocal.initialGuess = enrichedEdgesToRanks(s,f.tensor,newRank);
        end
        
        function adaptationDisplay(s,f,enrichedNodes)
            fprintf('\tEnriched nodes: [ %s ]\n\tRanks = [ %s ]\n',num2str(enrichedNodes(:).'),num2str(f.tensor.ranks));
        end
        
        function f = enrichedEdgesToRanks(s,f,newRanks)
            % ENRICHEDEDGESTORANKS - Enrichment of a subset of ranks of the tensor
            %
            % f = ENRICHEDEDGESTORANKS(s,f,newRanks)
            % s: TensorTrainTensorLearning
            % f: TensorTrain
            % newRanks: 1-by-length(f.ranks) integer
            
            if ~isa(s.lossFunction, 'SquareLossFunction')
                error('Method not implemented for this loss function.')
            end
            y = s.trainingData{2};
            
            if s.linearModelLearning.basisAdaptation && isempty(s.basesAdaptationPath)
                if ismethod(s.bases, 'adaptationPath')
                    s.basesAdaptationPath = adaptationPath(s.bases);
                else
                    warning('Cannot perform basis adaptation, disabling it.')
                    s.linearModelLearning = cellfun(@(x) setfield(x, 'basisAdaptation', false), ...
                        s.linearModelLearning, 'UniformOutput', false);
                end
            end
            
            H = cellfun(@full,s.basesEval,'uniformoutput',false);
            
            enrichedDims = find(newRanks>f.ranks);
            fH = timesMatrix(f,H);
            for mu = enrichedDims
                f = orth(f,mu);
                addedrank = newRanks(mu)-f.ranks(mu);
                
                alpha = evalDiagOnLeft(fH,mu);
                beta = evalDiagOnRight(fH,mu+1);
                Hleft = outerProductEvalDiag(FullTensor(H{mu}),alpha,1,1);
                Hright = outerProductEvalDiag(FullTensor(H{mu+1}),beta,1,1);
                Hleft = Hleft.data(:,:);
                Hright = Hright.data(:,:);
                rleft = size(alpha,2);
                rright = size(beta,2);
                nleft = size(H{mu},2);
                nright = size(H{mu+1},2);
                
                a0 = timesTensor(f.cores{mu},f.cores{mu+1},3,1);
                a0 = a0.data;
                a0 = reshape(a0,[nleft*rleft,nright*rright]);
                yres = y - sum((Hleft*a0).*Hright,2);
                
                C = CanonicalTensorLearning(2,s.lossFunction);
                if iscell(s.linearModelLearning)
                    C.linearModelLearning = s.linearModelLearning{1};
                else
                    C.linearModelLearning = s.linearModelLearning;
                end
                C.alternatingMinimizationParameters = s.alternatingMinimizationParameters;
                C.basesEval = {Hleft;Hright};
                C.trainingData = {[], yres};
                C.trainingData = s.trainingData;
                C.tolerance.onError = eps;
                C.display = false;
                C.alternatingMinimizationParameters.display = false;
                C.algorithm = 'greedy';
                C.initializationType = 'mean';
                if C.linearModelLearning.basisAdaptation
                    C.basesAdaptationPath = ...
                        {repmat(s.basesAdaptationPath{mu},rleft,1),...
                        repmat(s.basesAdaptationPath{mu+1},rright,1)};
                end
                C.rank = addedrank;
                C.warnings = structfun(@(x) false, s.warnings, 'UniformOutput', false);
                [a,~] = C.solve();
                r = size(a.space.spaces{1},2);
                leftcore = reshape(a.space.spaces{1},[nleft,rleft,r]);
                leftcore = permute(leftcore,[2,1,3]);
                leftcore = FullTensor(leftcore,3,[rleft,nleft,r]);
                
                rightcore = reshape(a.space.spaces{2}*diag(a.core.data),[nright,rright,r]);
                rightcore = permute(rightcore,[3,1,2]);
                rightcore = FullTensor(rightcore,3,[r,nright,rright]);
                
                f.cores{mu} = cat(f.cores{mu},leftcore,3);
                f.cores{mu+1} = cat(f.cores{mu+1},rightcore,1);
                [f.cores{mu},R] = orth(f.cores{mu},3);
                f.isOrth = mu+1;
                f.cores{mu+1} = timesMatrix(f.cores{mu+1},R,1);
                f.isOrth = false;
                
                f.ranks(mu) = f.ranks(mu)+r;
                fH.cores{mu} = timesMatrix(f.cores{mu},H{mu},2);
                fH.cores{mu+1} = timesMatrix(f.cores{mu+1},H{mu+1},2);
                fH.ranks(mu) = f.ranks(mu);
            end
        end
        
        function p = createBasisAdaptationPathDMRG(s,pleft,pright,rleft,rright)
            % CREATEBASISADAPTATIONPATHDMRG - Creation of the path for basis adaptation for the DMRG
            %
            % p = createBasisAdaptationPathDMRG(s,pleft,pright,rleft,rright)
            % s: TensorTrainTensorLearning
            % pleft: nleft-by-mleft logical (adaptation path for the left tensor)
            % pright: nright-by-mright logical (adaptation path for the right tensor)
            % rleft: integer (rank of the left tensor)
            % rright: integer (rank of the right tensor)
            % p: nleft*nright-by-N logical, with N = min({mleft,mright})
            
            nleft = size(pleft,1);
            nright = size(pright,2);
            if size(pleft,2)>=size(pright,2)
                N = size(pleft,2);
                pright = [pright,repmat(pright(:,end),1,N-size(pright,2))];
            elseif size(pright,2)>size(pleft,2)
                N = size(pright,2);
                pleft = [pleft,repmat(pleft(:,end),1,N-size(pleft,2))];
            end
            
            p=zeros(nleft*nright,N);
            for k=1:N
                temp = repmat(pleft(:,k),1,size(pright,1)).*...
                    repmat(pright(:,k)',size(pleft,1),1);
                p(:,k) = temp(:);
            end
            p = reshape(p,[nleft,nright,N]);
            p = permute(p,[1,4,2,5,3]);
            p = repmat(p,[1,rleft,1,rright,1]);
            p = reshape(p,nleft*rleft*nright*rright,N);
        end
        
        function [s,f,output] = adaptTree(s,f,looError,testError,output,i)
            % ADAPTTREE - Tree adaptation algorithm
            %
            % [s,f,output] = ADAPTTREE(s,f,error,testError,output,i)
            % s: TensorTrainTensorLearning
            % f: FunctionalTensor
            % error, testError: 1-by-1 double
            % output: struct
            % i: integer
            
            if ~s.treeAdaptation
                return
            end
            
            if ~isfield(output,'treeAdaptationIterations')
                output.treeAdaptationIterations{1} = 1:s.order;
            end
            output.treeAdaptationIterations{i} = output.treeAdaptationIterations{i-1};
            
            if isempty(s.treeAdaptationOptions.tolerance)
                if strcmpi(s.lossFunction.errorType,'relative')
                    if s.testError == false
                        s.treeAdaptationOptions.tolerance = looError;
                    elseif s.testError == true
                        s.treeAdaptationOptions.tolerance = testError;
                    end
                else
                    warning('Must provide a tolerance for the tree adaptation in the treeAdaptationOptions property. Disabling tree adaptation.')
                    s.treeAdaptation = false;
                    return
                end
            end
            
            [fPerm,newPerm] = optimizePermutationGlobal(f.tensor,s.treeAdaptationOptions.tolerance,s.treeAdaptationOptions.maxIterations);
            
            if ~all(newPerm==1:length(newPerm))
                output.adaptedTree = true;
                output.treeAdaptationIterations(i:i+1) = {output.treeAdaptationIterations{i-1}(newPerm)};
                f.tensor = fPerm;
                basesEvalPerm = s.basesEval(newPerm);
                f.bases = basesEvalPerm;
                s.basesEval = basesEvalPerm;
                if s.testError
                    if iscell(s.testData) && ~isempty(s.testData{1})
                        s.testData{1} = s.testData{1}(:,newPerm);
                    elseif ~iscell(s.testData) && ~isempty(s.testData)
                        s.testData = s.testData(:,newPerm);
                    end
                    s.basesEvalTest = s.basesEvalTest(newPerm);
                end
                
                if isa(s.bases,'FunctionalBases')
                    s.bases = permute(s.bases,newPerm);
                end
                if s.display
                    fprintf('\tTree adaptation:\n\t\tPermutation = [ %s ]\n',num2str(newPerm))
                    fprintf('\t\tRanks after permutation = [ %s ]\n',num2str(f.tensor.ranks))
                end
            else
                output.adaptedTree = false;
            end
        end
        
        %% Inner rank adaptation solver
        function [f,output] = solveInnerRankAdaptation(s)
            % SOLVEINNERRANKADAPTATION - Solver for the learning problem with tensor formats using an inner rank adaptation algorithm
            %
            % [f,output] = SOLVEINNERRANKADAPTATION(s)
            % s: TensorTrainTensorLearning
            % f: FunctionalTensor
            % output: struct
            
            if s.linearModelLearning.basisAdaptation && isempty(s.basesAdaptationPath)
                if ismethod(s.bases, 'adaptationPath')
                    s.basesAdaptationPath = adaptationPath(s.bases);
                else
                    warning('Cannot perform basis adaptation, disabling it.')
                    s.linearModelLearning = cellfun(@(x) setfield(x, 'basisAdaptation', false), ...
                        s.linearModelLearning, 'UniformOutput', false);
                end
            end
            H = s.basesEval;
            
            [~,f] = initialize(s);
            
            order = f.order;
            
            if numel(s.rank)==1
                s.rank = repmat(s.rank,1,order-1);
            end
            
            if ~isfield(s.rankAdaptationOptions,'algorithm')
                s.rankAdaptationOptions.algorithm = 'dmrg';
            end
            if ~isfield(s.rankAdaptationOptions,'maxRank')
                s.rankAdaptationOptions.maxRank = 100;
            end
            
            flag = 1;
            for k=1:s.alternatingMinimizationParameters.maxIterations
                if s.alternatingMinimizationParameters.display
                    fprintf('\n\nAlternating minimization: iteration %d\n',k)
                end
                
                f = orth(f,1);
                fH = timesMatrix(f,H);
                f00 = f;
                
                for mu=1:order-1
                    alpha = evalDiagOnLeft(fH,mu);
                    alpha = alpha.data;
                    beta = evalDiagOnRight(fH,mu+1);
                    beta = beta.data;
                    rleft = size(alpha,2);
                    rright = size(beta,2);
                    nleft = size(H{mu},2);
                    nright = size(H{mu+1},2);
                    Aleft = repmat(H{mu},[1,1,rleft]).*...
                        repmat(permute(alpha,[1,3,2]),[1,nleft,1]);
                    Aright = repmat(H{mu+1},[1,1,rright]).*...
                        repmat(permute(beta,[1,3,2]),[1,nright,1]);
                    
                    switch lower(s.rankAdaptationOptions.algorithm)
                        case 'dmrg'
                            A = repmat(Aleft,[1,1,1,nright,rright]).*...
                                repmat(permute(Aright,[1,4,5,2,3]),[1,nleft,rleft]);
                            A = A(:,:);
                            
                            lMLlocal = s.linearModelLearning;
                            if lMLlocal.basisAdaptation
                                lMLlocal.basisAdaptationPath = ...
                                    createBasisAdaptationPathDMRG(s,...
                                    s.basesAdaptationPath{mu},...
                                    s.basesAdaptationPath{mu+1},rleft,rright);
                            end
                            lMLlocal.basis = [];
                            lMLlocal.basisEval = A;
                            lMLlocal.trainingData = s.trainingData;
                            [a,outputls] = lMLlocal.solve();
                            
                            a = reshape(a,[nleft*rleft,nright*rright]);
                            T = Truncator('maxRank',s.rankAdaptationOptions.maxRank,'tolerance',s.tolerance);
                            
                            if isfield(outputls,'error')
                                error = outputls.error;
                                T.tolerance = error/10;
                            end
                            a = T.truncate(a);
                            a = orth(a);
                        case 'dmrggreedy'
                            Hleft = Aleft(:,:);
                            Hright = Aright(:,:);
                            C = CanonicalTensorLearning(2,s.lossFunction);
                            C.linearModelLearning = s.linearModelLearning;
                            C.alternatingMinimizationParameters = s.alternatingMinimizationParameters;
                            C.basesEval = {Hleft;Hright};
                            C.trainingData = s.trainingData;
                            C.rank = s.rankAdaptationOptions.maxRank;
                            C.tolerance = structfun(@(x) x ./ sqrt(order), s.tolerance,'UniformOutput',false);
                            C.display = false;
                            C.alternatingMinimizationParameters.display = false;
                            C.algorithm = 'greedy';
                            C.warnings = structfun(@(x) false, s.warnings, 'UniformOutput', false);
                            if C.linearModelLearning.basisAdaptation
                                C.basesAdaptationPath = ...
                                    {repmat(s.basesAdaptationPath{mu},rleft,1),...
                                    repmat(s.basesAdaptationPath{mu+1},rright,1)};
                            end
                            [a,outputgreedy] = C.solve();
                            output.outputInternalGreedy{k}{mu}=outputgreedy;
                            T = Truncator('tolerance',eps);
                            a = T.truncate(a);
                            a = orth(a);
                            if isfield(outputgreedy,'error')
                                error = outputgreedy.error;
                            end
                        case 'dmrglowrank'
                            Hleft = Aleft(:,:);
                            Hright = Aright(:,:);
                            C = CanonicalTensorLearning(2,s.lossFunction);
                            C.linearModelLearning = s.linearModelLearning;
                            C.alternatingMinimizationParameters = s.alternatingMinimizationParameters;
                            C.basesEval = {Hleft;Hright};
                            C.trainingData = s.trainingData;
                            C.tolerance = structfun(@(x) x ./ sqrt(order), s.tolerance,'UniformOutput',false);
                            C.display = false;
                            C.alternatingMinimizationParameters.display = false;
                            C.alternatingMinimizationParameters.maxIterations = 20;
                            C.linearModelLearning.errorEstimation = true;
                            C.warnings = structfun(@(x) false, s.warnings, 'UniformOutput', false);
                            storeerror = inf(1,s.rankAdaptationOptions.maxRank);
                            a = [];
                            for ra = 1:min(f.ranks(mu)+s.maxRankVariation,s.rankAdaptationOptions.maxRank)
                                if ra==1
                                    C.initializationType = 'mean';
                                else
                                    C.initializationType = 'initialGuess';
                                end
                                C.rank = ra;
                                if C.linearModelLearning.basisAdaptation
                                    C.basesAdaptationPath = ...
                                        {repmat(s.basesAdaptationPath{mu},rleft,1),...
                                        repmat(s.basesAdaptationPath{mu+1},rright,1)};
                                end
                                [atemp,outputlowrank] = C.solve();
                                C.initialGuess = atemp;
                                storeerror(ra) = outputlowrank.error;
                                if storeerror(ra)<=min(storeerror(1:ra))
                                    a = atemp;
                                    error = storeerror(ra);
                                elseif storeerror(ra)>min(storeerror(1:ra))
                                    break
                                end
                                if s.alternatingMinimizationParameters.display
                                    fprintf('dimension = %d, internalRank = %d, error = %d\n',mu,ra,storeerror(ra));
                                end
                                if (ra>1 && outputlowrank.flag==0) || storeerror(ra)<s.tolerance.onError || (ra>1 && storeerror(ra)>min(storeerror(1:ra)))
                                    break
                                end
                            end
                            T = Truncator();
                            if exist('error','var')
                                T.tolerance = error/2;
                            end
                            a = T.truncate(a.tensor);
                            a = orth(a);
                            output.outputInternalGreedy{k}{mu}.error = error;
                            output.outputInternalGreedy{k}{mu}.errors = storeerror(1:ra);
                        otherwise
                            error('Wrong algorithm name in rankAdaptationOptions.algorithm.')
                    end
                    
                    r = size(a.space.spaces{1},2);
                    leftcore = reshape(a.space.spaces{1},[nleft,rleft,r]);
                    leftcore = permute(leftcore,[2,1,3]);
                    
                    rightcore = reshape(a.space.spaces{2}*(a.core.data),[nright,rright,r]);
                    rightcore = permute(rightcore,[3,1,2]);
                    
                    f.cores{mu} = FullTensor(leftcore,3,[rleft,nleft,r]);
                    f.cores{mu+1} = FullTensor(rightcore,3,[r,nright,rright]);
                    [f.cores{mu},R] = orth(f.cores{mu},3);
                    f.isOrth = mu+1;
                    f.cores{mu+1} = timesMatrix(f.cores{mu+1},R,1);
                    f.isOrth = false;
                    
                    f.ranks(mu) = r;
                    fH.cores{mu} = timesMatrix(f.cores{mu},H{mu},2);
                    fH.cores{mu+1} = timesMatrix(f.cores{mu+1},H{mu+1},2);
                    fH.ranks(mu) = r;
                end
                
                % Stagnation
                stagnation = norm((f-f00))/(norm(f00)+norm(f));
                if s.alternatingMinimizationParameters.display
                    if ~exist('error','var')
                        fprintf(' stagnation = %.2d\n',stagnation);
                    else
                        fprintf(' error = %.2d, Stagnation = %.2d\n',error,stagnation);
                    end
                    fprintf(' ranks = [ %s ]\n',num2str(f.ranks))
                end
                
                if s.storeIterates
                    output.iterates{k} = f;
                end
                
                if s.testError
                    fEvalTest = FunctionalTensor(f,s.basesEvalTest);
                    output.testError = s.lossFunction.testError(fEvalTest(),s.testData);
                    output.testErrorIterations(k) = output.testError;
                    if s.alternatingMinimizationParameters.display
                        fprintf(' test error = %.2d\n',output.testError)
                    end
                end
                
                % Stopping criterium
                if (k>1) && stagnation < s.alternatingMinimizationParameters.stagnation
                    flag = 2;
                    break
                end
            end
            f = FunctionalTensor(f,s.bases);
            output.flag = flag;
            output.iter = k;
            if exist('error','var')
                output.error = error;
            end
        end
    end
end