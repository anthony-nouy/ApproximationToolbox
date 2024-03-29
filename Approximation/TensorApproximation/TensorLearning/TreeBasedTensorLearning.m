% Class TreeBasedTensorLearning: learning with tree-based tensor formats
%
% References:
% - Grelier, E., Nouy, A., & Chevreuil, M. (2018). Learning with tree-based
% tensor formats. arXiv preprint arXiv:1811.04455
% - Grelier, E., Nouy, A., & Lebrun, R. (2019). Learning high-dimensional
% probability distributions using tree tensor networks. arXiv preprint
% arXiv:1912.07913.

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

classdef TreeBasedTensorLearning < TensorLearning
    
    properties
        % TREE - DimensionTree
        tree
        % ISACTIVENODE - 1-by-N logical specifying the active nodes
        isActiveNode
    end

    methods
        function s = TreeBasedTensorLearning(tree,isActiveNode,varargin)
            % TREEBASEDTENSORLEARNING - Constructor for the TreeBasedTensorLearning class
            %
            % s = TREEBASEDTENSORLEARNING(tree,activeNodes,loss)
            % tree: DimensionTree
            % isActiveNode: 1-by-tree.nbNodes logical
            % loss: LossFunction
            
            s@TensorLearning(varargin{:});
            
            s.tree = tree;
            s.isActiveNode = isActiveNode;
            s.order = numel(tree.dim2ind);
            s.numberOfParameters = nnz(isActiveNode);
            
            s.initializationType = 'canonical';
            s.rankAdaptationOptions.rankOneCorrection = true;
            s.rankAdaptationOptions.theta = 0.8;
            s.linearModelLearningParameters.basisAdaptationInternalNodes = false;
        end
        
        %% Standard solver methods
        function [s,f] = initialize(s)
            if isempty(s.tree)
                error('Must provide a DimensionTree object in property tree.')
            end
            
            if numel(s.rank) == 1 || numel(s.rank) == s.order
                r = zeros(1,s.tree.nbNodes);
                r(logical(s.isActiveNode)) = s.rank;     
                s.rank = r;
            end
            s.rank(s.tree.root) = s.outputDimension;
            
            sz = cellfun(@(x) size(x,2),s.basesEval);
            switch lower(s.initializationType)
                case 'random'
                    f = TreeBasedTensor.randn(s.tree,s.rank,sz,s.isActiveNode);
                case 'ones'
                    f = TreeBasedTensor.ones(s.tree,s.rank,sz,s.isActiveNode);
                case 'initialguess'
                    f = s.initialGuess;
                    if ~all(f.ranks == s.rank)
                        tr = Truncator;
                        tr.tolerance = eps;
                        tr.maxRank = s.rank;
                        f = tr.truncate(f);
                    end
                case 'canonical'
                    if s.outputDimension ~= 1
                        warning(['Canonical initialization not implemented for ', ...
                            'outputDimension > 1, performing a random initialization.'])
                        f = TreeBasedTensor.randn(s.tree,s.rank,sz,s.isActiveNode);
                    else
                        f = canonicalInitialization(s,max(s.rank));
                    end
                    if ~all(f.ranks == s.rank)
                        tr = Truncator;
                        tr.tolerance = eps;
                        tr.maxRank = s.rank;
                        f = tr.truncate(f);
                    end
                otherwise
                    error('Wrong initialization type.')
            end
            if ~all(f.ranks == s.rank)
                f = enrichedEdgesToRanksRandom(f,s.rank);
            end
            
            % Exploration strategy of the tree by increasing level
            t = f.tree;
            explorationStrategy = zeros(1,s.numberOfParameters);
            ind = find(f.isActiveNode);
            i = 1;
            for lvl = 0:max(t.level)
                nodes = intersect(nodesWithLevel(t,lvl),ind);
                explorationStrategy(i:i+numel(nodes)-1) = nodes;
                i = i+numel(nodes);
            end
            s.explorationStrategy = explorationStrategy;
        end
        
        function [s,f] = preProcessing(s,f)
            if length(s.linearModelLearning) ~= f.tensor.tree.nbNodes
                c = cell(1,f.tensor.tree.nbNodes);
                c(f.tensor.isActiveNode) = s.linearModelLearning;
                s.linearModelLearning = c;
            end
        end
        
        function selmu = randomizeExplorationStrategy(s)
            selmu = zeros(1,s.numberOfParameters);
            for lvl = max(s.tree.level):-1:0
                [~,ind] = intersect(s.explorationStrategy,intersect(nodesWithLevel(s.tree,lvl),find(s.isActiveNode)));
                selmu(ind) = s.explorationStrategy(ind(randperm(length(ind))));
            end
        end
        
        function [s,A,b,f] = prepareAlternatingMinimizationSystem(s,f,mu)
            t = f.tensor.tree;
            if s.linearModelLearning{mu}.basisAdaptation
                if ismember(mu,t.internalNodes)
                    if s.linearModelLearningParameters.basisAdaptationInternalNodes
                        tr = Truncator('tolerance',eps,'MaxRank',max(f.tensor.ranks));
                        f.tensor = tr.hsvd(f.tensor);
                    elseif all(f.tensor.isActiveNode(nonzeros(t.children(:,mu))))
                        s.linearModelLearning{mu}.basisAdaptation = false;
                    end
                end
                f.tensor = orthAtNode(f.tensor,mu);
                s.tree = t;
                s.isActiveNode = f.tensor.isActiveNode;
                s.linearModelLearning{mu}.basisAdaptationPath = createBasisAdaptationPath(s,f.tensor.ranks,mu);
            else
                f.tensor = orthAtNode(f.tensor,mu);
            end
            g = parameterGradientEval(f,mu);
            if mu == t.root
                A = reshape(g.data, g.sz(1),[]);
            else
                A = reshape(g.data, g.sz(1),[], f.tensor.ranks(t.root));
            end
            
            if isa(s.lossFunction,'SquareLossFunction')
                b = s.trainingData{2};
                s.linearModelLearning{mu}.sharedCoefficients = (mu ~= t.root);
            elseif isa(s.lossFunction,'DensityL2LossFunction')
                if ~iscell(s.trainingData)
                    b = [];
                elseif iscell(s.trainingData) && length(s.trainingData) == 2
                    y = s.trainingData{2};
                    if isa(y,'FunctionalTensor')
                        y = y.tensor;
                    end
                    y = orth(y);
                    if t.isLeaf(mu)
                        a = y;
                        a.tensors(~t.isLeaf) = cellfun(@(v,c) FullTensor(v.data.*c.data,v.order,v.sz),y.tensors(~t.isLeaf),f.tensor.tensors(~t.isLeaf),'UniformOutput',false);
                        I = setdiff(1:s.order,find(t.dim2ind == mu));
                        C = cellfun(@(x) x.data, f.tensor.tensors(t.dim2ind),'UniformOutput',false);
                        b = timesVector(a,C(I),I);
                        b = b.tensors{1}.data;
                    else
                        b = dot(f.tensor,y)/f.tensor.tensors{mu}.data;
                    end
                end
            end
        end
        
        function f = setParameter(s,f,mu,a)
            f.tensor.tensors{mu}.data = reshape(a,f.tensor.tensors{mu}.sz);
            f.tensor.tensors{mu}.isOrth = false;
        end
        
        function stagnation = stagnationCriterion(s,f,f0)
            stagnation = norm(f.tensor-f0.tensor)/norm(f0.tensor);
        end
        
        function finalDisplay(s,f)
            fprintf('Ranks = [ %s ]',num2str(f.tensor.ranks));
        end
        
        function f = canonicalInitialization(s,r)
            % CANONICALINITIALIZATION - Rank-r canonical initialization
            %
            % f = CANONICALINITIALIZATION(s,r)
            % s: TreeBasedTensorLearning
            % r: 1-by-1 integer
            % f: TreeBasedTensor
            
            C = CanonicalTensorLearning(s.order,s.lossFunction);
            if iscell(s.linearModelLearning)
                C.linearModelLearning = s.linearModelLearning{1};
            else
                C.linearModelLearning = s.linearModelLearning;
            end
            C.alternatingMinimizationParameters = s.alternatingMinimizationParameters;
            C.tolerance.onStagnation = eps;
            C.tolerance.error = eps;
            C.bases = s.bases;
            C.basesEval = s.basesEval;
            C.basesEvalTest = s.basesEvalTest;
            C.display = false;
            C.alternatingMinimizationParameters.display = false;
            C.initializationType = 'mean';
            C.rankAdaptation = true;
            C.rankAdaptationOptions.maxIterations = r;
            C.basesAdaptationPath = s.basesAdaptationPath;
            C.testError = s.testError;
            C.trainingData = s.trainingData;
            C.testData = s.testData;
            C.warnings = s.warnings;
            
            f = C.solve();
            f = treeBasedTensor(f.tensor,s.tree,s.isActiveNode);
        end
        
        function f = canonicalCorrection(s,f,r)
            % CANONICALCORRECTION - Rank-r canonical correction
            %
            % f = CANONICALCORRECTION(s,f,r)
            % s: TreeBasedTensorLearning
            % f: FunctionalTensor
            % r: 1-by-1 integer
            
            if isa(f,'FunctionalTensor')
                fx = timesMatrixEvalDiag(f.tensor,s.basesEval);
            elseif isempty(f)
                fx = 0;
            else
                error('Not implemented.')
            end
            
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
            if isa(fadd,'FunctionalTensor')
                fadd=fadd.tensor;
            end
            if ~isempty(f)
                f = f.tensor+fadd;
            else
                f = fadd;
            end
        end
        
        function f = rankOneCorrection(s,f)
            % RANKONECORRECTION - Rank one canonical correction
            %
            % f = RANKONECORRECTION(s,f)
            % s: TreeBasedTensorLearning
            % f: FunctionalTensor
            
            if isa(f,'FunctionalTensor')
                fx = timesMatrixEvalDiag(f.tensor,s.basesEval);
            elseif isempty(f)
                fx = 0;
            else
                error('Not implemented.')
            end
            
            if isa(s.lossFunction,'SquareLossFunction')
                R = s.trainingData{2} - fx;
            elseif isa(s.lossFunction,'DensityL2LossFunction')
                R = f;
            end
            
            slocal = s;
            slocal.modelSelection = false;
            slocal.rankAdaptation = false;
            slocal.treeAdaptation = false;
            slocal.rank = 1;
            slocal.display = false;
            slocal.alternatingMinimizationParameters.display = false;
            slocal.initializationType = 'ones';
            slocal.alternatingMinimizationParameters.maxIterations = 1;
            
            if ~iscell(slocal.trainingData)
                slocal.trainingData = {slocal.trainingData};
            end
            slocal.trainingData{2} = R;
            
            fadd = slocal.solve();
            
            if isa(fadd,'FunctionalTensor')
                fadd = fadd.tensor;
            end
            if ~isempty(f)
                f = f.tensor+fadd;
            else
                f = fadd;
            end
        end
        
        function p = createBasisAdaptationPath(s,r,alpha)
            % CREATEBASISADAPTATIONPATH - Creation of the basis adaptation path
            %
            % p = CREATEBASISADAPTATIONPATH(s,r,alpha)
            % s: TreeBasedTensorLearning
            % r: 1-by-s.tree.nbNodes integer
            % alpha: 1-by-1 integer
            % p: logical matrix
            
            t = s.tree;
            r(t.root) = 1;
            if t.isLeaf(alpha)
                palpha = s.basesAdaptationPath{t.dim2ind == alpha};
                r = r(alpha);
                p = permute(palpha,[3,1,4,2]);
                p = repmat(p,[1,1,r,2]);
                p = reshape(p,[size(p,2)*r,size(p,4)]);
            else
                if s.linearModelLearningParameters.basisAdaptationInternalNodes
                    error('Basis adaptation for internal nodes is not implemented.')
                else
                    ch = nonzeros(t.children(:,alpha));
                    if all(s.isActiveNode(ch))
                        cha = ch(s.isActiveNode(ch));
                        p = true(prod(r([alpha ; cha])),1); % No working set
                    else
                        chA = ch(s.isActiveNode(ch));
                        chNa = ch(~s.isActiveNode(ch));
                        [~,J] = find(t.dim2ind == chNa);
                        palpha = cell(1, length(ch));
                        palpha(t.childNumber(chNa)) = s.basesAdaptationPath(J);
                        palpha(t.childNumber(chA)) = arrayfun(@(x) ones(x,1), r(chA), 'UniformOutput', false);
                        ralpha = r(alpha);
                        p = palpha{end};
                        for i = length(palpha)-1:-1:1
                            p = kron(p,palpha{i});
                        end
                        p = repmat(p,ralpha,1);
                    end
                end
            end
        end

        %% Rank adaptation solver methods
        function slocal = localSolver(s)
            slocal = s;
            slocal.modelSelection = false;
            slocal.rankAdaptation = false;
            slocal.storeIterates = false;
            slocal.testError = false;
        end
        
        function [f, newRank, enrichedNodes, tensorForSelection] = newRankSelection(s,f)
            if s.rankAdaptationOptions.rankOneCorrection
                slocal = s;
                ranksAdd = ones(1,f.tensor.tree.nbNodes); 
                ranksAdd([f.tensor.tree.root, f.tensor.nonActiveNodes]) = 0;
                slocal.rank = makeRanksAdmissible(f.tensor,f.tensor.ranks + ranksAdd);
                slocal.initializationType = 'InitialGuess';
                tr = Truncator('tolerance',0,'maxRank',slocal.rank);
                slocal.initialGuess = tr.truncate(rankOneCorrection(s,f));
                slocal.alternatingMinimizationParameters.maxIterations = 10;
                slocal.modelSelection = false;
                slocal.rankAdaptation = false;
                slocal.display = false;
                slocal.alternatingMinimizationParameters.display = false;
                tensorForSelection = slocal.solve();
                tensorForSelection = tensorForSelection.tensor;
            else
                tensorForSelection = f.tensor;
            end
            
            sv = singularValues(tensorForSelection);
            
            % Remove from the rank adaptation candidates: the inactive
            % nodes, the root, the leaf nodes with a rank equal to the
            % dimension of the basis associated to it, and the nodes for
            % which the smallest singular value is almost zero
            sv(cellfun(@isempty, sv)) = {NaN}; sv{f.tensor.tree.root} = NaN;
            dim2ind = intersect(f.tensor.tree.dim2ind,f.tensor.activeNodes);
            sv(dim2ind(~cellfun(@(x) range(x.sz), f.tensor.tensors(dim2ind)))) = {NaN};
            sv(slocal.rank ~= tensorForSelection.ranks) = {NaN};
            
            svmin = cellfun(@min,sv(1:end));
            svmin(svmin/norm(tensorForSelection) < eps) = NaN;
            
            % Remove nodes that cannot be enriched because their rank
            % is equal to the product of the ranks of their children,
            % and their children cannot be enriched themselves
            t = tensorForSelection.tree;
            r = f.tensor.ranks;
            desc = setdiff(1:t.nbNodes,find(t.isLeaf));
            cannotBeIncreased = false(1,t.nbNodes);
            cannotBeIncreased(t.root) = true;
            cannotBeIncreased(t.isLeaf) = isnan(svmin(t.isLeaf));
            for lvl = max(t.level)-1:-1:1
                nodLvl = intersect(nodesWithLevel(t,lvl),desc);
                for nod = nodLvl
                    ch = nonzeros(t.children(:,nod));
                    if all(cannotBeIncreased(ch)) && r(nod) == prod(r(ch))
                        cannotBeIncreased(nod) = true;
                    end
                end
            end
            cannotBeIncreasedNodes = t.nodesIndices(cannotBeIncreased);
            for lvl = 1:max(t.level)-1
                nodLvl = setdiff(nodesWithLevel(t,lvl), cannotBeIncreasedNodes);
                for nod = nodLvl
                    pa = t.parent(nod);
                    ind = setdiff(nonzeros(t.children(:,pa)), nod);
                    ind = [pa, ind(:).'];
                    if all(cannotBeIncreased(ind)) && ...
                            r(nod) == prod(r(ind))
                        cannotBeIncreased(nod) = true;
                    end
                end
            end
            svmin(cannotBeIncreased) = NaN;
            theta = s.rankAdaptationOptions.theta*max(svmin);
            
            enrichedNodes = find(svmin >= theta);
            newRank = f.tensor.ranks;
            newRank(enrichedNodes) = newRank(enrichedNodes) + 1;
            
            if ~isAdmissibleRank(f.tensor,newRank)
                % Add to the already enriched nodes nodes one by one in
                % decreasing order of singular value until the rank is
                % admissible
                enrichedNodesTheta = enrichedNodes;
                rTheta = newRank;
                svmin(enrichedNodesTheta) = NaN;
                svminSorted = uniquetol2(svmin,1e-2);
                svminSorted = flip(svminSorted);
                svminSorted(isnan(svminSorted)) = [];
                
                for i = 1:length(svminSorted)
                    newRank = rTheta;
                    ind = svmin >= svminSorted(i);
                    newRank(ind) = newRank(ind) + 1;
                    if isAdmissibleRank(f.tensor,newRank)
                        enrichedNodes = [enrichedNodesTheta, find(ind)];
                        break
                    end
                end
                if ~isAdmissibleRank(f.tensor,newRank)
                    newRank = f.tensor.ranks;
                    enrichedNodes = [];
                end
            end
        end
        
        function slocal = initialGuessNewRank(s,slocal,f,newRank)
            slocal.initializationType = 'initialguess';
            if ~all(f.ranks == newRank)
                tr = Truncator;
                tr.tolerance = 0;
                tr.maxRank = newRank;
                slocal.initialGuess = tr.truncate(f);
            else
                slocal.initialGuess = f;
            end
        end
        
        function adaptationDisplay(s,f,enrichedNodes)
            fprintf('\tEnriched nodes: [ %s ]\n\tRanks = [ %s]\n',num2str(enrichedNodes(:).'),num2str(f.tensor.ranks));
        end
        
        function [s,f,output] = adaptTree(s,f,looError,testError,output,varargin)
            % ADAPTTREE - Tree adaptation algorithm
            %
            % [s,f,output] = ADAPTTREE(s,f,error,testError,output)
            % s: TreeBasedTensorLearning
            % f: FunctionalTensor
            % error, testError: 1-by-1 double
            % output: struct
            
            if ~s.treeAdaptation
                return
            end
            
             output.adaptedTree = false;
            
            if any(f.tensor.ranks(f.tensor.activeNodes) == 0)
                warning('Some ranks equal to 0, disabling tree adaptation for this step.')
                return
            end
            
            if ~isempty(s.treeAdaptationOptions.tolerance)
                adaptTreeError = s.treeAdaptationOptions.tolerance;
            elseif strcmpi(s.lossFunction.errorType,'relative')
                if isempty(testError) || testError == 0
                    adaptTreeError = looError;
                elseif isempty(looError) && testError ~= 0
                    adaptTreeError = testError;
                end
            else
                warning('Must provide a tolerance for the tree adaptation in the treeAdaptationOptions property. Disabling tree adaptation.')
                s.treeAdaptation = false;
                return
            end
            
            fPerm = optimizeDimensionTree(f.tensor,adaptTreeError,s.treeAdaptationOptions.maxIterations);
            if storage(fPerm) < storage(f.tensor)
                f.tensor = fPerm;
                s.tree = f.tensor.tree;
                s.isActiveNode = f.tensor.isActiveNode;
                output.adaptedTree = true;
                if s.display
                    fprintf('\tTree adaptation:\n\t\tRanks after permutation = [ %s ]\n',num2str(f.tensor.ranks))
                end
            end
        end
        
%% Inner rank adaptation solver
        function [f,output] = solveDmrgRankAdaptation(s)
            if ~isfield(s.rankAdaptationOptions,'maxRank')
                s.rankAdaptationOptions.maxRank = 100;
            end
            if ~isfield(s.rankAdaptationOptions,'postAlternatingMinimization')
                s.rankAdaptationOptions.postAlternatingMinimization = false;
            end
            if strcmpi(s.rankAdaptationOptions.type, 'dmrgLowRank') && ...
                    ~isfield(s.rankAdaptationOptions, 'modelSelectionType')
                s.rankAdaptationOptions.modelSelectionType = 'cvError';
            end
            
            if s.display
                s.alternatingMinimizationParameters.display= true;
            end
            
            output.flag = 0;
            
            [s,f] = initialize(s); % Initialization
            f = FunctionalTensor(f,s.basesEval);
            
            % Exploration strategy of the tree by decreasing level
            t = f.tensor.tree;
            explorationStrategy = zeros(1,s.numberOfParameters);
            ind = find(f.tensor.isActiveNode);
            i = 1;
            for lvl = max(t.level):-1:0
                nodes = intersect(nodesWithLevel(t,lvl),ind);
                explorationStrategy(i:i+numel(nodes)-1) = nodes;
                i = i+numel(nodes);
            end
            s.explorationStrategy = setdiff(explorationStrategy, t.root, 'stable');
                
            % Replication of the LinearModelLearning objects
            if s.linearModelLearningParameters.identicalForAllParameters && length(s.linearModelLearning) == 1
                s.linearModelLearning = repmat({s.linearModelLearning},1,s.numberOfParameters);
            elseif length(s.linearModelLearning) ~= s.numberOfParameters
                error('Must provide numberOfParameters LinearModelLearning objects.')
            end
            
            if s.errorEstimation
                s.linearModelLearning = cellfun(@(x) setfield(x, 'errorEstimation', true), s.linearModelLearning, 'UniformOutput', false);
            end
            
            % Working set paths
            if any(cellfun(@(x) x.basisAdaptation,s.linearModelLearning)) && ...
                    isempty(s.basesAdaptationPath)
                dssdsd
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
                
                t = f.tensor.tree;
                for alpha = alphaList                    
                    if s.linearModelLearning{alpha}.basisAdaptation
                        if ismember(alpha,t.internalNodes)
                            if s.linearModelLearningParameters.basisAdaptationInternalNodes
                                tr = Truncator('tolerance',eps,'MaxRank',max(f.tensor.ranks));
                                f.tensor = tr.hsvd(f.tensor);
                            elseif all(f.tensor.isActiveNode(nonzeros(t.children(:,alpha))))
                                s.linearModelLearning{alpha}.basisAdaptation = false;
                            end
                        end
                        f.tensor = orthAtNode(f.tensor,t.parent(alpha));
                        s.tree = t;
                        s.isActiveNode = f.tensor.isActiveNode;
                        switch lower(s.rankAdaptationOptions.type)
                            case 'dmrg'
                                s.linearModelLearning{alpha}.basisAdaptationPath = createBasisAdaptationPathDMRG(s,f.tensor.ranks,alpha);
                            case 'dmrglowrank'
                                s.linearModelLearning{alpha}.basisAdaptationPath = createBasisAdaptationPathDMRGLowRank(s,f.tensor.ranks,alpha);
                            otherwise
                                error('Wrong rank adaptation type.')
                        end
                    else
                        f.tensor = orthAtNode(f.tensor,t.parent(alpha));
                    end
                    
                    g = parameterGradientEvalDMRG(f,alpha,[],s.rankAdaptationOptions.type);
                    
                    if isa(s.lossFunction,'SquareLossFunction')
                        b = s.trainingData{2};
                        s.linearModelLearning{alpha}.sharedCoefficients = (alpha ~= t.root);
                    elseif isa(s.lossFunction,'DensityL2LossFunction')
                        if ~iscell(s.trainingData)
                            b = [];
                        elseif iscell(s.trainingData) && length(s.trainingData) == 2
                            y = s.trainingData{2};
                            if isa(y,'FunctionalTensor')
                                y = y.tensor;
                            end
                            y = orth(y);
                            if t.isLeaf(alpha)
                                a = y;
                                a.tensors(~t.isLeaf) = cellfun(@(v,c) FullTensor(v.data.*c.data,v.order,v.sz),y.tensors(~t.isLeaf),f.tensor.tensors(~t.isLeaf),'UniformOutput',false);
                                I = setdiff(1:s.order,find(t.dim2ind == alpha));
                                C = cellfun(@(x) x.data, f.tensor.tensors(t.dim2ind),'UniformOutput',false);
                                b = timesVector(a,C(I),I);
                                b = b.tensors{1}.data;
                            else
                                b = dot(f.tensor,y)/f.tensor.tensors{alpha}.data;
                            end
                        end
                    end
                    
                    gamma = t.parent(alpha);
                    I = setdiff(1:f.tensor.tensors{gamma}.order, t.childNumber(alpha));
                    switch lower(s.rankAdaptationOptions.type)
                        case 'dmrg'
                            A = reshape(g.data, g.sz(1),[]);

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
                            
                            sz1 = prod(f.tensor.tensors{alpha}.sz(1:end-1));
                            sz2 = prod(f.tensor.tensors{gamma}.sz(I));
                            C = FullTensor(C, 2, [sz1, sz2]);
                            
                            tr = Truncator;
                            tr.tolerance = s.tolerance.onError/sqrt(nnz(f.tensor.isActiveNode)-1);
                            tr.maxRank = s.rankAdaptationOptions.maxRank;
                            C = tr.truncate(C);
                            r = size(C.space.spaces{1},2);
                            
                            sz1 = [f.tensor.tensors{alpha}.sz(1:end-1), r];
                            sz2 = [f.tensor.tensors{gamma}.sz(I), r];
                            
                            aAlpha = reshape(C.space.spaces{1}, sz1);
                            aGamma = reshape(C.space.spaces{2}.*repmat(C.core.data(:).', ...
                                size(C.space.spaces{2},1),1), sz2);
                            
                            aGamma = ipermute(aGamma, [I, t.childNumber(alpha)]);
                            sz2([I, t.childNumber(alpha)]) = sz2;
                        case 'dmrglowrank'
                            A = cellfun(@(x) reshape(x.data, x.sz(1),[]), g, 'UniformOutput', false);
                            A = A(:);
                            
                            slocal = TreeBasedTensorLearning(DimensionTree.trivial(2), [1,1,0], s.lossFunction);
                            slocal.rankAdaptation = true;
                            
%                             if strcmpi(s.rankAdaptationOptions.modelSelectionType, 'slopeHeuristic')
%                                 % Let the algorithm go to two times the maximal rank
%                                 slocal.tolerance.onError = 0;
%                                 slocal.tolerance.onStagnation = 0;
%                                 slocal.rankAdaptationOptions.maxIterations = 2*s.rankAdaptationOptions.maxRank;
%                             else
                                slocal.tolerance.onError = s.tolerance.onError/sqrt(nnz(f.tensor.isActiveNode)-1);
                                slocal.tolerance.onStagnation = s.tolerance.onStagnation;
                                slocal.rankAdaptationOptions.maxIterations = s.rankAdaptationOptions.maxRank;
%                             end
                            
                            slocal.alternatingMinimizationParameters = s.alternatingMinimizationParameters;
                            slocal.alternatingMinimizationParameters.display = false;
                            slocal.storeIterates = true;
                            slocal.testError = false;
                            slocal.errorEstimation = true;
                            slocal.display = false;
                            slocal.order = 2;
                            slocal.linearModelLearning = s.linearModelLearning{alpha};
                            slocal.warnings = structfun(@(x) false, s.warnings, 'UniformOutput', false);
                            slocal.basesAdaptationPath = s.linearModelLearning{alpha}.basisAdaptationPath;
                            slocal.trainingData = s.trainingData;
                            slocal.basesEval = A;
                            slocal.modelSelection = true;
                            slocal.modelSelectionOptions.type = s.rankAdaptationOptions.modelSelectionType;
                            [C, outputLML] = slocal.solve();
                            
%                             if strcmpi(s.rankAdaptationOptions.modelSelectionType, 'slopeHeuristic') && ...
%                                     C.tensor.ranks(2) > s.rankAdaptationOptions.maxRank
%                                 tr = Truncator('tolerance', 0, 'maxRank', s.rankAdaptationOptions.maxRank);
%                                 C.tensor = tr.truncate(C.tensor);
%                             end

                            sz1 = [f.tensor.tensors{alpha}.sz(1:end-1), C.tensor.ranks(2)];
                            sz2 = [f.tensor.tensors{gamma}.sz(I), C.tensor.ranks(2)];
                            
                            aAlpha = reshape(C.tensor.tensors{2}, sz1);
                            aGamma = reshape(permute(C.tensor.tensors{1}, [2,1]), sz2);
                            
                            aGamma = ipermute(aGamma, [I, t.childNumber(alpha)]);
                            sz2([I, t.childNumber(alpha)]) = sz2;
                            
                            aAlpha = aAlpha.data;
                            aGamma = aGamma.data;
                            
                            if (isempty(aAlpha(:)) || ~nnz(aAlpha(:)) || ~all(isfinite(aAlpha(:))) || any(isnan(aAlpha(:)))) || ...
                                    (isempty(aGamma(:)) || ~nnz(aGamma(:)) || ~all(isfinite(aGamma(:))) || any(isnan(aGamma(:))))
                                warning('Empty, zero or NaN solution, returning to the previous iteration.')
                                output.flag = -2;
                                output.error = Inf;
                                break
                            end
                        otherwise
                            error('Wrong rank adaptation type.')
                    end
                    f.tensor.tensors{alpha} = FullTensor(aAlpha, length(sz1), sz1);
                    f.tensor.tensors{gamma} = FullTensor(aGamma, length(sz2), sz2);
                end
                
                if s.rankAdaptationOptions.postAlternatingMinimization
                    fprintf('\t\tPost alternating minimization.\n')
                    slocal = TreeBasedTensorLearning(s.tree, s.isActiveNode, s.lossFunction);
                    slocal.basesEval = s.basesEval;
                    slocal.basesAdaptationPath = s.basesAdaptationPath;
                    slocal.trainingData = s.trainingData;
                    slocal.testError = s.testError;
                    slocal.testData = s.testData;
                    slocal.basesEvalTest = s.basesEvalTest;
                    slocal.modelSelection = false;
                    slocal.rankAdaptation = false;
                    slocal.storeIterates = false;
                    slocal.rank = f.tensor.ranks;
                    slocal.initializationType = 'initialGuess';
                    slocal.initialGuess = f.tensor;
                    slocal.tolerance.onError = s.tolerance.onError/sqrt(nnz(f.tensor.isActiveNode)-1);
                    slocal.tolerance.onStagnation = s.tolerance.onStagnation;
                    slocal.alternatingMinimizationParameters = s.alternatingMinimizationParameters;
                    slocal.alternatingMinimizationParameters.display = false;
                    slocal.errorEstimation = true;
                    slocal.display = false;
                    slocal.linearModelLearning = s.linearModelLearning;
                    slocal.warnings = structfun(@(x) false, s.warnings, 'UniformOutput', false);
                    [f, outputLML] = slocal.solve();
                end
                
                stagnation = stagnationCriterion(s,f,f0);
                output.stagnationIterations(k) = stagnation;
                
                if s.storeIterates
                    if isa(s.bases,'FunctionalBases')
                        output.iterates{k} = FunctionalTensor(f.tensor,s.bases);
                    else
                        output.iterates{k} = f;
                    end
                end
                
                if isfield(outputLML,'error')
                    output.error = outputLML.error;
                    output.errorIterations(k) = output.error;
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
                
                if s.treeAdaptation && k>1
                    Cold = storage(f.tensor);
                    [s,f,output] = adaptTree(s,f,output.error,[],output,k);
                    adaptedTree = output.adaptedTree;
                    
                    if adaptedTree
                        if s.display
                            fprintf('\t\tStorage complexity before permutation = %i\n',Cold);
                            fprintf('\t\tStorage complexity after permutation  = %i\n',storage(f.tensor));
                        end
                        if s.testError
                            fEvalTest = FunctionalTensor(f, s.basesEvalTest);
                            if s.display
                                fprintf('\t\tTest error after permutation = %.2d\n',s.lossFunction.testError(fEvalTest,s.testData));
                            end
                        end
                        if s.alternatingMinimizationParameters.display
                            fprintf('\n')
                        end
                    end
                end
            end
            
            if isa(s.bases,'FunctionalBases')
                f = FunctionalTensor(f.tensor,s.bases);
            end
            output.iter = k;
            
            if isfield(output, 'adaptedTree')
                output = rmfield(output, 'adaptedTree');
            end
            
            if s.display && ~s.modelSelection
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
        
        function p = createBasisAdaptationPathDMRG(s,r,alpha)
            % CREATEBASISADAPTATIONPATHDMRG - Creation of the basis adaptation path for the standard DMRG
            %
            % p = CREATEBASISADAPTATIONPATHDMRG(s,r,alpha)
            % s: TreeBasedTensorLearning
            % r: 1-by-s.tree.nbNodes integer
            % alpha: 1-by-1 integer
            % p: logical matrix
            
            t = s.tree;
            r(t.root) = 1;
            gamma = t.parent(alpha);
            
            pAlpha = createBasisAdaptationPath(s,r,alpha);            
            pAlpha = pAlpha(1:size(pAlpha,1)/r(alpha),:);
            
            ch = nonzeros(t.children(:,gamma));
            chA = setdiff(ch(s.isActiveNode(ch)), alpha);
            chNa = ch(~s.isActiveNode(ch));
            [~,J] = find(t.dim2ind == chNa);
            pGamma = cell(1, length(ch));
            pGamma(t.childNumber(chNa) + (t.childNumber(chNa) < t.childNumber(alpha))) = s.basesAdaptationPath(J);
            pGamma(t.childNumber(chA) + (t.childNumber(chA) < t.childNumber(alpha))) = arrayfun(@(x) ones(x,1), r(chA), 'UniformOutput', false);
            pGamma{1} = pAlpha;
            
            rGamma = r(gamma);
            p = pGamma{end};
            for i = length(pGamma)-1:-1:1
                p = kron(p,pGamma{i});
            end
            p = repmat(p,rGamma,1);
        end
        
        function p = createBasisAdaptationPathDMRGLowRank(s,r,alpha)
            % CREATEBASISADAPTATIONPATHDMRG - Creation of the basis adaptation path for low-rank the DMRG
            %
            % p = CREATEBASISADAPTATIONPATHDMRG(s,r,alpha)
            % s: TreeBasedTensorLearning
            % r: 1-by-s.tree.nbNodes integer
            % alpha: 1-by-1 integer
            % p: cell of 2 logical matrices
            
            t = s.tree;
            r(t.root) = 1;
            gamma = t.parent(alpha);
            
            pAlpha = createBasisAdaptationPath(s,r,alpha);            
            pAlpha = pAlpha(1:size(pAlpha,1)/r(alpha),:);
            
            ch = setdiff(nonzeros(t.children(:,gamma)), alpha);
            chA = ch(s.isActiveNode(ch));
            chNa = ch(~s.isActiveNode(ch));
            if ~isempty(chNa)
                [~,J] = find(t.dim2ind == chNa);
            else
                J = [];
            end
            pGamma = cell(1, length(ch));
            pGamma(t.childNumber(chNa) - ...
                (t.childNumber(chNa) > t.childNumber(alpha))) = s.basesAdaptationPath(J);
            pGamma(t.childNumber(chA) - ...
                (t.childNumber(chA) > t.childNumber(alpha))) = arrayfun(@(x) ones(x,1), r(chA), 'UniformOutput', false);
            
            rGamma = r(gamma);
            p = pGamma{end};
            for i = length(pGamma)-1:-1:1
                p = kron(p,pGamma{i});
            end
            p = repmat(p,rGamma,1);
            p = {pAlpha, p};
        end
    end
    
    methods (Static)
        function s = TensorTrain(d,varargin)
            % TENSORTRAIN - Call of the constructor of the class TreeBasedTensorLearning, with a tree and active nodes corresponding to the Tensor-Train format in dimension d
            %
            % s = TENSORTRAIN(d,loss)
            % d: 1-by-1 integer
            % loss: LossFunction
            %
            % See also TREEBASEDTENSORLEARNING
            
            tree = DimensionTree.linear(d);
            isActiveNode = true(1,tree.nbNodes);
            isActiveNode(tree.dim2ind(2:end)) = false;
            s = TreeBasedTensorLearning(tree,isActiveNode,varargin{:});
        end
        
        function s = TensorTrainTucker(d,varargin)
            % TENSORTRAINTUCKER - Call of the constructor of the class TreeBasedTensorLearning, with a tree and active nodes corresponding to the Tensor-Train Tucker format in dimension d
            %
            % s = TENSORTRAINTUCKER(d,loss)
            % d: 1-by-1 integer
            % loss: LossFunction
            %
            % See also TREEBASEDTENSORLEARNING
            
            tree = DimensionTree.linear(d);
            isActiveNode = true(1,tree.nbNodes);
            s = TreeBasedTensorLearning(tree,isActiveNode,varargin{:});
        end
    end
end

function f = enrichedEdgesToRanksRandom(f,newRank)
% ENRICHEDEDGESTORANKSRANSOM - Enrichment of the ranks of specified edges of the tensor f using random additions for each child / parent couple of the enriched edges
% f = ENRICHEDEDGESTORANKSRANSOM(f,newRank)
% f: TreeBasedTensor
% newRank: 1-by-s.tree.nbNodes integer

f.isOrth = false;
t = f.tree;
enrichedDims = find(newRank>f.ranks);

for l = 1:max(t.level)
    nodLvl = intersect(nodesWithLevel(t,l),enrichedDims);
    for alpha = nodLvl
        gamma = t.parent(alpha);
        r = newRank(alpha)-f.ranks(alpha);
        
        A = reshape(f.tensors{alpha}.data,[],f.ranks(alpha));
        A = [A, repmat(A(:,end),1,r).*(1+randn(size(A,1),r))];
        A(:,end-r+1:end) = A(:,end-r+1:end) ./ sqrt(sum(A(:,end-r+1:end).^2,1));
        f.tensors{alpha}.sz(end) = f.tensors{alpha}.sz(end)+r;
        f.tensors{alpha}.data = reshape(A,f.tensors{alpha}.sz);
        
        ch = f.tree.childNumber(alpha);
        ind = 1:f.tensors{gamma}.order;
        ind(ch) = [];
        ind = [ind , ch];
        A = permute(f.tensors{gamma}.data,ind);
        A = reshape(A,[],f.ranks(alpha));
        A = [A, repmat(A(:,end),1,r).*(1+randn(size(A,1),r))];
        A(:,end-r+1:end) = A(:,end-r+1:end) ./ sqrt(sum(A(:,end-r+1:end).^2,1));
        f.tensors{gamma}.sz(ch) = f.tensors{gamma}.sz(ch)+r;
        A = reshape(A,f.tensors{gamma}.sz(ind));
        f.tensors{gamma}.data = ipermute(A,ind);
        
        f = updateProperties(f);
    end
end
end

function [r,d] = makeRanksAdmissible(f,r)
% MAKERANKSADMISSIBLE - Adjustment of the ranks to make the associated tree-based tensor f rank-admissible, by enriching new edges associated with nodes of the tree until all the rank admissibility conditions are met
%
% [r,d] = MAKERANKSADMISSIBLE(f,r)
% f: TreeBasedTensor
% r: 1-by-s.tree.nbNodes integer
% d: 1-by-N integer, with N the number of nodes whose rank has been increased

% Do not increase the ranks of leaf nodes with rank equal to the dimension of the approximation space
I = f.activeNodes;
ind = f.tree.isLeaf(I) & r(I) > cellfun(@(x) x.sz(1), f.tensors(I));
r(I(ind)) = cellfun(@(x) x.sz(1), f.tensors(I(ind)));
r(~f.isActiveNode) = 0;

delta = r - f.ranks;
if isAdmissibleRank(f,f.ranks+delta)
    r = f.ranks + delta;
    d = find(delta);
    return
end

ind = find(delta);
for i = 1:nnz(delta)
    pos = nchoosek(1:nnz(delta),i);
    pos = pos(randperm(size(pos,1)),:);
    for j = 1:size(pos,1)
        deltaLoc = delta;
        deltaLoc(ind(pos(j,:))) = 0;
        if isAdmissibleRank(f,f.ranks+deltaLoc)
            r = f.ranks + deltaLoc;
            d = find(deltaLoc);
            return
        end
    end
end

warning('Cannot find a representation with admissible ranks, returning previous ranks.')
r = f.ranks;
d = [];
end

function b = uniquetol2(a,tol)
% UNIQUETOL2 - Unique values within tolerance, with sorted output
%
% Two values a and b are within tolerance if abs(b-a)/b <= tol.
%
% b = UNIQUETOL2(a,tol)
% a: N-by-1 or 1-by-N double
% b: 1-by-M double, with M the number of unique terms in a within tolerance tol
% tol: 1-by-1 double
%
% See also UNIQUETOL

a = sort(a);
b = a(1);

for i = 2:length(a)
    if abs(b(end)-a(i))/b(end) > tol
        b = [b, a(i)];
    end
end
end