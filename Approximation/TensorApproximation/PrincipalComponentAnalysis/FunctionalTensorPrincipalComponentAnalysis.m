% Class FunctionalTensorPrincipalComponentAnalysis
% 
% Approximation of multivariate functions using higher-order principal
% component analysis.
%
% Implementation based on the article:
% Anthony Nouy. Higher-order principal component analysis for the approximation of
% tensors in tree-based low-rank formats. Numerische Mathematik, 141(3):743--789, Mar 2019.

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

classdef FunctionalTensorPrincipalComponentAnalysis
    
    properties
        display = true;
        PCASamplingFactor = 1;
        PCAAdaptiveSampling = false;
        PCATestError = false;
        projectionType = 'interpolation' ; % 'interpolation' or 'projection'
        regressionSamplingFactor = 10;
        regressionTensorizedSampling = false;
        interpolationSVDAfterInterpolation = true;
        sparse = false;
        tol = 10e-8;
    end
    
    methods
        
        function s = FunctionalTensorPrincipalComponentAnalysis()
            % Principal component analysis of multivariate functions
            % s.projectionType = 'interpolation' or 'regression'
            % s.PCATestError (true or false): error estimation for determining the rank for prescribed tolerance
            % s.PCAAdaptiveSampling (true or false): adaptive sampling for determining the principal components with prescribed precision
            % s.PCASamplingFactor: factor for determining the number of samples for the estimation of principal components (1 by default)
            % s.tol : tolerance (if t is a double <1) or tuple containing the desired ranks.
            
        end
        
        function [f,output] = TTApproximation(FPCA,fun,X,bases,grids,varargin)
            % [f,outputs] = TTApproximation(FPCA,fun,X,bases,grids)
            % Approximation of a function in Tensor Train format based on
            % Principal Component Analysis for the estimation of subspaces
            % X: RandomVector of dimension d
            % bases: FunctionalBases or cell containing d FunctionalBasis
            % grids: FullTensorGrid or cell containing d interpolation grids
            
            d = ndims(X);
            
            if numel(FPCA.tol)==1 && FPCA.tol<1
                Copt = sqrt(d-1);
                FPCA.tol=FPCA.tol/Copt;
            end
            
            if numel(FPCA.tol)==1
                FPCA.tol = repmat(FPCA.tol,1,d-1);
            end
            
            if isa(bases,'FunctionalBases')
                bases = bases.bases;
            end
            if nargin<5
                grids = [];
            end
            
            if ~isa(grids,'cell')
                grids = repmat({grids},1,d);
            end
            
            if ~all(fun.outputSize==1)
                getcomp = @(x,I) x(I);
                n=prod(fun.outputSize);
                funext = @(x) getcomp(reshape(fun(x(:,1:d)),n,1),x(:,d+1));
                funext = UserDefinedFunction(funext,d+1,1);
                Xext = X;
                Xext.randomVariables{end+1} = DiscreteRandomVariable(1:prod(n));
                FPCA.tol(d) = 0;
                grids{d+1} = (1:n)';
            end
            
            fpc = cell(1,d-1);
            if strcmp(FPCA.projectionType,'interpolation')
                magicgrids = cell(1,d-1);
            end
            outputs = cell(1,d-1);
            samples = cell(1,d);
            cores = cell(1,d);
            nu=1;
            alpha=1;
            basisalpha = bases{nu};
            if strcmp(FPCA.projectionType,'interpolation')
                gridnu = FPCA.alphaGridInterpolation(nu,X,bases{nu},grids{nu});
                gridalpha = gridnu;
            else
                gridalpha=[];
            end
            
            if all(fun.outputSize==1)
                [fpc{1},outputs{1}] = ...
                    FPCA.alphaFunctionalPrincipalComponents(fun,X,nu,basisalpha,FPCA.tol(1),gridalpha,varargin{:});
            else
                [fpc{1},outputs{1}] = ...
                    FPCA.alphaFunctionalPrincipalComponents(funext,Xext,nu,basisalpha,FPCA.tol(1),gridalpha,varargin{:});
            end
            
            numberOfEvaluations = outputs{1}.numberOfEvaluations;
            samples{1}=outputs{1}.samples;
            if strcmp(FPCA.projectionType,'interpolation')
                magicgrids{1} = magicPoints(fpc{1},outputs{1}.grid);
            end
            sznu = [1,cardinal(bases{nu}),cardinal(fpc{nu})];
            cores{nu} = FullTensor(fpc{nu}.basis,3,sznu);
            if FPCA.display
                fprintf('alpha ={%d}: rank = %d, nbeval = %d\n',nu,sznu(3),outputs{nu}.numberOfEvaluations);
            end
            
            for nu=2:d-1+~all(fun.outputSize==1)
                alpha = 1:nu;
                basisalpha = FullTensorProductFunctionalBasis({fpc{nu-1},bases{nu}});
                if strcmp(FPCA.projectionType,'interpolation')
                    gridnu = FPCA.alphaGridInterpolation(nu,X,bases{nu},grids{nu});
                    gridalpha = array(FullTensorGrid({magicgrids{nu-1},gridnu}));
                else
                    gridalpha = [];
                end
                if all(fun.outputSize==1)
                    [fpc{nu},outputs{nu}] = ...
                        FPCA.alphaFunctionalPrincipalComponents(fun,X,alpha,basisalpha,FPCA.tol(nu),gridalpha,varargin{:});
                else
                    if nu==d
                        %could be optimized, too many calls to the function
                    end
                    [fpc{nu},outputs{nu}] = ...
                        FPCA.alphaFunctionalPrincipalComponents(funext,Xext,alpha,basisalpha,FPCA.tol(nu),gridalpha,varargin{:});
                end
                numberOfEvaluations = numberOfEvaluations + outputs{nu}.numberOfEvaluations;
                samples{nu}=outputs{nu}.samples;
                if strcmp(FPCA.projectionType,'interpolation')
                    magicgrids{nu} = magicPoints(fpc{nu},gridalpha);
                end
                sznu = [cardinal(fpc{nu-1}),cardinal(bases{nu}),cardinal(fpc{nu})];
                cores{nu} = FullTensor(fpc{nu}.basis,3,sznu);
                if FPCA.display
                    fprintf('alpha ={1,...,%d}: rank = %d, nbeval = %d\n',nu,sznu(3),outputs{nu}.numberOfEvaluations);
                end
            end
            
            if all(fun.outputSize==1)
                nu=d;
                basis = FullTensorProductFunctionalBasis({fpc{nu-1},bases{nu}});
                switch FPCA.projectionType
                    case 'interpolation'
                        gridnu = FPCA.alphaGridInterpolation(nu,X,bases{nu},grids{nu});
                        magicgrid = {magicgrids{nu-1},gridnu};
                        [cores{d},out] = basis.tensorProductInterpolation(@(x) fun(x),magicgrid);
                        sznu=[cardinal(fpc{nu-1}),cardinal(bases{nu}),1];
                        cores{d} = reshape(cores{d}.tensor,sznu);
                        numberOfEvaluations = numberOfEvaluations + out.numberOfEvaluations;
                        samples{d} = array(out.grid);
                        if FPCA.display
                            fprintf('Interpolation - nbeval = %d\n',out.numberOfEvaluations);
                        end
                    case 'regression'
                        nbeval = cardinal(basis)*FPCA.regressionSamplingFactor;
                        xs = random(X,nbeval);
                        ls = LinearModelLearningSquareLoss;
                        ls.basis = [];
                        ls.basisEval = basis.eval(xs);
                        ls.trainingData = {[], fun(xs)};
                        cores{d} = ls.solve();
                        samples{d} = xs;
                        sznu=[cardinal(fpc{nu-1}),cardinal(bases{nu}),1];
                        numberOfEvaluations = numberOfEvaluations + nbeval;
                        cores{d} = reshape(FullTensor(cores{d}),sznu);
                        if FPCA.display
                            fprintf('Least-squares projection - nbeval = %d\n',nbeval);
                        end
                        
                end
                f = TTTensor(cores);
                f = FunctionalTensor(f,FunctionalBases(bases));
            else
                basis = fpc{d};
                feval = fun(magicgrids{d});
                finterp = basis.interpolate(feval,magicgrids{d});
                data = finterp.data;
                cores{d+1} = FullTensor(data,3,[cardinal(basis),n,1]);
                numberOfEvaluations = numberOfEvaluations + size(magicgrids{d},1);
                samples{d} = magicgrids{d};
                if FPCA.display
                    fprintf('Interpolation - nbeval = %d\n',size(magicgrids{d},1));
                end
                f = TTTensor(cores);
                f = FunctionalTensor(f,FunctionalBases(bases),1:d);
                
            end
            
            output.numberOfEvaluations = numberOfEvaluations;
            if strcmp(FPCA.projectionType,'interpolation')
                output.magicPoints = magicgrids;
            end
            output.outputs = outputs;
            output.fpc = fpc;
            output.samples = samples;
        end
        
        function [f,output] =  TTTuckerApproximation(FPCA,fun,X,bases,grids,varargin)
            % [f,outputs] = TTTuckerApproximation(FPCA,fun,X,bases,grids)
            % Approximation of a function in Tensor Train Tucker format based on
            % Principal Component Analysis for the estimation of subspaces
            % X: RandomVector of dimension d
            % bases: cell containing d FunctionalBasis or FunctionalBases
            % grids: FullTensorGrid or cell containing d interpolation grids
            
            d = ndims(X);
            tini=FPCA.tol;
            fixedRank = ~(numel(tini)==1 && tini<1);
            if ~fixedRank
                FPCA.tol=FPCA.tol/sqrt(2*d-2);
            end
            
            if numel(FPCA.tol)==1
                FPCA.tol = repmat(FPCA.tol,1,d);
            else
                FPCA.tol = tini(1:d);
            end
            
            if isa(bases,'FunctionalBases')
                bases = bases.bases;
            end
            
            if nargin<5
                grids = [];
            end
            
            if ~isa(grids,'cell')
                grids = repmat({grids},1,d);
            end
            
            numberOfEvaluations = 0;
            fpc = cell(1,2*d-2);
            outputs=cell(1,2*d-2);
            magicgrids=cell(1,2*d-2);
            samples = cell(1,2*d-1);
            
            for alpha = 2:d
                grids{alpha} = FPCA.alphaGridInterpolation(alpha,X,bases{alpha},grids{alpha});
                [fpc{alpha},outputs{alpha}] = ...
                    FPCA.alphaFunctionalPrincipalComponents(fun,X,alpha,bases{alpha},FPCA.tol(alpha),grids{alpha},varargin{:});
                numberOfEvaluations = numberOfEvaluations + outputs{alpha}.numberOfEvaluations;
                bases{alpha} = fpc{alpha};
                magicgrids{alpha} = magicPoints(fpc{alpha},outputs{alpha}.grid);
                grids{alpha} = magicgrids{alpha};
                samples{alpha} = outputs{alpha}.samples;
                if FPCA.display
                    fprintf('alpha ={%d}: rank = %d, nbeval = %d\n',alpha,cardinal(fpc{alpha}),outputs{alpha}.numberOfEvaluations);
                end
            end
            
            FPCA.tol = tini;
            if ~fixedRank
                FPCA.tol = FPCA.tol/sqrt(2*d-2)*sqrt(d-1);
            end
            
            if fixedRank && numel(FPCA.tol)>1
                FPCA.tol = FPCA.tol([1,d+1:2*d-2]);
            end
            
            [f,outputTT] = FPCA.TTApproximation(fun,X,bases,grids,varargin{:});
            fpc([1,d+1:2*d-2]) = outputTT.fpc;
            samples([1,d+1:2*d-1]) = outputTT.samples;
            
            if strcmp(FPCA.projectionType,'interpolation')
                magicgrids([1,d+1:2*d-2]) = outputTT.magicPoints;
                output.magicPoints = magicgrids;
            end
            
            output.numberOfEvaluations = numberOfEvaluations + outputTT.numberOfEvaluations;
            output.outputs = outputs;
            output.fpc = fpc;
            output.samples = samples;
        end
        
        function [f,output] = TuckerApproximation(FPCA,fun,X,bases,varargin)
            % [f,output] = tuckerApproximation(FPCA,fun,X,bases,grids)
            % Approximation of a function in Tucker format based on
            % Principal Component Analysis for the estimation of subspaces
            % X: RandomVector of dimension d
            % bases: cell containing d FunctionalBasis or FunctionalBases
            % grids: FullTensorGrid or cell containing d interpolation grids
            %
            % f: FunctionalTensor with a tensor in Tucker-Like format
            % output.magicPoints: magicPoints associated with basis of functions
            
            d = ndims(X);
            if numel(FPCA.tol)==1 && FPCA.tol<1
                FPCA.tol=FPCA.tol/sqrt(d);
            end
            samples = cell(1,d+1);
            [fpc,outputs] = FPCA.hopca(fun,X,bases,varargin{:});
            samples(1:d) = cellfun(@(x) x.grid,outputs,'uniformoutput',false);
            numberOfEvaluations = sum(cellfun(@(x) x.numberOfEvaluations,outputs));
            output.fpc=fpc;
            output.outputs = outputs;
            H = FullTensorProductFunctionalBasis(fpc);
            if FPCA.sparse == false
                switch FPCA.projectionType
                    case 'interpolation'
                        magicgrid = cellfun(@(f,out) magicPoints(f,out.grid),fpc,outputs,'uniformoutput',false);
                        [f,out] = H.tensorProductInterpolation(@(x) fun.eval(x),magicgrid);
                        samples{d+1} = out.grid;
                        numberOfEvaluations = numberOfEvaluations + out.numberOfEvaluations;
                        tcore = f.tensor;
                        output.magicPoints = magicgrid;
                    case 'regression'
                        nbeval = numel(H)*FPCA.regressionSamplingFactor;
                        xs = random(X,nbeval);
                        ls = LinearModelLearningSquareLoss;
                        ls.basisEval = H.eval(xs);
                        ls.trainingData = {[], fun(xs)};
                        tcore = ls.solve();
                        szalpha = cellfun(@numel,fpc);
                        tcore = FullTensor(tcore,length(szalpha),szalpha);
                        samples{d+1} = xs;
                        numberOfEvaluations = numberOfEvaluations + nbeval;
                end
            else
                switch FPCA.projectionType
                    case 'interpolation'
                        magicgrid = cellfun(@(f,out) magicPoints(f,out.grid),fpc,outputs,'uniformoutput',false);
                        alg = AdaptiveSparseTensorAlgorithm();
                        alg.fullOutput = true;
                        alg.maxIndex = [];
                        alg.displayIterations = true;
                        if FPCA.tol < 1
                            alg.tol = FPCA.tol/100;
                            % alg.tol = 0;
                        else
                            alg.tol = 10e-12;
                            alg.tol = 0;
                        end
                        fun.store = true;
                        %fpc = FunctionalBases(fpc);
                        [f,out] = alg.interpolate(UserDefinedFunction(fun,4),FunctionalBases(fpc),magicgrid);
                        samples{d+1} = out.grid;
                        numberOfEvaluations = numberOfEvaluations + out.numberOfEvaluations;
                        out.numberOfEvaluations
                        szalpha = cellfun(@cardinal,fpc);
                        tcore = FullTensor(f.data,length(szalpha),szalpha);
                        output.magicPoints = magicgrid;
                    case 'regression'
                        nbeval = numel(H)*FPCA.regressionSamplingFactor;
                        xs = random(X,nbeval);
                        ls = LinearModelLearningSquareLoss;
                        ls.basisEval = H.eval(xs);
                        ls.trainingData = {[], fun(xs)};
                        tcore = ls.solve();
                        szalpha = cellfun(@numel,fpc);
                        tcore = FullTensor(tcore,length(szalpha),szalpha);
                        samples{d+1} = xs;
                        numberOfEvaluations = numberOfEvaluations + nbeval;
                end
            end
            tspace = cell(d,1);
            for k=1:d
                tspace{k}=fpc{k}.basis;
            end
            tspace = TSpaceVectors(tspace);
            f = FunctionalTensor(TuckerLikeTensor(tcore,tspace),FunctionalBases(bases));
            output.numberOfEvaluations = numberOfEvaluations;
            output.samples = samples;
        end
        
        function [f,output] = TBApproximation(FPCA,fun,X,bases,tree,isActiveNode,grids,varargin)
            % [f,outputs] = TBApproximation(FPCA,fun,X,bases,tree,isActiveNode,grids)
            % Approximation of a function in Tree Based tensor format based on
            % Principal Component Analysis for the estimation of subspaces
            % X: RandomVector of dimension d
            % bases: FunctionalBases or cell containing d FunctionalBasis
            % tree: DimensionTree
            % isActiveNode: logical array indicating which nodes of the tree are active
            % grids: FullTensorGrid or cell containing d interpolation grids
            
            d = ndims(X);
            
            if numel(FPCA.tol)==1 && FPCA.tol<1
                Copt = sqrt(tree.nbNodes-1);
                FPCA.tol=FPCA.tol/Copt;
            end
            
            if numel(FPCA.tol)==1
                FPCA.tol = repmat(FPCA.tol,1,tree.nbNodes);
            end
            if isa(bases,'FunctionalBases')
                bases = bases.bases;
            end
            if nargin<7
                grids = [];
            end
            
            if ~isa(grids,'cell')
                grids = repmat({grids},1,d);
            end
            
            if ~all(fun.outputSize==1)
                error('not implemented')
            end
            
            if nargin<6
                isActiveNode = true(1,tree.nbNodes);
            end
            
            alphaBasis = cell(1,tree.nbNodes);
            alphaGrids = cell(1,tree.nbNodes);
            outputs = cell(1,tree.nbNodes);
            samples = cell(1,tree.nbNodes);
            tensors = cell(1,tree.nbNodes);
            numberOfEvaluations = 0;
            
            for nu=1:d
                alpha = tree.dim2ind(nu);
                if strcmp(FPCA.projectionType,'interpolation')
                    grids{nu} = FPCA.alphaGridInterpolation(nu,X,bases{nu},grids{nu});
                end
                if isActiveNode(alpha)
                    [alphaBasis{alpha},outputs{alpha}] = ...
                        FPCA.alphaFunctionalPrincipalComponents(fun,X,nu,bases{nu},FPCA.tol(nu),grids{nu},varargin{:});
                    szalpha = [cardinal(bases{nu}),cardinal(alphaBasis{alpha})];
                    tensors{alpha} = FullTensor(alphaBasis{alpha}.basis,2,szalpha);
                    samples{alpha} = outputs{alpha}.samples;
                    alphaGrids{alpha}=magicPoints(alphaBasis{alpha},outputs{alpha}.grid);
                    numberOfEvaluations = numberOfEvaluations + outputs{alpha}.numberOfEvaluations;
                    if FPCA.display
                        fprintf('alpha = %d : rank = %d, nbeval = %d\n',alpha,szalpha(end),outputs{alpha}.numberOfEvaluations);
                    end
                else
                    alphaGrids{alpha} = grids{nu};
                    alphaBasis{alpha} = bases{nu};
                end
            end
            
            for l = max(tree.level)-1:-1:1
                Tl = intersect(tree.nodesWithLevel(l),tree.internalNodes);
                for alpha = Tl
                    Salpha = nonzeros(tree.children(:,alpha))';
                    alphaBasis{alpha} = FullTensorProductFunctionalBasis(alphaBasis(Salpha));
                    if strcmp(FPCA.projectionType,'interpolation')
                        alphaGrids{alpha} = array(FullTensorGrid(alphaGrids(Salpha)));
                    end
                    [alphaBasis{alpha},outputs{alpha}] = ...
                        FPCA.alphaFunctionalPrincipalComponents(fun,X,tree.dims{alpha},alphaBasis{alpha},FPCA.tol(alpha),alphaGrids{alpha},varargin{:});
                    szalpha = [cellfun(@cardinal,alphaBasis(Salpha)),...
                        cardinal(alphaBasis{alpha})];
                    tensors{alpha} = FullTensor(alphaBasis{alpha}.basis,length(Salpha)+1,szalpha);
                    samples{alpha} = outputs{alpha}.samples;
                    if strcmp(FPCA.projectionType,'interpolation')
                        alphaGrids{alpha}=magicPoints(alphaBasis{alpha},outputs{alpha}.grid);
                    end
                    numberOfEvaluations = numberOfEvaluations + outputs{alpha}.numberOfEvaluations;
                    if FPCA.display
                        fprintf('alpha = %d: rank = %d, nbeval = %d\n',alpha,szalpha(end),outputs{alpha}.numberOfEvaluations);
                    end
                end
            end
            
            alpha=tree.root;
            Salpha = nonzeros(tree.children(:,alpha))';
            alphaBasis{alpha} = FullTensorProductFunctionalBasis(alphaBasis(Salpha));
            szalpha = cellfun(@cardinal,alphaBasis(Salpha));
            switch FPCA.projectionType
                case 'interpolation'
                    [~,I]=ismember(1:d,tree.dims{alpha});
                    [tensors{alpha},out] = alphaBasis{alpha}.tensorProductInterpolation(@(x) fun(x(:,I)),alphaGrids(Salpha));
                    alphaGrids{alpha} = out.grid;
                    tensors{alpha} = tensors{alpha}.tensor;
                    numberOfEvaluations = numberOfEvaluations + out.numberOfEvaluations;
                    samples{alpha} = array(out.grid);
                    if FPCA.display
                        fprintf('Interpolation - nbeval = %d\n',out.numberOfEvaluations);
                    end
                case 'regression'
                    nbeval = cardinal(alphaBasis{alpha})*FPCA.regressionSamplingFactor;
                    xs = random(X,nbeval);
                    ls = LinearModelLearningSquareLoss;
                    ls.basisEval = alphaBasis{alpha}.eval(xs(:,tree.dims{alpha}));
                    ls.trainingData = {[], fun(xs)};
                    samples{alpha} = xs;
                    tensors{alpha} = ls.solve();
                    tensors{alpha} = FullTensor(tensors{alpha},length(szalpha),szalpha);
                    if FPCA.display
                        fprintf('Least-squares projection - nbeval = %d\n',nbeval);
                    end
            end
            f = TreeBasedTensor(tensors,tree);
            f = FunctionalTensor(f,FunctionalBases(bases));
            
            output.numberOfEvaluations = numberOfEvaluations;
            output.samples = samples;
            output.alphaBasis = alphaBasis;
            output.alphaGrids = alphaGrids;
            output.outputs = outputs;
        end
        
        
        function [f,output] = VectorValuedApproximation(FPCA,fun,X,bases,pcafun,varargin)
            % [f,output] = TBApproximationVectorValued(FPCA,fun,X,bases,pcafun,varargin)
            
            d = ndims(X);
            getcomp = @(x,I) x(I);
            n = fun.outputSize;
            funext = @(x) getcomp(reshape(fun(x(:,1:d)),n,1),x(:,d+1));
            funext = UserDefinedFunction(funext,d+1,1);
            Xext = X;
            Xext.randomVariables{end+1} = DiscreteRandomVariable(1:prod(n));
            if isa(bases,'FunctionalBases')
                bases = bases.bases;
            end
            basesext = bases;
            basesext{end+1} = DeltaFunctionalBasis((1:prod(n))');
            [fext,output] = pcafun(FPCA,funext,Xext,basesext,varargin{:});
            f = FunctionalTensor(fext.tensor,FunctionalBases(bases),1:d);
        end
        
        
        function [fpc,outputs] = hopca(FPCA,fun,X,bases,grids,varargin)
            % [fpc,output] = hopca(FPCA,fun,X,bases,t,grids)
            % Returns the set of functional alpha-principal components of the
            % function fun, for all alpha in {1,2,...,d}.
            % fun: function of d variables
            % X: RandomVector of dimension d
            % bases: cell containing d FunctionalBasis or FunctionalBases
            % grids: FullTensorGrid or cell containing d interpolation grids
            % fpc: cell containing the functional alpha-principal components (SubFunctionalBasis) obtained by interpolation on the FunctionalBasis given in bases.
            % outputs: cell containing the outputs of alphaFunctionalPrincipalComponents
            %
            % The function calls for all alpha the function alphaFunctionalPrincipalComponents(fun,X,alpha,bases{alpha},t(alpha),grids{alpha},N)
            
            
            d = ndims(X);
            fpc = cell(1,d);
            outputs = cell(1,d);
            
            if isa(bases,'FunctionalBases')
                bases = bases.bases;
            end
            
            if numel(FPCA.tol)==1
                FPCA.tol = repmat(FPCA.tol,1,d);
            end
            
            if nargin<5
                grids = [];
            end
            
            if ~isa(grids,'cell')
                grids = repmat({grids},1,d);
            end
            
            for alpha = 1:d
                [fpc{alpha},outputs{alpha}] = ...
                    FPCA.alphaFunctionalPrincipalComponents(fun,X,alpha,bases{alpha},FPCA.tol(alpha),grids{alpha},varargin{:});
            end
        end
        
        function [fpc,output] = alphaFunctionalPrincipalComponents(FPCA,fun,X,alpha,basis,tol,gridalpha)
            % [fpc,output] = alphaFunctionalPrincipalComponents(fun,X,alpha,basis,t,gridalpha)
            % For alpha in {1,...,d}, it evaluates the alpha-principal components of the
            % function fun, that means  the principal components of
            % the matricisations f_\alpha(x_\alpha,x_notalpha) of fun.
            % It evaluates f_\alpha on the product of a given interpolation grid in dimension
            % alpha and a random grid (N samples) in the complementary dimensions.
            % Then, it computes approximations of functional alpha-principal components
            % in a given basis of functions (by interpolation).
            % If t is an integer, t is the number of principal components
            % If t<1, the number of principal components is determined such
            % that the relative error after truncation is t.
            % fun: function of d variables x_1,...,x_d
            % X: RandomVector of dimension d
            % alpha: set of indices in {1,...,d}
            % basis: FunctionalBasis
            % t: number of components or a positive number <1 (tolerance)
            % gridalpha: array of size n-by-#alpha
            % (if n is higher than the dimension of basis, selection of magic points associated with the basis)
            %
            % FPCA.PCASamplingFactor : factor for determining the number of samples N for the estimation of principal
            %         components (1 by default)
            %            - if t<1, N = FPCA.PCASamplingFactor*numel(basis)
            %            - if t is the rank, N=FPCA.PCASamplingFactor*t
            % fpc: functional principal components obtained by
            % interpolation on the given FunctionalBasis
            % output.pcOnGrid: principal components on the grid gridalpha
            % output.sv: corresponding singular values
            % output.grid: interpolation grid in dimension alpha
            
            d = ndims(X);
            notalpha = setdiff(1:d,alpha);
            
            if tol< 1
                N = FPCA.PCASamplingFactor * cardinal(basis);
            else
                N = FPCA.PCASamplingFactor * tol;
            end
            
            N=ceil(N);
            
            Xnotalpha = RandomVector(X.randomVariables(notalpha));
            Xalpha = RandomVector(X.randomVariables(alpha));
            
            gridnotalpha = random(Xnotalpha,N);
            
            notalpha = setdiff(1:d,alpha);
            [~,I]=ismember(1:d,[alpha,notalpha]);
            if tol <1 && ~FPCA.PCAAdaptiveSampling && FPCA.PCATestError
                if strcmp(FPCA.projectionType,'regression')
                    warning('not implemented, using interpolation')
                end
                
                gridalpha = FPCA.alphaGridInterpolation(alpha,X,basis,gridalpha);
                grid = FullTensorGrid({gridalpha,gridnotalpha});
                xgrid=array(grid);
                ualphax = fun(xgrid(:,I));
                ualphax = FullTensor(ualphax,2,[size(gridalpha,1),N]);
                
                [V,sv] = principalComponents(ualphax,size(gridalpha,1));
                sv = diag(sv);
                [~,a]=basis.interpolate(V,gridalpha);
                
                r=0;err=Inf;
                while err > FPCA.tol
                    r=r+1;
                    err = sqrt(1 - sum(sv(1:r).^2)/sum(sv.^2));
                end
                rmin=r;
                
                Ntest = 5;
                gridnotalphatest = random(Xnotalpha,Ntest);
                gridtest = FullTensorGrid({gridalpha,gridnotalphatest});
                xgridtest = array(gridtest);
                ualphaxtest = fun(xgridtest(:,I));
                ualphaxtest = reshape(ualphaxtest,[size(gridalpha,1),Ntest]);
                
                ualphatest = basis.interpolate(ualphaxtest,gridalpha);
                ttest = inf;
                
                Ntestalpha = 1000;
                gridalphatest = random(Xalpha,Ntestalpha);
                
                for r=rmin:size(gridalpha,1)
                    fpc = SubFunctionalBasis(basis,a(:,1:r));
                    [xmagic,Imagic] = magicPoints(fpc,gridalpha);
                    Iualphatest = fpc.interpolate(ualphaxtest(Imagic,:),xmagic);
                    yIV = Iualphatest.eval(gridalphatest);
                    y = ualphatest.eval(gridalphatest);
                    ttest = norm(yIV-y,'fro')/norm(y,'fro');
                    if ttest < FPCA.tol
                        break
                    end
                end
                
                if ttest >FPCA.tol
                    warning('Precision not reached, should adapt sample.')
                end
                
                a = a(:,1:r);
                output.pcOnGrid = V(:,1:r);
                output.numberOfEvaluations = size(xgrid,1) + size(xgridtest,1);
                
            elseif tol < 1 && FPCA.PCAAdaptiveSampling
                switch FPCA.projectionType
                    case 'interpolation'
                        gridalpha = FPCA.alphaGridInterpolation(alpha,X,basis,gridalpha);
                        A = FullTensor(zeros(size(gridalpha,1),0),2,[size(gridalpha,1),0]);
                        for k=1:N
                            grid = FullTensorGrid({gridalpha,gridnotalpha(k,:)});
                            xgrid=array(grid);
                            A.data = [A.data,fun.eval(xgrid(:,I))];
                            [V,sv] = principalComponents(A,tol);
                            if sv(end)<1e-15 || size(V,2)<ceil(k/FPCA.PCASamplingFactor)
                                break
                            end
                        end
                        [~,a]=basis.interpolate(V,gridalpha);
                        output.pcOnGrid = V;
                        output.numberOfEvaluations = size(gridalpha,1)*k;
                    case 'regression'
                        gridalpha = FPCA.alphaGridRegression(alpha,X,basis);
                        ls = LinearModelLearningSquareLoss;
                        ls.basisEval = basis.eval(gridalpha);
                        As =  zeros(size(gridalpha,1),0) ;
                        
                        for k=1:N
                            grid = FullTensorGrid({gridalpha,gridnotalpha(k,:)});
                            xgrid=array(grid);
                            As = [As,fun.eval(xgrid(:,I))];
                            ls.trainingData = {[], As};
                            A = ls.solve();
                            A = FullTensor(A,2,[cardinal(basis),k]);
                            [a,sv] = principalComponents(A,tol);
                            if  sv(end)<1e-15 || size(a,2)< ceil(k/FPCA.PCASamplingFactor)
                                break
                            end
                        end
                        
                        output.numberOfEvaluations = size(gridalpha,1)*k;
                end
            else
                switch FPCA.projectionType
                    case 'interpolation'
                        gridalpha = FPCA.alphaGridInterpolation(alpha,X,basis,gridalpha);
                        grid = FullTensorGrid({gridalpha,gridnotalpha});
                        xgrid=array(grid);
                        A = fun(xgrid(:,I));
                        
                        if FPCA.interpolationSVDAfterInterpolation
                            A = reshape(A,[size(gridalpha,1),N]);
                            [~,A]=basis.interpolate(A,gridalpha);
                            A = FullTensor(A,2,[cardinal(basis),N]);
                            [a,sv] = principalComponents(A,tol);
                            
                        else
                            A = FullTensor(A,2,[size(gridalpha,1),N]);
                            [V,sv] = principalComponents(A,tol);
                            [~,a]=basis.interpolate(V,gridalpha);
                            output.pcOnGrid = V;
                        end
                        if FPCA.sparse == true
                            gridalpha = FPCA.alphaGridInterpolation(alpha,X,basis,gridalpha);
                            alg = AdaptiveSparseTensorAlgorithm();
                            alg.fullOutput = true;
                            alg.maxIndex = [];
                            alg.displayIterations = false;
                            fun.store = true;
                            if FPCA.tol < 1
                                alg.tol = FPCA.tol./100;
                                % alg.tol = 0;
                            else
                                alg.tol = 10e-7;
                            end
                            s.type = '()';
                            s.subs = {':',I};
                            falpha = @(xalpha,xnotalpha) fun.eval(subsref(array(FullTensorGrid({xalpha,xnotalpha})),s));
                            
                            for k=1:size(gridnotalpha,1)
                                [f{k},out{k}] = alg.interpolate(UserDefinedFunction(@(x) falpha(x,gridnotalpha(k,:)),1),basis,gridalpha);
                            end
                            output.numberOfEvaluations = 0;
                            for i=1:length(f)
                                FBIG(:,i) =f{i}.data;
                            end
                            A = FullTensor(FBIG,2,[size(FBIG,1) N]');
                            [a,sv] = principalComponents(A,min(tol));
                            output.numberOfEvaluations = numel(find(FBIG>0));
                            %
                            %                             A = FullTensor(A,2,[size(gridalpha,1),N]);
                            %                             [V,sv] = principalComponents(A,t);
                            %                             [~,a]=basis.interpolate(V,gridalpha);
                            %                             output.pcOnGrid = V;
                        end
                        if FPCA.sparse == false
                            output.numberOfEvaluations = size(xgrid,1);
                        end
                    case 'regression'
                        gridalpha = FPCA.alphaGridRegression(alpha,X,basis);
                        grid = FullTensorGrid({gridalpha,gridnotalpha});
                        xgrid=array(grid);
                        As = fun(xgrid(:,I));
                        As = reshape(As,[size(gridalpha,1),N]);
                        ls = LinearModelLearningSquareLoss;
                        ls.basisEval = basis.eval(gridalpha);
                        ls.trainingData = {[], As};
                        A = ls.solve();
                        A = FullTensor(A,2,[cardinal(basis),N]);
                        [a,sv] = principalComponents(A,tol);
                        output.numberOfEvaluations = size(xgrid,1);
                    otherwise
                        error('not implemented')
                        output.numberOfEvaluations = size(xgrid,1);
                end
            end
            
            fpc = SubFunctionalBasis(basis,a);
            
            output.grid = gridalpha;
            output.sv = sv;
            output.samples = xgrid;
            
            %                 fprintf('alpha = [%s]\n',num2str(alpha));
            %                 fprintf('dimension of basis = %d\n',cardinal(basis));
            %                 fprintf('number of random samples = %d\n',N);
            %                 fprintf('size of the grid = %d\n',size(gridalpha,1));
            %                 fprintf('number of evaluations = %d\n\n',output.numberOfEvaluations);
        end
    end
    
    methods (Hidden)
        
        function gridalpha=alphaGridInterpolation(~,alpha,X,basis,gridalpha)
            if numel(gridalpha)==1 && round(gridalpha)==gridalpha
                %    warning('The number is not considered as an integer.')
            end
            
            if isempty(gridalpha)
                gridalpha = max(cardinal(basis)*10,1000);
                Xalpha = RandomVector(X.randomVariables(alpha));
                gridalpha = random(Xalpha,gridalpha);
            end
            
            if size(gridalpha,1)>cardinal(basis)
                gridalpha = magicPoints(basis,gridalpha);
            elseif size(gridalpha,1)<cardinal(basis)
                error('the number of grid points must be higher than the dimension of the basis')
            end
        end
        
        function gridalpha=alphaGridRegression(FPCA,alpha,X,basis)
            Xalpha = RandomVector(X.randomVariables(alpha));
            if FPCA.regressionTensorizedSampling && numel(alpha)>1 && isa(basis,'FullTensorProductFunctionalBasis')
                grids = cell(1,length(basis.bases.bases));
                m=0;
                for k=1:length(basis.bases.bases)
                    % Warning for non ordered dimensions
                    Xbeta = RandomVector(Xalpha.randomVariables(m+1:m+ndims(basis.bases.bases{k})));
                    m=m+ndims(basis.bases.bases{k});
                    grids{k} = random(Xbeta,numel(basis.bases.bases{k})*FPCA.regressionSamplingFactor);
                end
                gridalpha = FullTensorGrid(grids);
                gridalpha = array(gridalpha);
            else
                gridalpha = random(Xalpha,ceil(cardinal(basis)*FPCA.regressionSamplingFactor));
            end
        end
    end
end
