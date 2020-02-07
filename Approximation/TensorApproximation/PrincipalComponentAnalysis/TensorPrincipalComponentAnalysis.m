% Class TensorPrincipalComponentAnalysis
%
% Approximation of multidimensional arrays (tensors) using higher-order 
% principal component analysis.
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

classdef TensorPrincipalComponentAnalysis
    
    properties
        display = true;
        PCASamplingFactor = 1;
        PCAAdaptiveSampling = false;
        PCATestError = false;
        tol = 10e-8;
    end
    
    
    methods
        
        function s = TensorPrincipalComponentAnalysis()
            % function s = TensorPrincipalComponentAnalysis()
            % Principal component analysis of an algebraic tensor
            % s.PCATestError (true or false): error estimation for determining the rank for prescribed tolerance
            % s.PCAAdaptiveSampling (true or false): adaptive sampling for determining the principal components with prescribed precision
            % s.PCASamplingFactor: factor for determining the number of samples for the estimation of principal components (1 by default)
        end
        
        function [fpc,output] = alphaFunctionalPrincipalComponents(TPCA,fun,sz,alpha,basis,tol,gridalpha)
            % [fpc,output] = alphaFunctionalPrincipalComponents(TPCA,fun,sz,alpha,basis,t,gridalpha)
            % For alpha in {1,...,d}, it evaluates the alpha-principal components of a
            % tensor f, that means the principal components of
            % the matricisations f_\alpha(i_\alpha,i_notalpha).
            % It evaluates f_\alpha on the product of a given interpolation grid in dimension
            % alpha and a random grid (N samples) in the complementary dimensions.
            % Then, it computes approximations of alpha-principal components
            % in a given basis.
            % If t is an integer, t is the number of principal components
            % If t<1, the number of principal components is determined such
            % that the relative error after truncation is t.
            %
            % fun: function of d variables i_1,...,i_d which returns the
            % entries of the tensor
            % sz : size of the tensor
            % X: RandomVector of dimension d
            % alpha: set of indices in {1,...,d}
            % basis: FunctionalBasis
            % t: number of components or a positive number <1 (tolerance)
            % gridalpha: array of size n-by-#alpha
            % (if n is higher than the dimension of basis, selection of magic points associated with the basis)
            %
            % TPCA.PCASamplingFactor : factor for determining the number of samples N for the estimation of principal
            %         components (1 by default)
            %            - if t<1, N = TPCA.PCASamplingFactor*numel(basis)
            %            - if t is the rank, N=TPCA.PCASamplingFactor*t
            % fpc: functional principal components obtained by
            % interpolation on the given FunctionalBasis
            % output.pcOnGrid: principal components on the grid gridalpha
            % output.sv: corresponding singular values
            % output.grid: interpolation grid in dimension alpha
            
            X = randomMultiIndices(sz);
            d = length(sz);
            notalpha = setdiff(1:d,alpha);
            if tol< 1
                N = TPCA.PCASamplingFactor * cardinal(basis);
            else
                N = TPCA.PCASamplingFactor * tol;
            end
            
            N=ceil(N);
            
            Xnotalpha = RandomVector(X.randomVariables(notalpha));
            Xalpha = RandomVector(X.randomVariables(alpha));
            
            gridnotalpha = random(Xnotalpha,N);
            
            notalpha = setdiff(1:d,alpha);
            [~,I]=ismember(1:d,[alpha,notalpha]);
            
            if tol <1 && ~TPCA.PCAAdaptiveSampling && TPCA.PCATestError
                
                gridalpha = TPCA.alphaGridInterpolation(alpha,X,basis,gridalpha);
                grid = FullTensorGrid({gridalpha,gridnotalpha});
                xgrid=array(grid);
                ualphax = fun(xgrid(:,I));
                ualphax = FullTensor(ualphax,2,[size(gridalpha,1),N]);
                
                [V,sv] = principalComponents(ualphax,size(gridalpha,1));
                sv = diag(sv);
                [~,a]=basis.interpolate(V,gridalpha);
                
                r=0;err=Inf;
                while err > TPCA.tol
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
                    if ttest < TPCA.tol
                        break
                    end
                end
                
                if ttest >TPCA.tol
                    warning('Precision not reached, should adapt sample.')
                end
                
                a = a(:,1:r);
                output.pcOnGrid = V(:,1:r);
                output.numberOfEvaluations = size(xgrid,1) + size(xgridtest,1);
                
            elseif tol < 1 && TPCA.PCAAdaptiveSampling
                gridalpha = TPCA.alphaGridInterpolation(alpha,X,basis,gridalpha);
                A = FullTensor(zeros(size(gridalpha,1),0),2,[size(gridalpha,1),0]);
                for k=1:N
                    grid = FullTensorGrid({gridalpha,gridnotalpha(k,:)});
                    xgrid=array(grid);
                    A.data = [A.data,fun.eval(xgrid(:,I))];
                    [V,sv] = principalComponents(A,tol);
                    if sv(end)<1e-15 || size(V,2)<ceil(k/TPCA.PCASamplingFactor)
                        break
                    end
                end
                [~,a]=basis.interpolate(V,gridalpha);
                output.pcOnGrid = V;
                output.numberOfEvaluations = size(gridalpha,1)*k;
                
            else
                
                gridalpha = TPCA.alphaGridInterpolation(alpha,X,basis,gridalpha);
                grid = FullTensorGrid({gridalpha,gridnotalpha});
                xgrid=array(grid);
                A = fun(xgrid(:,I));
                A = reshape(A,[size(gridalpha,1),N]);
                [~,A]=basis.interpolate(A,gridalpha);
                A = FullTensor(A,2,[cardinal(basis),N]);
                [a,sv] = principalComponents(A,tol);
                
                output.numberOfEvaluations = size(xgrid,1);
                
            end
            
            fpc = SubFunctionalBasis(basis,a);
            
            output.grid = gridalpha;
            output.sv = sv;
            output.samples = xgrid;
            
        end
        
        
        function [fpc,outputs] = hopca(TPCA,fun,sz)
            % [fpc,output] = hopca(TPCA,fun,sz)
            % Returns the set of functional alpha-principal components of the
            % function fun, for all alpha in {1,2,...,d}.
            % fun: function of d variables i_1,...,i_d which returns the entries of the tensor
            % sz : size of the tensor
            % X: RandomVector of dimension d            %
            % t: tuple containing tolerance (if t(alpha)<1) or the desired alpha-ranks.
            % If numel(t)=1, the same tolerance or rank is used for all alpha
            % fpc: cell containing the functional alpha-principal components
            % outputs: cell containing the outputs of alphaFunctionalPrincipalComponents
            
            
            d = length(sz);
            bases = TPCA.createFunctionalBases(sz);
            bases = bases.bases;
            
            fpc = cell(1,d);
            outputs = cell(1,d);
            
            if numel(TPCA.tol)==1
                TPCA.tol = repmat(TPCA.tol,1,d);
            end
            
            
            for alpha = 1:d
                basis = bases{alpha};
                gridalpha = (1:sz(alpha))';
                [fpc{alpha},outputs{alpha}] = ...
                    TPCA.alphaFunctionalPrincipalComponents(fun,sz,alpha,basis,TPCA.tol(alpha),gridalpha);
                fpc{alpha} = eval(fpc{alpha},gridalpha);
            end
            
            
        end
        
        
        function [f,output] = TuckerApproximation(TPCA,fun,sz,varargin)
            % [f,output] = tuckerApproximation(TPCA,fun,sz)
            % Approximation of a function in Tucker format based on
            % Principal Component Analysis for the estimation of subspaces
            % sz : size of the tensor
            %
            % f: FunctionalTensor with a tensor in Tucker-Like format
            % output.magicPoints: magicPoints associated with basis of functions
            
            d = length(sz);
            bases = TPCA.createFunctionalBases(sz);
            
            if numel(TPCA.tol)==1 && TPCA.tol<1
                TPCA.tol=TPCA.tol/sqrt(d);
                TPCA.tol=repmat(TPCA.tol,1,d);
            end
            outputs = cell(1,d);
            samples = cell(1,d+1);
            numberOfEvaluations = 0;
            tspace = cell(d,1);
            fpc = cell(1,d);
            magicgrid = cell(1,d);
            
            for alpha=1:d
                basis = bases.bases{alpha};
                gridalpha = (1:sz(alpha))';
                [fpc{alpha},outputs{alpha}] = ...
                    TPCA.alphaFunctionalPrincipalComponents(fun,sz,alpha,basis,TPCA.tol(alpha),gridalpha);
                tspace{alpha}=eval(fpc{alpha},gridalpha);
                samples{alpha} = outputs{alpha}.grid;
                magicgrid{alpha} = magicPoints(fpc{alpha},samples{alpha});
                numberOfEvaluations = numberOfEvaluations + outputs{alpha}.numberOfEvaluations;
            end
            output.outputs = outputs;
            H = FullTensorProductFunctionalBasis(fpc);
            [f,out] = H.tensorProductInterpolation(@(x) fun.eval(x),magicgrid);
            samples{d+1} = out.grid;
            numberOfEvaluations = numberOfEvaluations + out.numberOfEvaluations;
            tcore = f.tensor;
            output.magicPoints = magicgrid;
            
            tspace = TSpaceVectors(tspace);
            f = TuckerLikeTensor(tcore,tspace);
            output.numberOfEvaluations = numberOfEvaluations;
            output.samples = samples;
        end
        
        
        function [f,output] = TBApproximation(TPCA,fun,sz,tree,isActiveNode)
            % [f,outputs] = TBApproximation(TPCA,fun,sz,tree)
            % Approximation of a function in Tree Based tensor format based on
            % Principal Component Analysis for the estimation of subspaces
            % sz : size of the tensor
            % tree: DimensionTree
            % isActiveNode: logical array indicating which nodes of the tree are active
            
            d = length(sz);
            bases = TPCA.createFunctionalBases(sz);
            bases = bases.bases;
            
            if numel(TPCA.tol)==1 && TPCA.tol<1
                Copt = sqrt(tree.nbNodes-1);
                TPCA.tol=TPCA.tol/Copt;
            end
            
            if numel(TPCA.tol)==1
                TPCA.tol = repmat(TPCA.tol,1,tree.nbNodes);
            end
            
            if nargin<5
                isActiveNode = true(1,tree.nbNodes);
            end
            
            grids = cell(1,d);
            alphaBasis = cell(1,tree.nbNodes);
            alphaGrids = cell(1,tree.nbNodes);
            outputs = cell(1,tree.nbNodes);
            samples = cell(1,tree.nbNodes);
            tensors = cell(1,tree.nbNodes);
            numberOfEvaluations = 0;
            
            for nu=1:d
                alpha = tree.dim2ind(nu);
                grids{nu} = (1:sz(nu))';
                %grids{nu} = TPCA.alphaGridInterpolation(nu,X,bases{nu},grids{nu});
                if isActiveNode(alpha)
                    [alphaBasis{alpha},outputs{alpha}] = ...
                        TPCA.alphaFunctionalPrincipalComponents(fun,sz,nu,bases{nu},TPCA.tol(nu),grids{nu});
                    szalpha = [cardinal(bases{nu}),cardinal(alphaBasis{alpha})];
                    tensors{alpha} = FullTensor(alphaBasis{alpha}.basis,2,szalpha);
                    samples{alpha} = outputs{alpha}.samples;
                    alphaGrids{alpha}=magicPoints(alphaBasis{alpha},outputs{alpha}.grid);
                    numberOfEvaluations = numberOfEvaluations + outputs{alpha}.numberOfEvaluations;
                    if TPCA.display
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
                    alphaGrids{alpha} = array(FullTensorGrid(alphaGrids(Salpha)));
                    [alphaBasis{alpha},outputs{alpha}] = ...
                        TPCA.alphaFunctionalPrincipalComponents(fun,sz,tree.dims{alpha},alphaBasis{alpha},TPCA.tol(alpha),alphaGrids{alpha});
                    szalpha = [cellfun(@cardinal,alphaBasis(Salpha)),...
                        cardinal(alphaBasis{alpha})];
                    tensors{alpha} = FullTensor(alphaBasis{alpha}.basis,length(Salpha)+1,szalpha);
                    samples{alpha} = outputs{alpha}.samples;
                    alphaGrids{alpha}=magicPoints(alphaBasis{alpha},outputs{alpha}.grid);
                    
                    numberOfEvaluations = numberOfEvaluations + outputs{alpha}.numberOfEvaluations;
                    if TPCA.display
                        fprintf('alpha = %d: rank = %d, nbeval = %d\n',alpha,szalpha(end),outputs{alpha}.numberOfEvaluations);
                    end
                end
            end
            
            alpha=tree.root;
            Salpha = nonzeros(tree.children(:,alpha))';
            alphaBasis{alpha} = FullTensorProductFunctionalBasis(alphaBasis(Salpha));
            szalpha = cellfun(@cardinal,alphaBasis(Salpha));
            [~,I]=ismember(1:d,tree.dims{alpha});
            [tensors{alpha},out] = alphaBasis{alpha}.tensorProductInterpolation(@(x) fun(x(:,I)),alphaGrids(Salpha));
            alphaGrids{alpha} = out.grid;
            tensors{alpha} = tensors{alpha}.tensor;
            numberOfEvaluations = numberOfEvaluations + out.numberOfEvaluations;
            samples{alpha} = array(out.grid);
            if TPCA.display
                fprintf('Interpolation - nbeval = %d\n',out.numberOfEvaluations);
            end
            
            f = TreeBasedTensor(tensors,tree);
            
            output.numberOfEvaluations = numberOfEvaluations;
            output.samples = samples;
            output.alphaBasis = alphaBasis;
            output.alphaGrids = alphaGrids;
            output.outputs = outputs;
        end
        
        
        
        function [f,output] = TTApproximation(TPCA,fun,sz)
            % [f,outputs] = TTApproximation(TPCA,fun,sz)
            % Approximation of a function in Tensor Train format based on
            % Principal Component Analysis
            X = randomMultiIndices(sz);
            bases = TPCA.createFunctionalBases(sz);
            bases = bases.bases;
            d = length(sz);
            
            if numel(TPCA.tol)==1 && TPCA.tol<1
                Copt = sqrt(d-1);
                TPCA.tol=TPCA.tol/Copt;
            end
            
            if numel(TPCA.tol)==1
                TPCA.tol = repmat(TPCA.tol,1,d-1);
            end
            
            
            fpc = cell(1,d-1);
            magicgrids = cell(1,d-1);
            outputs = cell(1,d-1);
            samples = cell(1,d);
            cores = cell(1,d);
            grids = cell(1,d);
            for nu=1:d
                grids{nu} = (1:sz(nu))';
            end
            nu=1;
            gridnu = TPCA.alphaGridInterpolation(nu,X,bases{nu},grids{nu});
            
            basisalpha = bases{nu};
            [fpc{1},outputs{1}] = ...
                TPCA.alphaFunctionalPrincipalComponents(fun,sz,nu,basisalpha,TPCA.tol(1),gridnu);
            
            
            numberOfEvaluations = outputs{1}.numberOfEvaluations;
            samples{1}=outputs{1}.samples;
            magicgrids{1} = magicPoints(fpc{1},outputs{1}.grid);
            
            sznu = [1,cardinal(bases{nu}),cardinal(fpc{nu})];
            cores{nu} = FullTensor(fpc{nu}.basis,3,sznu);
            if TPCA.display
                fprintf('alpha ={%d}: rank = %d, nbeval = %d\n',nu,sznu(3),outputs{nu}.numberOfEvaluations);
            end
            
            for nu=2:d-1
                alpha = 1:nu;
                basisalpha = FullTensorProductFunctionalBasis({fpc{nu-1},bases{nu}});
                gridnu = TPCA.alphaGridInterpolation(nu,X,bases{nu},grids{nu});
                gridalpha = array(FullTensorGrid({magicgrids{nu-1},gridnu}));
                
                [fpc{nu},outputs{nu}] = ...
                    TPCA.alphaFunctionalPrincipalComponents(fun,sz,alpha,basisalpha,TPCA.tol(nu),gridalpha);
                
                numberOfEvaluations = numberOfEvaluations + outputs{nu}.numberOfEvaluations;
                samples{nu}=outputs{nu}.samples;
                magicgrids{nu} = magicPoints(fpc{nu},gridalpha);
                
                sznu = [cardinal(fpc{nu-1}),cardinal(bases{nu}),cardinal(fpc{nu})];
                cores{nu} = FullTensor(fpc{nu}.basis,3,sznu);
                if TPCA.display
                    fprintf('alpha ={1,...,%d}: rank = %d, nbeval = %d\n',nu,sznu(3),outputs{nu}.numberOfEvaluations);
                end
            end
            
            nu=d;
            basis = FullTensorProductFunctionalBasis({fpc{nu-1},bases{nu}});
            
            gridnu = TPCA.alphaGridInterpolation(nu,X,bases{nu},grids{nu});
            magicgrid = {magicgrids{nu-1},gridnu};
            [cores{d},out] = basis.tensorProductInterpolation(@(x) fun(x),magicgrid);
            sznu=[cardinal(fpc{nu-1}),cardinal(bases{nu}),1];
            cores{d} = reshape(cores{d}.tensor,sznu);
            numberOfEvaluations = numberOfEvaluations + out.numberOfEvaluations;
            samples{d} = array(out.grid);
            if TPCA.display
                fprintf('Interpolation - nbeval = %d\n',out.numberOfEvaluations);
            end
            
            
            f = TTTensor(cores);            
            
            output.numberOfEvaluations = numberOfEvaluations;
            output.magicPoints = magicgrids;
            
            output.outputs = outputs;
            output.fpc = fpc;
            output.samples = samples;
        end
        
        
        
        
    end
    
    
    methods (Hidden)
        
        
        function bases = createFunctionalBases(~,sz)
            d = length(sz);
            bases = cell(1,length(sz));
            for k=1:d
                bases{k} = DeltaFunctionalBasis((1:sz(k))');
            end
            bases = FunctionalBases(bases);
        end
        
        function gridalpha=alphaGridInterpolation(~,alpha,X,basis,gridalpha)
            
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
        
    end
end
