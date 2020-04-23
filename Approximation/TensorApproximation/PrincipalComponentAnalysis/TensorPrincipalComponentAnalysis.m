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
        tol = 1e-8; % relative precision
        maxRank = Inf; % maximum rank
    end
    
    
    methods
        
        function s = TensorPrincipalComponentAnalysis()
            % function s = TensorPrincipalComponentAnalysis()
            % Principal component analysis of an algebraic tensor
            %
            % TPCA.maxRank: array containing the maximum alpha-ranks (length depends on the format)
            % If numel(TPCA.maxRank)=1, use the same value for all alpha
            % Set TPCA.maxRank = inf for prescribing the precision.
            % 
            % TPCA.tol : array containing the prescribed relative precisions (length depends on the format)
            % If numel(TPCA.tol)=1, use the same value for all alpha
            % Set TPCA.tol = inf for prescribing the rank.
            %            %
            % TPCA.PCASamplingFactor : factor for determining the number 
            %    of samples N for the estimation of principal components (1 by default)
            %         
            %            - if prescribed precision, N = TPCA.PCASamplingFactor*N_alpha
            %            - if prescribed rank, N=TPCA.PCASamplingFactor*t
            %
            % s.PCAAdaptiveSampling (true or false): adaptive sampling for 
            %       determining the principal components with prescribed precision
        end
        
        function [pc,output] = alphaPrincipalComponents(TPCA,fun,sz,alpha,tol,Balpha,Ialpha)
            % [pc,output] = alphaPrincipalComponents(TPCA,fun,sz,alpha,t,Balpha,Ialpha)
            % 
            % For alpha in {1,...,d}, it evaluates the alpha-principal components of a
            % tensor f, that means the principal components of
            % the matricisations f_alpha(i_alpha,i_notalpha), where i_alpha
            % and i_notalpha are groups of indices
            %
            % It evaluates f_alpha on the product of a set of indices in dimension
            % alpha (of size Nalpha) and a set of random indices (N samples) in the complementary dimensions.
            % Then, it computes approximations of alpha-principal components
            % in a given basis phi_1(i_alpha) ... phi_Nalpha(i_alpha)
            %
            % If t is an integer, t is the rank (number of principal components)
            % If t<1, the rank (number of principal components) is determined such
            % that the relative error after truncation is t.
            %
            % fun: function of d variables i_1,...,i_d which returns the entries of the tensor
            % sz : size of the tensor
            % alpha: array containing a tuple in {1,...,d}
            % t: number of principal components or a positive number <1 (tolerance)
            % Ialpha: array of size N_alpha-by-#alpha containing N_alpha tuple i_alpha
            % Balpha: array of size N_\alpha-by-N_\alpha whose i-th column is the evaluation of phi_i at the
            % set of indices i_alpha in Ialpha.
            %
            % TPCA.PCASamplingFactor : factor for determining the number of samples N for the estimation of principal
            %         components (1 by default)
            %            - if t<1, N = TPCA.PCASamplingFactor*N_alpha
            %            - if t is the rank, N=TPCA.PCASamplingFactor*t
            % pc: array of size N_alpha-by-rank whose columns are the principal components
            % output.sv: corresponding singular values
            % output.samples: set of indices at which the tensor has been evaluated

            X = randomMultiIndices(sz);
            d = length(sz);
            notalpha = setdiff(1:d,alpha);
            if tol< 1
                N = TPCA.PCASamplingFactor * size(Balpha,2);
            else
                N = TPCA.PCASamplingFactor * tol;
            end
            
            N=ceil(N);
            
            Xnotalpha = RandomVector(X.randomVariables(notalpha));
            
            Inotalpha = random(Xnotalpha,N);
            
            notalpha = setdiff(1:d,alpha);
            [~,I]=ismember(1:d,[alpha,notalpha]);
            
            if tol < 1 && TPCA.PCAAdaptiveSampling
                A = FullTensor(zeros(size(Ialpha,1),0),2,[size(Ialpha,1),0]);
                for k=1:N
                    grid  = FullTensorGrid({Ialpha,Inotalpha(k,:)});
                    productgrid = array(grid);
                    Ak = Balpha\fun(productgrid(:,I));
                    A.data = [A.data,Ak];
                    [pc,sv] = principalComponents(A,tol);
                    if sv(end)<1e-15 || size(pc,2)<ceil(k/TPCA.PCASamplingFactor)
                        break
                    end
                end
                output.numberOfEvaluations = size(Ialpha,1)*k;
                
            else
                
                grid = FullTensorGrid({Ialpha,Inotalpha});
                productgrid=array(grid);
                A = fun(productgrid(:,I));
                A = reshape(A,[size(Ialpha,1),N]);
                A = Balpha\A;
                A = FullTensor(A,2,[size(Balpha,1),N]);
                [pc,sv] = principalComponents(A,tol);
                output.numberOfEvaluations = size(Ialpha,1)*N;
                
            end
                        
            output.sv = sv;
            output.samples = productgrid;
            
        end
        
        
        function [fpc,outputs] = hopca(TPCA,fun,sz)
            % [fpc,output] = hopca(TPCA,fun,sz)
            % Returns the set of alpha-principal components of an algebraic tensor, for all alpha in {1,2,...,d}.
            %
            % fun: function of d variables i_1,...,i_d which returns the entries of the tensor
            % sz : size of the tensor
            %
            % fpc: 1-by-d cell containing the alpha principal components
            % outputs: 1-by-d cell containing the outputs of the method alphaPrincipalComponents
            %
            % For prescribed precision, set TPCA.maxRank = inf and TPCA.tol
            % to the desired precision (possibly an array of length d)
            %
            % For prescribed rank, set TPCA.tol = inf and TPCA.maxRank to
            % the desired rank (possibly an array of length d)
            %
            % For other options, 
            % See also TensorPrincipalComponentAnalysis.TensorPrincipalComponentAnalysis
 
            
            
            d = length(sz);
            
            fpc = cell(1,d);
            outputs = cell(1,d);
            
            if numel(TPCA.tol)==1
                TPCA.tol = repmat(TPCA.tol,1,d);
            elseif numel(TPCA.tol)~=d
                error('tol should be a scalar or an array of length d')
            end
            
            if numel(TPCA.maxRank)==1
                TPCA.maxRank = repmat(TPCA.maxRank,1,d);
            elseif numel(TPCA.maxRank)~=d
                error('maxRank should be a scalar or an array of length d')
            end
            
            
            for alpha = 1:d
                Ialpha = (1:sz(alpha))';
                Balpha = speye(sz(alpha));
                tolalpha = min(TPCA.tol(alpha),TPCA.maxRank(alpha));
                [fpc{alpha},outputs{alpha}] = ...
                    TPCA.alphaPrincipalComponents(fun,sz,alpha,tolalpha,Balpha,Ialpha);
            end
            
            
        end

        function [f,output] = TuckerApproximation(TPCA,fun,sz)
            % [f,outputs] = TuckerApproximation(TPCA,fun,sz)
            % Approximation of a tensor of order d 
            % in Tucker format based on
            % Principal Component Analysis
            %
            % fun: function of d variables i_1,...,i_d which returns the entries of the tensor
            % sz : size of the tensor
            % f : a tensor in tree based format with a trivial tree
            %
            % For prescribed precision, set TPCA.maxRank = inf and TPCA.tol
            % to the desired precision (possibly an array of length d)
            %
            % For prescribed rank, set TPCA.tol = inf and TPCA.maxRank to
            % the desired rank (possibly an array of length d)
            %
            % For other options, 
            % See also TensorPrincipalComponentAnalysis.TensorPrincipalComponentAnalysis
            
            d = length(sz);
            tree = DimensionTree.trivial(d);
            if length(TPCA.tol)==d
                t = TPCA.tol;
                TPCA.tol = zeros(1,d+1);
                TPCA.tol(tree.dim2ind)=t;
            elseif length(TPCA.tol)>1 
                error('tol should be a scalar or an array of length d')
            end
        
            if length(TPCA.maxRank)==d
                r = TPCA.maxRank;
                TPCA.maxRank = zeros(1,d+1);
                TPCA.maxRank(tree.dim2ind)=r;
            elseif length(TPCA.maxRank)>1 
                error('maxRank should be a scalar or an array of length d')
            end
            
            [f,output] = TBApproximation(TPCA,fun,sz,tree);
            
        end        
        
        function [f,output] = TTApproximation(TPCA,fun,sz)
            % [f,outputs] = TTApproximation(TPCA,fun,sz)
            % Approximation of a tensor of order d 
            % in Tensor Train format based on
            % Principal Component Analysis
            %
            % fun: function of d variables i_1,...,i_d which returns the entries of the tensor
            % sz : size of the tensor
            % f : a tensor in tree based format with a linear tree
            %
            % For prescribed precision, set TPCA.maxRank = inf and TPCA.tol
            % to the desired precision (possibly an array of length d-1)
            %
            % For prescribed rank, set TPCA.tol = inf and TPCA.maxRank to
            % the desired rank (possibly an array of length d-1, the desired TT-ranks)
            %
            % For other options, 
            % See also TensorPrincipalComponentAnalysis.TensorPrincipalComponentAnalysis
            
           
            d = length(sz);
            tree = DimensionTree.linear(d);
            isActiveNode = true(1,tree.nbNodes);
            isActiveNode(tree.dim2ind(2:end)) = false;
            repTT = find(isActiveNode);
            repTT = flip(repTT(2:end));
            
            if length(TPCA.tol)==d-1
                t = TPCA.tol;
                TPCA.tol = zeros(1,tree.nbNodes);
                TPCA.tol(repTT)=t;
            elseif length(TPCA.tol)>1  
                error('tol should be a scalar or an array of length d-1')
            end
        
            if length(TPCA.maxRank)==d-1
                r = TPCA.maxRank;
                TPCA.maxRank = zeros(1,tree.nbNodes);
                TPCA.maxRank(repTT)=r;
            elseif length(TPCA.maxRank)>1  
                error('maxRank should be a scalar or an array of length d-1')
            end

            
            [f,output] = TBApproximation(TPCA,fun,sz,tree,isActiveNode);
            
        end
        
        function [f,output] = TBApproximation(TPCA,fun,sz,tree,isActiveNode)
            % [f,outputs] = TBApproximation(TPCA,fun,sz,tree,isActiveNode)
            % Approximation of a tensor in 
            % Tree Based tensor format based on
            % Principal Component Analysis
            %
            % fun: function of d variables i_1,...,i_d which returns the entries of the tensor
            % sz : size of the tensor
            % tree: DimensionTree
            % isActiveNode: logical array indicating which nodes of the tree are active
            % f : a tensor in tree based format 
            %
            % For prescribed precision, set TPCA.maxRank = inf and TPCA.tol
            % to the desired precision (possibly an array of length tree.nbNodes)
            %
            % For prescribed rank, set TPCA.tol = inf and TPCA.maxRank to
            % the desired rank (possibly an array of length tree.nbNode, the tree-based rank)
            %
            % For other options, 
            % See also TensorPrincipalComponentAnalysis.TensorPrincipalComponentAnalysis
            
            d = length(sz);
            if nargin<5
                isActiveNode = true(1,tree.nbNodes);
            end
            
            if numel(TPCA.tol)==1 && TPCA.tol<1
                Copt = sqrt(nnz(isActiveNode)-1);
                TPCA.tol=TPCA.tol/Copt;
            end
            
            if numel(TPCA.tol)==1
                TPCA.tol = repmat(TPCA.tol,1,tree.nbNodes);
            elseif length(TPCA.tol)>1 && length(TPCA.tol)~=tree.nbNodes
                error('tol should be a scalar or an array of length tree.nbNodes')
            end
            
            if numel(TPCA.maxRank)==1
                TPCA.maxRank = repmat(TPCA.maxRank,1,tree.nbNodes);
            elseif length(TPCA.maxRank)>1 && length(TPCA.maxRank)~= tree.nbNodes
                error('maxRank should be a scalar or an array of length tree.nbNodes')
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
                Balpha = eye(sz(nu));
                if isActiveNode(alpha)
                    tolalpha = min(TPCA.tol(alpha),TPCA.maxRank(alpha));
                    [pcalpha,outputs{alpha}] = ...
                        TPCA.alphaPrincipalComponents(fun,sz,nu,tolalpha,Balpha,grids{nu});
                    samples{alpha} = outputs{alpha}.samples;
                    szalpha = [sz(nu),size(pcalpha,2)];
                    tensors{alpha} = FullTensor(pcalpha,2,szalpha);
                    
                    Balpha = Balpha*pcalpha;
                    Ialpha = magicIndices(Balpha);
                    alphaGrids{alpha} = grids{nu}(Ialpha,:);
                    alphaBasis{alpha} = Balpha(Ialpha,:);
                    
                    
                    numberOfEvaluations = numberOfEvaluations + outputs{alpha}.numberOfEvaluations;
                    if TPCA.display
                        fprintf('alpha = %d : rank = %d, nbeval = %d\n',alpha,szalpha(end),outputs{alpha}.numberOfEvaluations);
                    end
                else
                    alphaGrids{alpha} = grids{nu};
                    alphaBasis{alpha} = Balpha;
                end
            end
            
            for l = max(tree.level)-1:-1:1
                Tl = intersect(tree.nodesWithLevel(l),tree.internalNodes);
                for alpha = Tl
                    Salpha = nonzeros(tree.children(:,alpha))';
                    Balpha = TensorPrincipalComponentAnalysis.tensorProductBalpha(alphaBasis(Salpha));
                    alphaGrids{alpha} = array(FullTensorGrid(alphaGrids(Salpha)));
                    
                    tolalpha = min(TPCA.tol(alpha),TPCA.maxRank(alpha));
                    [pcalpha,outputs{alpha}] = ...
                        TPCA.alphaPrincipalComponents(fun,sz,tree.dims{alpha},tolalpha,Balpha,alphaGrids{alpha});
                    samples{alpha} = outputs{alpha}.samples;
                    szalpha = [cellfun(@(x)size(x,2),alphaBasis(Salpha)),size(pcalpha,2)];
                    tensors{alpha} = pcalpha;
                    tensors{alpha} = FullTensor(tensors{alpha},length(Salpha)+1,szalpha);
                    
                    Balpha = Balpha*pcalpha;
                    Ialpha= magicIndices(Balpha);
                    alphaGrids{alpha} = alphaGrids{alpha}(Ialpha,:);
                    alphaBasis{alpha} = Balpha(Ialpha,:);
                    numberOfEvaluations = numberOfEvaluations + outputs{alpha}.numberOfEvaluations;
                    if TPCA.display
                        fprintf('alpha = %d: rank = %d, nbeval = %d\n',alpha,szalpha(end),outputs{alpha}.numberOfEvaluations);
                    end
                end
            end
            
            alpha=tree.root;
            Salpha = nonzeros(tree.children(:,alpha))';
            Balpha = TensorPrincipalComponentAnalysis.tensorProductBalpha(alphaBasis(Salpha));
            Ialpha = array(FullTensorGrid(alphaGrids(Salpha)));
            szalpha = cellfun(@(x) size(x,2),alphaBasis(Salpha));
            [~,I]=ismember(1:d,tree.dims{alpha});
            tensors{alpha} = Balpha\fun(Ialpha(:,I));
            tensors{alpha} = FullTensor(tensors{alpha},length(Salpha),szalpha);
            alphaGrids{alpha} = Ialpha;
            numberOfEvaluations = numberOfEvaluations + size(Ialpha,1);
            samples{alpha} = Ialpha;
            if TPCA.display
                fprintf('Interpolation - nbeval = %d\n',size(Ialpha,1));
            end
            
            f = TreeBasedTensor(tensors,tree);
            
            output.numberOfEvaluations = numberOfEvaluations;
            output.samples = samples;
            output.alphaBasis = alphaBasis;
            output.alphaGrids = alphaGrids;
            output.outputs = outputs;
        end
        
        
        
    end
    
    
    methods (Static,Hidden)
        
        function B = tensorProductBalpha(Bs)
            % Bs : cell containing s matrices B1 , ... , Bs 
            % where Bi is a n(i)-by-r(i) matrix
            % B : matrix of size prod(n)-by-prod(r) matrix whose entry 
            % B(I , J) = B1(i_1,j_1) ... Bs(i_s,j_s)
            % with I = (i_1,...,is) and J = (j1,...,js)
            
            Bs = cellfun(@(x) FullTensor(x,2,size(x)),Bs,'uniformoutput',false);
            B = Bs{1};
            for k=2:length(Bs)
                B = timesTensor(B,Bs{k},[],[]);
            end            
            B = permute(B,[1:2:B.order-1,2:2:B.order]);
            B = reshape(B,[prod(B.sz(1:B.order/2)),prod(B.sz(B.order/2+1:end))]);
            B = B.data;
        end
        
    end
end
