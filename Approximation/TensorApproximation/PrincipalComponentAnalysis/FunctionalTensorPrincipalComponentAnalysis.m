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
        tol = 1e-8; % relative precision
        maxRank = Inf; % maximum rank
        bases
        grid
        display = true;
        PCASamplingFactor = 1;
        PCAAdaptiveSampling = false;
        projectionType = 'interpolation';
        subSampling = 'eim'; % method for selecting indices
        subSamplingTol = []; % tolerance for subsampling
    end
    
    methods
        function s = FunctionalTensorPrincipalComponentAnalysis()
            % Principal component analysis of multivariate functions
            % based on TensorPrincipalComponentAnalysis for algebraic
            % tensors
            %
            % s.bases : FunctionalBases
            % s.grid : FullTensorGrid for the projection of the function on
            % the functional bases
            %
            % FPCA.maxRank: array containing the maximum alpha-ranks (length depends on the format)
            % If numel(FPCA.maxRank)=1, use the same value for all alpha
            % Set FPCA.maxRank = inf for prescribing the precision.
            %
            % FPCA.tol : array containing the prescribed relative precisions (length depends on the format)
            % If numel(FPCA.tol)=1, use the same value for all alpha
            % Set FPCA.tol = inf for prescribing the rank.
            %            %
            % FPCA.PCASamplingFactor : factor for determining the number
            %    of samples N for the estimation of principal components (1 by default)
            %
            %            - if prescribed precision, N = FPCA.PCASamplingFactor*N_alpha
            %            - if prescribed rank, N=FPCA.PCASamplingFactor*t
            %
            % s.PCAAdaptiveSampling (true or false): adaptive sampling for
            %       determining the principal components with prescribed precision

        end
        
        function [fpc,outputs] = hopca(FPCA,fun)
            % [fpc,output] = hopca(FPCA,fun)
            % Returns the set of alpha-principal components of a function, for all alpha in {1,2,...,d}.
            %
            % fun: function of d variables
            %
            % fpc: 1-by-d cell containing the alpha principal components
            % outputs: 1-by-d cell containing the outputs of the method alphaPrincipalComponents
            %
            % For prescribed precision, set FPCA.maxRank = inf and FPCA.tol
            % to the desired precision (possibly an array of length d)
            %
            % For prescribed rank, set FPCA.tol = inf and FPCA.maxRank to
            % the desired rank (possibly an array of length d)
            %
            % For other options,
            % See also FunctionalTensorPrincipalComponentAnalysis.FunctionalTensorPrincipalComponentAnalysis
            
            [FPCA,tfun,sz,tpca] = prepare(FPCA,fun);
            [tpc,outputs] = hopca(tpca,tfun,sz);
            
            P = projectionOperators(FPCA);
            tpc = cellfun(@(P,x) P*x,P,tpc,'uniformoutput',false);
            fpc = cellfun(@(b,a) SubFunctionalBasis(b,a),FPCA.bases.bases(:),tpc(:),'uniformoutput',false);
        end
        
        function [f,output] = TuckerApproximation(FPCA,fun)
            % [f,outputs] = TuckerApproximation(FPCA,fun)
            % Approximation of a function of d variables in Tucker format based on a Principal Component Analysis
            %
            % fun: function of d variables
            % sz : size of the tensor
            % f : a function in tree based format with a trivial tree
            %
            % For prescribed precision, set FPCA.maxRank = inf and FPCA.tol
            % to the desired precision (possibly an array of length d)
            %
            % For prescribed rank, set FPCA.tol = inf and FPCA.maxRank to
            % the desired rank (possibly an array of length d)
            %
            % For other options,
            % See also FunctionalTensorPrincipalComponentAnalysis.FunctionalTensorPrincipalComponentAnalysis
            
            [FPCA,tfun,sz,tpca] = prepare(FPCA,fun); 
            [t,output] = TuckerApproximation(tpca,tfun,sz);
            f = project(FPCA,t);
            
        end
        
        function [f,output] = TTApproximation(FPCA,fun)
            % [f,outputs] = TTApproximation(FPCA,fun)
            % Approximation of a function of d variables in Tensor Train format based on a Principal Component Analysis
            %
            % fun: function of d variables
            % f : a function in tree based format with a linear tree
            %
            % For prescribed precision, set FPCA.maxRank = inf and FPCA.tol
            % to the desired precision (possibly an array of length d-1)
            %
            % For prescribed rank, set FPCA.tol = inf and FPCA.maxRank to
            % the desired rank (possibly an array of length d-1, the desired TT-ranks)
            %
            % For other options,
            % See also FunctionalTensorPrincipalComponentAnalysis.FunctionalTensorPrincipalComponentAnalysis
            
            [FPCA,tfun,sz,tpca] = prepare(FPCA,fun); 
            [t,output] = TTApproximation(tpca,tfun,sz);
            f = project(FPCA,t);
            
        end
        
        function [f,output] = TBApproximation(FPCA,fun,varargin)
            % [f,outputs] = TBApproximation(FPCA,fun,tree,isActiveNode)
            % Approximation of a function in Tree Based tensor format based on a Principal Component Analysis
            %
            % fun: function of d variables
            % tree: DimensionTree
            % isActiveNode: logical array indicating which nodes of the tree are active
            % f : a function in tree based format
            %
            % For prescribed precision, set FPCA.maxRank = inf and FPCA.tol
            % to the desired precision (possibly an array of length tree.nbNodes)
            %
            % For prescribed rank, set FPCA.tol = inf and FPCA.maxRank to
            % the desired rank (possibly an array of length tree.nbNode, the tree-based rank)
            %
            % For other options,
            % See also FunctionalTensorPrincipalComponentAnalysis.FunctionalTensorPrincipalComponentAnalysis
            
            [FPCA,tfun,sz,tpca] = prepare(FPCA,fun); 
            [t,output] = TBApproximation(tpca,tfun,sz,varargin{:});
            f = project(FPCA,t);
        end
    end
    
    methods (Hidden) 
        function [FPCA,tfun,sz,TPCA] = prepare(FPCA,fun)
            FPCA.bases = FunctionalBases(FPCA.bases);
            % creating the tensor product grid
            switch FPCA.projectionType
                case 'interpolation'
                    if isempty(FPCA.grid)
                        FPCA.grid = interpolationPoints(FPCA.bases);
                    else
                        FPCA.grid = interpolationPoints(FPCA.bases,FPCA.grid);
                    end
                otherwise
                    error('wrong property projectionType')
            end
            FPCA.grid = FullTensorGrid(FPCA.grid);
            
            % creating the function which provides the values of the function on the grid 
            tfun = @(i) fun(evalAtIndices(FPCA.grid,i));
            sz = cardinals(FPCA.bases).';
            
            % Creating a TensorPrincipalComponentAnalysis with the same
            % values of properties as the FunctionalPrincipalComponentAnalysis
            TPCA = TensorPrincipalComponentAnalysis();
            p = properties(TPCA);
            for i=1:length(p)
                TPCA = setfield(TPCA,p{i},getfield(FPCA,p{i}));
            end
                       
        end
        
        function P = projectionOperators(FPCA)
            % creating a cell of matrices of length d. 
            % The matrix P{nu} represents the operator which associates to
            % the values of a function of the variable x_nu on a grid the coefficients of the
            % projection on the basis of functions of the variable x_nu. 
            
            switch FPCA.projectionType
                case 'interpolation'
                    P = eval(FPCA.bases,FPCA.grid.grids);
                    P = cellfun(@(x) pinv(x),P,'UniformOutput',false);
                otherwise
                    error('wrong property projectionType')
            end
            P = reshape(P,1,length(P));
        end
        
        function f = project(FPCA,t)
            % takes an AlgebraicTensor t whose entries are the values of the function
            % on a product grid, and returns a FunctionalTensor obtained by
            % applying the projections obtained by the method
            % projectionOperators.
            
            P = projectionOperators(FPCA); 
            for nu=1:t.order
                alpha = t.tree.dim2ind(nu);
                if t.isActiveNode(alpha)
                    data = P{nu}*t.tensors{alpha}.data;
                    t.tensors{alpha} = FullTensor(data,2,size(data));
                else
                    p = t.tree.parent(alpha);
                    ch = find(t.tree.children(:,p)==alpha);
                    t.tensors{p} = timesMatrix(t.tensors{p},P{nu},ch);
                end
            end
            f = FunctionalTensor(t,FPCA.bases);
        end
    end
end