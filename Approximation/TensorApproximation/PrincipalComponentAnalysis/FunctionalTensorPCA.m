% Class FunctionalTensorPCA

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

classdef FunctionalTensorPCA
    
    properties
        display = false
        vpca = [] % VectorPCA
        fpca = []
        basisAdaptation = false;
        constantCorrected = false;
    end
    
    methods
        function FTPCA = FunctionalTensorPCA()
            % Principal component analysis of multivariate functions
        end
        
        function [f,output] = TBApproximation(FTPCA,fun,X,bases,tree,isActiveNode,grids)
            % [f,outputs] = TBApproximation(FTPCA,fun,X,bases,tree,t,grids)
            % Approximation of a function in Tree Based tensor format based on
            % Principal Component Analysis for the estimation of subspaces
            % X: RandomVector of dimension d
            % bases: FunctionalBases or cell containing d FunctionalBasis
            % tree: DimensionTree
            % t: tolerance (if t is a double <1) or tuple containing the desired  alpha-ranks.
            % If numel(t)=1, the same rank is used for all alpha
            % isActiveNode: logical array indicating which nodes of the tree are active
            % grids: FullTensorGrid or cell containing d interpolation grids
            
            d = ndims(X);
            if length(FTPCA.tol)==1 && FTPCA.tol<1
                Copt = sqrt(tree.nbNodes-1);
                FTPCA.tol=FTPCA.tol/Copt;
            end
            if numel(FTPCA.tol)==1
                FTPCA.tol = repmat(FTPCA.tol,1,tree.nbNodes);
            end
            
            if isa(bases,'FunctionalBases')
                bases = bases.bases;
            end
            if nargin<8
                grids = [];
            end
            
            if ~isa(grids,'cell')
                grids = repmat({grids},1,d);
            end
            
            if ~all(fun.outputSize==1)
                error('Not implemented.')
            end
            
            alphaBasis = cell(1,tree.nbNodes);
            alphaGrids = cell(1,tree.nbNodes);
            outputs = cell(1,tree.nbNodes);
            samples = cell(1,tree.nbNodes);
            tensors = cell(1,tree.nbNodes);
            numberOfEvaluations = 0;
            
            for nu=1:d
                alpha = tree.dim2ind(nu);
                [grids{nu},nbeval] = FTPCA.alphaGrid(nu,X,bases{nu},grids{nu});
                numberOfEvaluations = numberOfEvaluations + nbeval;
                if isActiveNode(alpha)
                    if FTPCA.constantCorrected
                        FTPCA.tol(nu) = constantCorrection(FTPCA.tol(nu), alpha, tree , cardinal(bases{nu}));
                    end
                    [alphaBasis{alpha},outputs{alpha},bases{nu}] = ...
                        FTPCA.alphaFunctionalPrincipalComponents(fun,X,nu,bases{nu},grids{nu},FTPCA.tol(nu));
                    szalpha = [cardinal(bases{nu}),cardinal(alphaBasis{alpha})];
                    tensors{alpha} = FullTensor(alphaBasis{alpha}.basis,2,szalpha);
                    [subTree{alpha},nod{alpha}] = subDimensionTree(tree,alpha);
                    treeTensor{alpha} = TreeBasedTensor(tensors(nod{alpha}),subTree{alpha});
                    [alphaGrids{alpha},nbeval] = FTPCA.alphaGrid(nu,X,alphaBasis{alpha},outputs{alpha}.grid);
                    alphaBasis{alpha} = FunctionalTensorBasis(FunctionalTensor(treeTensor{alpha},FunctionalBases({bases{nu}})));
                    alphaBasis{alpha}.isOrthonormal = true;
                    % samples{alpha} = outputs{alpha}.grid;
                    numberOfEvaluations = numberOfEvaluations + outputs{alpha}.numberOfEvaluations;
                    if FTPCA.display
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
                    % alphaBasis{alpha} = FullTensorProductFunctionalBasis(alphaBasis(Salpha));
                    alphaBasis{alpha} = FunctionalBases(alphaBasis(Salpha));
                    alphaGrids{alpha} = FullTensorGrid(alphaGrids(Salpha));
                    if FTPCA.constantCorrected
                        FTPCA.tol(alpha) = constantCorrection(FTPCA.tol(alpha), alpha, tree , cardinal(FullTensorProductFunctionalBasis(alphaBasis(Salpha))));
                    end
                    [alphaBasis{alpha},outputs{alpha}] = ...
                        FTPCA.alphaFunctionalPrincipalComponents(fun,X,tree.dims{alpha},alphaBasis{alpha},alphaGrids{alpha},FTPCA.tol(alpha));
                    szalpha = [cellfun(@cardinal,alphaBasis(Salpha)),...
                        cardinal(alphaBasis{alpha})];
                    tensors{alpha} = FullTensor(alphaBasis{alpha}.basis,length(Salpha)+1,szalpha);
                    [subTree{alpha},nod{alpha}] = subDimensionTree(tree,alpha);
                    treeTensor{alpha} = TreeBasedTensor(tensors(nod{alpha}),subTree{alpha});
                    alphaBasis{alpha} = FunctionalTensorBasis(FunctionalTensor(treeTensor{alpha},FunctionalBases({bases{tree.dims{alpha}}}),tree.dims{alpha}));
                    alphaBasis{alpha}.isOrthonormal = true;
                    alphaGrids{alpha} = FTPCA.alphaGrid(tree.dims{alpha},X,alphaBasis{alpha},outputs{alpha}.grid);
                    numberOfEvaluations = numberOfEvaluations + outputs{alpha}.numberOfEvaluations;
                    if FTPCA.display
                        fprintf('alpha = %d: rank = %d, nbeval = %d\n',alpha,szalpha(end),outputs{alpha}.numberOfEvaluations);
                    end
                end
            end
            
            alpha=tree.root;
            %alpha = 1;
            Salpha = nonzeros(tree.children(:,alpha))';
            alphaBasis{alpha} = FullTensorProductFunctionalBasis(alphaBasis(Salpha));
            %alphaBasis{alpha} = FunctionalBases(alphaBasis(Salpha));
            alphaGrids{alpha} = FullTensorGrid(alphaGrids(Salpha));
            [~,I]=ismember(1:d,tree.dims{alpha});
            projection = FTPCA.getalphaProjection(tree.dims{alpha},alphaBasis{alpha},alphaGrids{alpha});
            s.type = '()';
            s.subs = {':',I};
            froot = @(x) fun.eval(subsref(x,s));
            froot = UserDefinedFunction(froot,d);
            tensors{alpha} = projection(froot);
            szalpha = [cellfun(@cardinal,alphaBasis(Salpha)),1];
            tensors{alpha} = FullTensor(tensors{alpha}.data,length(Salpha)+1,szalpha);
            f = TreeBasedTensor(tensors,tree);
            f = FunctionalTensor(f,FunctionalBases(bases));
            output.numberOfEvaluations = numberOfEvaluations + cardinal(FullTensorProductFunctionalBasis(alphaBasis(Salpha)));
            output.samples = samples;
            output.alphaBasis = alphaBasis;
            output.alphaGrids = alphaGrids;
            output.outputs = outputs;
        end
        
        function [Ualpha, output, Valpha] = alphaFunctionalPrincipalComponents(FTPCA,fun,X,alpha,Valpha,gridalpha,tol, varargin)
            % [fpc,output] = alphaFunctionalPrincipalComponents(FTPCA,fun,X,alpha,Valpha,t,gridalpha)
            % For alpha in {1,...,d}, it evaluates the alpha-principal components of the
            % function fun, that means  the principal components of
            % the matricisations f_\alpha(x_\alpha,x_notalpha) of fun.
            % It evaluates f_\alpha on the product of a grid from sampling (random or optimal) in dimension
            % alpha and a random grid (N samples) in the complementary dimensions.
            % Then, it computes approximations of functional alpha-principal components
            % in a given basis of functions (by least-squares regression).
            %             % Then, it computes an approximation of f_\alpha by a (standard or
            %             % optimal) least-squares regression in a given basis of functions.
            %             % The functional alpha-principal components are computed from
            %             % this approximation.
            % If t is an integer, t is the number of principal components
            % If t<1, the number of principal components is determined such
            % that the relative error after truncation is t.
            % fun: function of d variables x_1,...,x_d
            % X: RandomVector of dimension d
            % alpha: set of indices in {1,...,d}
            % Valpha: FunctionalBasis
            % t: number of components or a positive number <1 (tolerance)
            % gridalpha: array of size n-by-#alpha
            %
            % FPCA.PCASamplingFactor : factor for determining the number of samples N for the estimation of principal
            %         components (1 by default)
            %            - if t<1, N = FPCA.PCASamplingFactor*cardinal(basis)
            %            - if t is the rank, N=FPCA.PCASamplingFactor*t
            % fpc: functional principal components obtained by regression on the given FunctionalBasis
            % output.pcOnGrid: principal components on the grid gridalpha
            % output.sv: corresponding singular values
            % output.grid: interpolation grid in dimension alpha
            d = ndims(X);
            notalpha = setdiff(1:d,alpha);
            Xnotalpha = RandomVector(X.randomVariables(notalpha));
            falpha = alphaFunction(fun,alpha,notalpha);
            falpha.measure = Xnotalpha;
            falpha.store = true;
            [PValpha,outputprojection] = FTPCA.getalphaProjection(alpha,Valpha,gridalpha);
            Valpha = outputprojection.basis;
            [~,I]=ismember(1:d,[alpha,notalpha]);
            s.type = '()';
            s.subs = {':',I};
            if FTPCA.basisAdaptation == true && numel(alpha) == 1
                [a, b] = PValpha(random(falpha));
                outputprojection.numberOfEvaluations = b.numberOfEvaluations;
                % Valpha = a.basis;
            end
            F2 = RandomFunction(@()wrapperGetCoefficients(PValpha(random(falpha))), cardinal(Valpha));
            [Ualpha, outputpca] = principalComponents(FTPCA.vpca, F2, tol);
            if FTPCA.basisAdaptation == true && numel(alpha) == 1
                Valpha = PolynomialFunctionalBasis(Valpha.basis,(0:(size(Ualpha,1)-1)));
            end
            Ualpha = SubFunctionalBasis(Valpha,Ualpha);
            Ualpha.isOrthonormal = true;
            output.numberOfEvaluations = outputpca.numberOfSamples*outputprojection.numberOfEvaluations;
            output.grid = gridalpha;
        end
    end
end

function [s, dimb] = wrapperGetCoefficients(f)
dimb = cardinal(f.basis);
s = reshape(f.data,[dimb,f.sz]);

end

function Cnew = constantCorrection(Cini, alpha, tree , dimValpha)
delta = 0.5;
eta = 0.01;
cdelta = -delta + (1+delta).*(log(1+delta));
M = 1;
l = tree.level(alpha);
C = 1/2*(4*(1+cdelta^(-1)*log(2*dimValpha/eta)*(1-delta)^(-1)*(1-eta^M)^(-1)*M)).^(l+1);
C = 1/2*(1+(1-delta)^(-1)*(1-eta)^(-1)).^(l+1);
Cnew = Cini./C;
end