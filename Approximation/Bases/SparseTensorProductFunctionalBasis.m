% Class SparseTensorProductFunctionalBasis

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

classdef SparseTensorProductFunctionalBasis < FunctionalBasis
    
    properties
        bases
        indices
    end
    
    methods
        
        function H = SparseTensorProductFunctionalBasis(bases,indices)
            % Class SparseTensorProductFunctionalBasis
            %
            % H = SparseTensorFunctionalBasis(bases,indices)
            % bases: FunctionalBases
            % indices: MultiIndices (indices start at 0)
            
            if ~isa(bases,'FunctionalBases')
                error('first argument must be a FunctionalBases')
            end
            
            if ~isa(indices,'MultiIndices')
                error('second argument must be a MultiIndices')
            end
            
            H.bases = bases;
            H.indices = indices;
            H.measure = ProductMeasure(cellfun(@(x) x.measure,bases.bases,'uniformoutput',false));
            H.isOrthonormal = all(cellfun(@(x) x.isOrthonormal,bases.bases));
        end
        
        function n = length(H)
            n = length(H.bases);
        end
        
        function n = cardinal(H)
            n = cardinal(H.indices);
        end
        
        function n = ndims(H)
            n = ndims(H.indices);
        end
        
        function D = domain(H)
            D = domain(H.bases);
        end
        
        function H = removeBases(H,k)
            H.bases = removeBases(H.bases,k);
        end
        
        function H = keepBases(H,k)
            H.bases = keepBases(H.bases,k);
        end
        
        function H = removeMapping(H,k)
            if isa(H.bases,'FunctionalBasesWithMappedVariables')
                H.bases = removeMapping(H.bases,k);
            end
        end
        
        function H = keepMapping(H,k)
            if isa(H.bases,'FunctionalBasesWithMappedVariables')
                H.bases = keepMapping(H.bases,k);
            end
        end
        
        function H = permute(H,num)
            H.bases = permute(H.bases,num);
        end
        
        function ok = eq(H,G)
            % ok = eq(H,G)
            % Checks if the two objects H and G are identical
            % H: SparseTensorProductFunctionalBasis
            % G: SparseTensorProductFunctionalBasis
            % ok: boolean
            
            if ~isa(G,'SparseTensorProductFunctionalBasis')
                ok = 0;
            else
                try
                    ok = all(H.bases == G.bases) & all(H.indices == G.indices);
                catch
                    ok = 0;
                end
            end
        end
        
        function y = eval(H,x)
            % y = eval(H,x)
            % Computes evaluations of the basis functions of H at points x
            % H: SparseTensorProductFunctionalBasis
            % x: n-by-number(H) double
            % y: n-by-cardinal(H) double
            
            Hx = eval(H.bases,x);
            y = evalWithFunctionalBasesEvals(H,Hx);
            
        end
        
        function y = evalWithFunctionalBasesEvals(H,Hx)
            % function y = evalWithFunctionalBasesEval(H,Hx)
            y = Hx{1}(:,H.indices.array(:,1)+1);
            for i=2:length(Hx)
                y = y.*Hx{i}(:,H.indices.array(:,i)+1);
            end
        end
        
        function y = evalDerivative(H,n,x)
            % y = evalDerivative(f,n,x)
            % Computes the n-derivative of f at points x in R^d, with n a
            % multi-index of size d
            % f: SparseTensorProductFunctionalBasis
            % n: 1-by-d array of integers
            % x: N-by-d array of doubles
            
            dnHx = evalDerivative(H.bases,n,x);
            y = evalWithFunctionalBasesEvals(H,dnHx);
        end
        
        function H = derivative(H,n)
            % df = derivative(f,n)
            % Computes the n-derivative of f
            % f: FunctionalBasisArray
            % n: 1-by-d array of integers
            % df: FunctionalBasisArray
            
            H.bases = derivative(H.bases,n);
        end
        
        function P = adaptationPath(H,p)
            % P = adaptationPath(H,p)
            % Creates an adaptation path associated with increasing p-norm of
            % multi-indices
            % H: SparseTensorProductFunctionalBasis
            % p: positive real scalar, p=1 by default
            
            if nargin==1
                p=1;
            end
            n = norm(H.indices,p);
            l = sort(unique(n));
            P = false(cardinal(H.indices),length(l));
            for k=1:length(l)
                P(n<=l(k),k)=true;
            end
            
        end
        
        function m = mean(H,varargin)
            % m = mean(H,rv)
            % Computes the mean of the basis functions of H, according to
            % the ProbabilityMeasure rv if provided, or to the standard RandomVector
            % associated with each polynomial if not
            % H: SparseTensorProductFunctionalBasis
            % rv: RandomVector or RandomVariable (optional)
            % m: P-by-1 double, where P is the number of basis functions
            
            M = mean(H.bases,[],varargin{:});
            m = M{1}(H.indices.array(:,1)+1);
            for i = 2:length(M)
                m = m.*M{i}(H.indices.array(:,i)+1);
            end
        end
        
        function h = conditionalExpectation(h,dims,varargin)
            % h = conditionalExpectation(h,dims,XdimsC)
            %
            % See also FunctionalBasis/conditionalExpectation
            
            
            if isa(dims,'logical')
                dims = find(dims);
            end
            
            dimsC = setdiff(1:ndims(h),dims);
            
            if ~isempty(dimsC)
                M = mean(h.bases,dimsC,varargin{:});
                I = h.indices;
                J = keepDims(I,dims);
                m = M{1}(I.array(:,dimsC(1))+1);
                for i=2:numel(dimsC)
                    m = m.*M{i}(I.array(:,dimsC(i))+1);
                end
                
                if isempty(dims)
                    h=m;
                    return
                end
                
                [~,i] = ismember(I.array(:,dims),J.array,'rows');
                d = sparse(i,1:cardinal(I),m,cardinal(J),cardinal(I));
                h = keepBases(h,dims);
                h = keepMapping(h,dims);
                h.indices = J;
                h = FunctionalBasisArray(d,h,[cardinal(I),1]);
            else
                h = FunctionalBasisArray(speye(cardinal(h)),h,[cardinal(h),1]);
            end
            
            
        end
        
        function rv = getRandomVector(H)
            % rv = getRandomVector(H)
            % Gets the random vector rv associated with the basis functions of H
            % H: SparseTensorProductFunctionalBasis
            % rv: RandomVector
            
            rv = getRandomVector(H.bases);
        end
        
        function M = gramMatrix(H)
            if H.isOrthonormal
                M = speye(cardinal(H.indices));
            else
                G = gramMatrix(H.bases);
                I = H.indices.array+1;
                M = G{1}(I(:,1),I(:,1));
                for i=2:size(I,2)
                    M=M.*G{i}(I(:,i),I(:,i));
                end
            end
        end
        
        function plotMultiIndices(H,varargin)
            % plotMultiIndices(H)
            % Plots the multi-index set of H
            % H: SparseTensorProductFunctionalBasis
            
            plot(H.indices,varargin{:});
        end
        
        function [finterp,output] = tensorProductInterpolation(H,fun,varargin)
            % [finterp,output] = tensorProductInterpolation(H,fun,grids)
            % Interpolation of function fun on a sparse grid
            %
            % Inputs:
            % H: SparseTensorProductFunctionalBasis
            % fun: function of d variables, d = ndims(H)
            % grids: cell containing d grids or FullTensorGrid
            % if one grid have more points than the dimension of the
            % corresponding basis, use magicPoints for the selection of a subset of points
            % adapted to the basis
            %
            % Outputs:
            % finterp: FunctionalTensor
            % output.numberOfEvaluations: number of evaluations of the
            % function
            
            grid = interpolationPoints(H.bases,varargin{:});
            grid = SparseTensorGrid(grid,H.indices+1);
            xgrid = array(grid);
            finterp = H.interpolate(fun,xgrid);
            output.numberOfEvaluations = size(xgrid,1);
            output.grid = grid;
            
        end
    end
    
    methods (Static)
        
    end
end