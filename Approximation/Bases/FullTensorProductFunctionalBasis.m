% Class FullTensorProductFunctionalBasis

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

classdef FullTensorProductFunctionalBasis < FunctionalBasis
    
    properties
        bases
    end
    
    methods
        
        function H = FullTensorProductFunctionalBasis(bases)
            % Class FullTensorProductFunctionalBasis
            %
            % H = FullTensorProductFunctionalBasis(bases)
            % bases: FunctionalBases or cell containing FunctionalBasis
            
            if isa(bases,'cell')
                bases = FunctionalBases(bases);
            end
            
            if ~isa(bases,'FunctionalBases')
                error('first argument must be a FunctionalBases')
            end
            
            H.bases = bases;
            H.measure = ProductMeasure(cellfun(@(x) x.measure,bases.bases,'uniformoutput',false));
            H.isOrthonormal = all(cellfun(@(x) x.isOrthonormal,bases.bases));
        end
        
        function n = length(H)
            n = length(H.bases);
        end
        
        function n = cardinal(H)
            n = prod(cellfun(@cardinal,H.bases.bases));
        end
        
        function n = ndims(H)
            n = sum(ndims(H.bases));
        end
        
        function D = domain(H)
            D = domain(H.bases);
        end
        
        function H = orthonormalize(H)
            H.bases = orthonormalize(H.bases);
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
        
        function H = mean(H,varargin)
            H = mean(H.bases,[],varargin{:});
            H = CanonicalTensor(H,1);
            H = double(full(H));
            H = H(:);
        end
        
        function h = conditionalExpectation(h,dims,varargin)
            % h = conditionalExpectation(h,dims,XdimsC)
            %
            % See also FunctionalBasis/conditionalExpectation
            
            I = MultiIndices.boundedBy(numel(h.bases)-1);
            h = SparseTensorProductFunctionalBasis(h.bases,I);
            h = conditionalExpectation(h,dims,varargin{:});
        end
        
        function ok = eq(H,G)
            % ok = eq(H,G)
            % Checks if the two objects H and G are identical
            % H: FullTensorProductFunctionalBasis
            % G: FullTensorProductFunctionalBasis
            % ok: boolean
            
            if ~isa(G,'FullTensorProductFunctionalBasis')
                ok = 0;
            else
                ok = (H.bases == G.bases);
            end
        end
        
        function y = eval(H,x)
            % y = eval(H,x)
            % Computes evaluations of the basis functions of H at points x
            % H: FullTensorProductFunctionalBasis
            % x: n-by-number(H) double
            % y: n-by-numel(H) double
            
            Hx = eval(H.bases,x);
            I = MultiIndices.boundedBy(cardinals(H.bases),1);
            y = Hx{1}(:,I.array(:,1));
            for i=2:length(Hx)
                y = y.*Hx{i}(:,I.array(:,i));
            end
        end
        
        function rv = getRandomVector(H)
            % rv = getRandomVector(H)
            % Gets the random vector rv associated with the basis functions of H
            % H: FullTensorProductFunctionalBasis
            % rv: RandomVector
            
            warning('getRandomVector method will be removed in a future release. Use the property measure instead.')
            rv = getRandomVector(H.bases);
        end
        
        function M = gramMatrix(H,varargin)
            % M = gramMatrix(H,dims,rv)
            % Computes the cell of the Gram matrices in each dimension in
            % dims if provided, in all the dimensions if not. The Gram
            % matrix is the matrix of the dot products between each
            % possible couple of basis functions of H. The dot product in
            % the dimension i is computed according to the i-th
            % RandomVariable of the RandomVector rv if provided, or according to the
            % RandomVariable in H if not.
            % H: FullTensorProductFunctionalBasis
            % dims: n-by-1 or 1-by-n double (optional)
            % rv: RandomVector containing d RandomVariable (optional)
            % M: n-by-1 cell containing arrays of doubles
            % See FunctionalBases/gramMatrix
            
            M = gramMatrix(H.bases,varargin{:});
        end
        
        function [fproj,output] = projection(H,fun,I)
            % [fproj,output] = projection(H,fun,I)
            % Projection of function fun on the basis functions of H
            %
            % H: FullTensorProductFunctionalBasis
            % fun: function
            % I: IntegrationRule
            % fproj: FunctionalTensor
            % output.numberOfEvaluations: number of evaluations of the function
            
            if isa(I,'FullTensorProductIntegrationRule')
                u = fun.evalOnTensorGrid(I.points);
                output.numberOfEvaluations = numel(u);
                Hx = eval(H.bases,[I.points.grids{:}]);
                Mx = cellfun(@(x,y) x'*diag(y)*x,Hx,I.weights,'UniformOutput',false);
                HxW = cellfun(@(m,x,y) m\(x'*diag(y)),Mx,Hx,I.weights,'UniformOutput',false);
                fdims = 1:numel(HxW);
                % fdims = (u.order-numel(HxW)+1):u.order;
                ucoeff = timesMatrix(u,HxW,fdims);
                fproj = FunctionalTensor(ucoeff,H.bases,fdims);
            else
                error('not implemented')
            end
        end
        
        function nu = optimalSamplingMeasure(f)
            B = f.bases.bases;
            nu = cell(1,length(B));
            for i =1:length(B)
                nu{i} = optimalSamplingMeasure(B{i});
            end
            nu = ProductMeasure(nu);
        end
        
        function [points,grid] = interpolationPoints(H,varargin)
            grid = interpolationPoints(H.bases,varargin{:});
            points = array(FullTensorGrid(grid));
        end
        
        function [finterp,output] = tensorProductInterpolation(H,fun,grid)
            % [finterp,output] = tensorProductInterpolation(H,fun,grids)
            % Interpolation of function fun on a product grid
            %
            % H: FullTensorProductFunctionalBasis
            % fun: - function of d variables, d = ndims(H)
            %      - or tensor of order d whose entries are the evaluations of 
            %           the function on a product grid 
            % grids: cell containing d grids or FullTensorGrid
            % if one grid have more points than the dimension of the
            % corresponding basis, use magicPoints for the selection of a subset of points
            % adapted to the basis
            % 
            % finterp: FunctionalTensor
            % output.numberOfEvaluations: number of evaluations of the
            % function
            
            if nargin==2
                grid = interpolationPoints(H.bases);
            else
                grid = interpolationPoints(H.bases,grid);
            end
            grid = FullTensorGrid(grid);
            
            if isa(fun,'Function') || isa(fun,'function_handle')              
                xgrid = array(grid);
                y = fun(xgrid);
                y = FullTensor(y,grid.dim,grid.sz);
                output.numberOfEvaluations = numel(y);
            elseif isa(fun,'AlgebraicTensor')
                y = fun;
            else
                error('argument fun should be a Function, function_handle, or an AlgebraicTensor')                
            end
            output.grid = grid;
            
            B = eval(H.bases,grid.grids);
            B = cellfun(@(x) ImplicitMatrix.inverse(x),B,'UniformOutput',false);
            y = timesMatrix(y,B);
            finterp = FunctionalTensor(y,H.bases);
        end
    end
end