% Class FunctionalBases

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

classdef FunctionalBases
    
    properties
        bases
        measure
    end
    
    methods
        function H = FunctionalBases(bases)
            % Class FunctionalBases
            %
            % H = FunctionalBases(bases)
            % bases: cell of length d containing objects of type 
            % FunctionalBasis
            %
            % To create a FunctionalBases by replication of
            % FunctionalBasis, see FunctionalBases.duplicate
            
            switch nargin
                case 0
                    H.bases = {};
                case 1
                    if isa(bases,'FunctionalBases')
                        H = bases;
                    elseif iscell(bases) && all(cellfun(@(x) isa(x,'FunctionalBasis'),bases))
                        H.bases = bases(:);
                        H.measure = ProductMeasure(cellfun(@(x) x.measure,bases,'UniformOutput',false));
                    else
                        error('Wrong argument.')
                    end
                otherwise
                    error('Wrong number of input parameters.')
            end
            if ~isempty(H.bases) && ~all(cellfun(@(x) isa(x,'FunctionalBasis'),H.bases))
                error('Property bases must contain object of type FunctionalBasis.')
            end
        end
        
        function n = number(f)
            % n = number(f)
            % Returns the number of bases in f
            % f: FunctionalBases
            % n: 1-by-1 double
            
            warning('Will be replaced by the method length in a future release.')
            n = length(f.bases);
        end
        
        function n = length(f)
            % n = length(f)
            % Returns the number of bases in f
            % f: FunctionalBases
            % n: 1-by-1 double
            
            n = length(f.bases);
        end
        
        function n = numel(f,varargin)
            warning('numel is replaced by cardinals and will be removed in a future release.')
            n = cardinals(f,varargin{:});
        end
        
        function n = cardinals(f,k)
            % n = cardinals(f,k)
            % Returns the number of functions in each basis of f, or in
            % basis k if k is provided
            % f: FunctionalBases
            % k: 1-by-n or n-by-1 double, n <= d, d = length(f)
            % n: d-by-1 or n-by-1 double
            
            if nargin==1
                n = cellfun(@cardinal,f.bases);
            else
                n = cellfun(@cardinal,f.bases(k));
            end
        end
        
        function n = ndims(f)
            % n = ndims(f)
            % Returns the dimension of each basis of f
            % f: FunctionalBases
            % n: d-by-1 double
            
            n = cellfun(@ndims,f.bases);
        end
        
        function D = domain(f)
            % D = domain(f)
            % Returns the domain of each basis of f
            % f: FunctionalBases
            % D: d-by-1 cell containing 1-by-2 doubles
            
            D = cellfun(@domain,f.bases,'UniformOutput',false);
        end
        
        function f = orthonormalize(f)
            % f = orthonormalize(f)
            % Orthonormalizes the basis functions of f
            % f: FunctionalBases
            
            f.bases = cellfun(@orthonormalize,f.bases,'UniformOutput',false);
        end
        
        function h = kron(f,g)
            h=f;
            h.bases = cellfun(@(x,y) kron(x,y),f.bases,g.bases,'UniformOutput',false);
        end
        
        function f = removeBases(f,k)
            % f = removeBases(f,k)
            % Removes bases of f of index k
            % f: FunctionalBases
            % k: 1-by-n or n-by-1 double
            
            f.bases(k) = [];
        end
        
        function f = keepBases(f,k)
            % f = removeBases(f,k)
            % Keeps only bases of f of index k
            % f: FunctionalBases
            % k: 1-by-n or n-by-1 double
            
            f.bases = f.bases(k);
        end
        
        function f = permute(f,num)
            % f = permute(f,p)
            % Returns f with the basis permutation num
            % f: FunctionalBases
            % p: array containing a permutation of 1:numel(f)
            
            f.bases = f.bases(num);
        end
        
        function ok = eq(f,g)
            % ok = eq(f,g)
            % Checks if the two FunctionalBases f and g are identical
            % f: FunctionalBases
            % g: FunctionalBases
            % ok: boolean
            
            if ~isa(g,'FunctionalBases')
                ok = 0;
            else
                ok = all(cellfun(@(x,y) x == y, f.bases, g.bases));
                ok = ok & f.measure == g.measure;
            end
        end
        
        function a = adaptationPath(f,varargin)
            % a = adaptationPath(f,varargin)
            % Computes the adaptationPath for each basis in f
            % f: FunctionalBases
            % See also FunctionalBasis.adaptationPath
            
            a = cellfun(@(x) adaptationPath(x,varargin{:}),f.bases,'UniformOutput',false);
        end
        
        function [fx,x] = eval(f,x,dims)
            % [fx,x] = eval(f,x,dims)
            % Computes evaluations of the basis functions of f at points x 
            % in dimensions dims if provided, in all the dimensions if not
            % f: FunctionalBases
            % dims: d-by-1 or 1-by-d double (optional)
            %
            % x: n-by-length(f) double
            % fx: d-by-1 cell containing doubles of size n-by-p, where p is
            % the number of basis functions in each dimension
            % or 
            % x: 1-by-length(f) cell, with x{i} containing a array of size
            % ni-by-1
            % fx: d-by-1 cell containing doubles of size ni-by-pi, where 
            % pi is the number of basis functions in each dimension
            
            if nargin==2
                dims = 1:length(f);
            end
            fx = cell(numel(dims),1);
            
            if ~isa(x,'cell')
                n=ndims(f);
                x = mat2cell(x,size(x,1),n(dims));
            end
            
            for i = 1:numel(dims)
                fx{i} = eval(f.bases{dims(i)},x{i});
            end
            
        end
        
        function [fx,x] = evalDerivative(f,n,x,dims)
            % [fx,x] = evalDerivative(f,n,x,dims)
            % Computes evaluations of the n-derivative of the basis 
            % functions of f at points x in each dimension in dims if 
            % provided, in all the dimensions if not
            % f: FunctionalBases
            % n: 1-by-d or 1-by-length(dims) array of integers 
            % dims: m-by-1 or 1-by-m double (optional)
            %
            % x: n-by-length(f) double
            % fx: d-by-1 or length(dims)-by-1 cell containing doubles of 
            % size n-by-p, where p is the number of basis functions in 
            % each dimension
            % or 
            % x: 1-by-length(f) cell, with x{i} containing a array of size
            % ni-by-1
            % fx: d-by-1 or length(dims)-by-1 cell containing doubles of 
            % size ni-by-pi, where pi is the number of basis functions in
            % each dimension
            
            if nargin==3
                dims = 1:length(f);
            end
            fx = cell(numel(dims),1);
            
            if ~isa(x,'cell')
                nd = ndims(f);
                x = mat2cell(x,size(x,1),nd(dims));
            end
            
            for i = 1:numel(dims)
                fx{i} = evalDerivative(f.bases{dims(i)},n(i),x{i});
            end
        end
        
        function df = derivative(f,n)
            % df = derivative(f,n)
            % Computes the n-derivative of the basis functions of f
            % f: FunctionalBases
            % n: 1-by-d array of integers
            % df: FunctionalBases
            
            df = f;
            for i = 1:length(f)
                df.bases{i} = derivative(f.bases{i},n(i));
            end
        end
        
        function [H,x] = random(f,varargin)
            % [H,x] = random(f)
            % Computes random evaluations of the bases in f
            % f: FunctionalBases
            % H: d-by-1 cell containing N1-by-P doubles
            % x: N-by-1 double
            % See also randomDims
            
            dims = 1:length(f);
            [H,x] = randomDims(f,dims,varargin{:});
        end
        
        function [H,x] = randomDims(f,dims,n,rv)
            % [H,x] = randomDims(f,dims,n,rv)
            % Evaluates the bases in dimensions dims of the bases of f
            % using n points generated with rv
            % f: FunctionalBases
            % dims: d-by-1 or 1-by-d double
            % n: N-by-1 double
            % rv: RandomVector or RandomVariable
            % H: d-by-1 cell containing N-by-P doubles
            % x: N-by-1 double
            % See also eval
            
            if nargin==2
                n=1;
            end
            if nargin==4
                if ~isa(rv,'ProbabilityMeasure')
                    error('Must provide a ProbabilityMeasure.')
                end
                
                x = random(marginal(rv,dims),n);
                [H,x] = eval(f,x,dims);
                
                if length(H) == 1
                    H = H{1};
                end
            else
                [H,x] = randomDims(f,dims,n,f.measure);
            end
        end
        
        function rv = getRandomVector(f)
            % rv = getRandomVector(f)
            % Returns the random vector rv associated with the functional 
            % bases f
            % f: FunctionalBases
            % rv: RandomVector
            
            if isa(f.measure,'RandomVector')
                rv = f.measure;
            elseif isa(f.measure,'ProbabilityMeasure')
                rv = randomVector(f.measure);
            else
                rv = [];
            end
        end
        
        function H = one(f,dims)
            % H = one(f,dims)
            % Returns a FunctionalBases returning one
            % f: FunctionalBases
            % dims: d-by-1 or 1-by-d double
            % H: FunctionalBases
            
            if nargin==1
                dims = 1:length(f);
            end
            H = cell(numel(dims),1);
            for i=1:numel(dims)
                H{i} = one(f.bases{dims(i)});
            end
        end
        
        function M = gramMatrix(f,dims)
            % M = gramMatrix(f,dims)
            % f: FunctionalBases
            % dims: n-by-1 or 1-by-n double (optional)
            % M: n-by-1 cell containing arrays of doubles
            
            if nargin == 1 || isempty(dims)
                dims = 1:length(f);
            end
                   
            M = cell(length(dims),1);
            for i = 1:length(dims)
                M{i} = gramMatrix(f.bases{dims(i)});
            end
        end
        
        function s = storage(f)
            % s = storage(f)
            % Returns the storage requirement of the FunctionalBases
            % f: FunctionalBases
            % s: 1-by-1 double
            
            s=0;
            for k=1:length(f)
                s=s+storage(f.bases{k});
            end
        end
        
        function H = mean(f,dims,rv)
            % H = mean(f,dims,rv)
            % Computes the mean of f in the dimensions in dims according to
            % the RandomVector rv if provided, or to the standard
            % RandomVector associated with each basis if not.
            % If dims is not provided or empty, the mean is computed for
            % all the dimensions.
            % f: FunctionalBases
            % dims: d-by-1 or 1-by-d double (optional)
            % rv: RandomVector or RandomVariable (optional)
            % H: d-by-1 cell containing doubles of size the number of basis
            % functions in each dimension
            
            if nargin==1 || isempty(dims)
                dims = 1:length(f);
            end
            H = cell(numel(dims),1);
            if nargin == 3 && isa(rv,'RandomVector')
                for i=1:numel(dims)
                    H{i} = mean(f.bases{dims(i)},rv.randomVariables{i});
                end
            elseif nargin == 3 && isa(rv,'RandomVariable')
                for i=1:numel(dims)
                    H{i} = mean(f.bases{dims(i)},rv);
                end
            else
                for i=1:numel(dims)
                    H{i} = mean(f.bases{dims(i)});
                end
            end
        end
        
        function points = interpolationPoints(f,x)
            if nargin>=2
                if isa(x,'FullTensorGrid')
                    x = x.grids;
                end
                if isa(x,'double')
                    x = mat2cell(x,size(x,1),ones(1,size(x,2)));
                end
            end
            points = cell(1,length(f));
            for k=1:length(f)
                if nargin==1
                    points{k}= interpolationPoints(f.bases{k});
                elseif nargin==2
                    points{k}= interpolationPoints(f.bases{k},x{k});
                end
            end
        end
        
        function [finterp,output] = tensorProductInterpolation(H,varargin)
            % function [finterp,output] = tensorProductInterpolation(H,fun,grid)
            % Interpolates a function on a product grid
            % Returns a FunctionalTensor
            %
            % See also FullTensorProductFunctionalBasis.tensorProductInterpolation
            H = FullTensorProductFunctionalBasis(H);
            [finterp,output] = tensorProductInterpolation(H,varargin{:});
        end
        
        function [points,I] = magicPoints(f,x,J)
            % [points,I] = magicPoints(f,x)
            % Provides the magic points associated with the functional bases f
            % selected in a given set of points x
            % f: FunctionalBases
            % x: cell array such that x{nu} contains a set of points in dimension nu, or FullTensorGrid
            % points: cell array such that points{nu} contains the magic points
            % associated with h.bases{nu}
            % I: cell array such that I{nu} contains the locations of the
            % magic points points{nu} in X{nu}
            % Use [points{nu},I{nu}] = magicPoints(f.bases{nu},x{nu})
            %
            % function [points,I] = magicPoints(f,x,J)
            % J: cell array
            % Use [points{nu},I{nu}] = magicPoints(f.bases{nu},x{nu},J{nu})
            
            if isa(x,'FullTensorGrid')
                x = x.grids;
            end
            if isa(x,'double')
                x = mat2cell(x,size(x,1),ones(1,size(x,2)));
            end
            points = cell(1,length(f));
            I = cell(1,length(f));
            for k=1:length(f)
                if nargin<=2
                    [points{k},I{k}]= magicPoints(f.bases{k},x{k});
                else
                    [points{k},I{k}]= magicPoints(f.bases{k},x{k},J{k});
                end
            end
        end
    end
    
    methods (Static)
        function H = duplicate(basis,d)
            % H = duplicate(basis,d)
            % Creates a FunctionalBases with bases created with a
            % duplication of basis d times
            % basis: FunctionalBasis
            % d: 1-by-1 double
            % H: FunctionalBases
            
            basis = repmat({basis},1,d);
            H = FunctionalBases(basis);
        end
    end
end