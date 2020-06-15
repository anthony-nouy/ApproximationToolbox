% Class FunctionalBasisArray

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

classdef FunctionalBasisArray < Function
    
    properties
        data
        basis
        sz
    end
    
    methods
        function x = FunctionalBasisArray(data,basis,sz)
            % Class FunctionalBasisArray
            %
            % x = FunctionalBasisArray(data,basis)
            % Function with value in R
            % f: FunctionalBasisArray
            % basis: FunctionalBasis
            % data: array of size [cardinal(basis),1]
            % containing the coefficents of the function on the basis
            %
            % x = FunctionalBasisArray(data,basis,sz)
            % Function with value in R^(sz(1)xsz(2)x...)
            % data: array of size [cardinal(basis),sz] containing the
            % coefficents of the function on the basis
            
            if nargin==0
                data=[];
                basis=[];
                sz=[];
            elseif nargin<=2
                sz=[1,1];
            end
            x.data = data(:);
            x.basis = basis;
            x.sz = sz;
            x.outputSize = sz;
            x.data = reshape(x.data,[cardinal(basis),sz]);
            
        end
        
        function f = plus(f,g)
            % f = plus(f,g)
            % f: FunctionalBasisArray
            % g: FunctionalBasisArray
            
            f.data = f.data + g.data;
        end
        
        function f = uminus(f)
            % f = uminus(f)
            % f: FunctionalBasisArray
            
            f.data = - f.data ;
        end
        
        function f = minus(f,g)
            % f = minus(f,g)
            % f: FunctionalBasisArray
            % g: FunctionalBasisArray
            
            f = plus(f,-g);
        end
        
        function f = times(f,v)
            % f = times(f,v)
            % f: FunctionalBasisArray
            % v: FunctionalBasisArray or double
            
            if isa(f,'FunctionalBasisArray') && isa(v,'FunctionalBasisArray')
                f.data = f.data .* v.data;
            elseif isa(f,'FunctionalBasisArray') && isa(v,'double')
                f.data = f.data * v;
                s = size(f.data);
                f.sz = s(2:end);
            elseif isa(f,'double') && isa(v,'FunctionalBasisArray')
                f = times(v,f);
            else
                error('Not implemented')
            end
        end
        
        function f = mtimes(f,v)
            % f = mtimes(f,v)
            % Multiplies the double f.data by v
            % f: FunctionalBasisArray
            % v: double
            
            if isa(f,'FunctionalBasisArray') && isa(v,'double')
                f.data = f.data * v;
                s = size(f.data);
                f.sz = s(2:end);
            elseif isa(f,'double') && isa(v,'FunctionalBasisArray')
                f = mtimes(v,f);
            else
                error('Not implemented')
            end
        end
        
        function f = mrdivide(f,v)
            % f = mrdivide(f,v)
            % Right divides the double f.data by v
            % f: FunctionalBasisArray
            % v: double
            
            if isa(f,'FunctionalBasisArray') && isa(v,'double')
                f.data = f.data / v;
            else
                error('Not implemented')
            end
        end
        
        function c = dot(f,g,varargin)
            % c = dot(f,g)
            % Computes the dot product between the arrays f.data and g.data
            % treated as collections of vectors. The function calculates
            % the dot product of corresponding vectors along the first
            % array dimension whose size does not equal 1.
            %
            % c = dot(f,g,dim)
            % Evaluates the scalar dot product of f.data and g.data along dimension dim
            % f: FunctionalBasisArray
            % g: FunctionalBasisArray
            % dim: positive integer scalar (optional)
            % c: 1-by-s double, where s is the size of f
            
            c = dot(f.data,g.data,varargin{:});
        end
        
        
        function n = norm(f,varargin)
            % n = norm(f,p)
            % Computes the p-norm of the array f.data
            % f: FunctionalBasisArray
            % p: positive integer scalar or Inf or -Inf or 'fro' (optional), 'fro' by default
            % n: 1-by-1 double
            
            if nargin == 1
                n = norm(f,'fro');
            else
                n = norm(f.data(:,:),varargin{:});
            end
        end
        
        function x = israndom(~)
            % x = israndom(f)
            % Determines if input f is random
            % f: FunctionalBasisArray
            % x: boolean
            
            x = 1;
        end
        
        function m = mean(f,varargin)
            % m = mean(f,rv)
            % Computes the expectation of f, according to the measure
            % associated with the RandomVector rv if provided, or to the
            % standard RandomVector associated with each polynomial if not
            % f: FunctionalBasisArray
            % rv: RandomVector or RandomVariable (optional)
            % m: 1-by-s double, where s is the size of f
            
            M = mean(f.basis,varargin{:});
            m = dot(f.data,repmat(M,[1,f.sz]),1);
        end
        
        function m = expectation(f,varargin)
            % m = expectation(f)
            % Computes the expectation of f
            % f: FunctionalBasisArray
            % See FunctionalBasisArray/mean
            m = mean(f,varargin{:});
        end
        
        function v = variance(f,varargin)
            % v = variance(f,X)
            % Computes the variance of the random variable f(X).
            % If X is not provided, uses
            % random variables associated with underlying basis.
            % f: FunctionalBasisArray
            % X: RandomVector (optional)
            % v: 1-by-si double, where si is the size of f
            
            m = expectation(f,varargin{:});
            v = dotProductExpectation(f,f,[],varargin{:});
            v = v - m.^2;
        end
        
        function s = std(f,varargin)
            % s = std(f,X)
            % Computes the standard deviation of the random variable f(X).
            % If X is not provided, uses
            % random variables associated with underlying basis.
            % f: FunctionalBasisArray
            % X: RandomVector (optional)
            % s: 1-by-si double, where si is the size of f
            
            s = sqrt(variance(f,varargin{:}));
        end
        
        function c = dotProductExpectation(f,g,dims,varargin)
            % c = dotProductExpectation(f,g)
            % Computes the expectation of f(X)g(X) where X is the random
            % vector associated with underlying basis.
            %
            % c = dotProductExpectation(f,g,[],X)
            % Computes the expectation of f(X)g(X) for the provided
            % RandomVector X
            %
            % c = dotProductExpectation(f,g,Xdims,X)
            % For vector-valued functions of X, specify
            % the dimensions of f and g corresponding to RandomVector X
            %
            % f,g: FunctionalBasisArray
            % Xdims: D-by-1 or 1-by-D double (optional)
            % X: RandomVector (optional)
            % c: 1-by-s double, where s is the size of f
            
            if nargin == 2 || isempty(dims)
                dims = 1:length(f.basis);
            end
            if ~(f.basis == g.basis) || ~f.basis.isOrthonormal
                error('Not implemented');
            end
            
            c = dot(f,g,1);
        end
        
        function n = normExpectation(f,varargin)
            % n = normExpectation(f,X)
            % Computes the L^2 norm of f(X).
            % If X is not provided, uses
            % random variables associated with underlying basis of f.
            % f: FunctionalBasisArray
            % rv: RandomVector (optional)
            % n: 1-by-s double, where s is the size of f
            
            n = sqrt(dotProductExpectation(f,f,[],varargin{:}));
        end
        
        function f = conditionalExpectation(f,dims,varargin)
            % y = conditionalExpectation(f,dims,XdimsC)
            % Computes the conditional expectation of f with respect to
            % the random variables dims (a subset of 1:d). The expectation
            % with respect to other variables (in the complementary set of
            % dims) is taken with respect to the probability measure given by RandomVector XdimsC
            % if provided, or with respect to the probability measure
            % associated with the corresponding bases of f.
            % f: FunctionalBasisArray
            % dims: D-by-1 or 1-by-D double
            %   or 1-by-d logical
            % XdimsC: RandomVector containing (d-D) RandomVariable (optional)
            
            h = conditionalExpectation(f.basis,dims,varargin{:});
            f.data = h.data * f.data(:,:);
            f.data = reshape(f.data,[size(f.data,1),f.sz]);
            f.basis = h.basis;
        end
        
        function v = varianceConditionalExpectation(f,alpha)
            % v = varianceConditionalExpectation(f,alpha)
            % Computes the variance of the conditional expectation of f in dimensions in alpha
            % f: FunctionalBasisArray
            % alpha: n-by-D double, where D is equal to the number of random variables or n-by-d logical
            % v: n-by-1 double
            
            m = expectation(f);
            v = zeros(size(alpha,1),prod(f.sz));
            for i = 1:size(alpha,1)
                u = alpha(i,:);
                if isa(u,'logical')
                    u = find(u);
                end
                if isempty(u)
                    v(i,:)=0;
                else
                    mi = conditionalExpectation(f,u);
                    vi = dotProductExpectation(mi,mi) - m.^2;
                    v(i,:) = vi(:);
                end
            end
            v = reshape(v,[size(alpha,1),f.sz]);
        end
        
        function y = eval(f,x,varargin)
            % y = eval(f,x)
            % Computes evaluations of f at points x
            % f: FunctionalBasisArray
            % x: array of size N-by-d or cell of length d
            H = eval(f.basis,x,varargin{:});
            y = evalWithBasesEvals(f,H,varargin{:});
        end
        
        function y = evalWithBasesEvals(f,H)
            % y = evalWithBasesEvals(f,H)
            
            y = H*reshape(f.data,cardinal(f.basis),prod(f.sz));
            y = reshape(y,[size(H,1),f.sz]);
        end
        
        function y = evalDerivative(f,n,x)
            % y = evalDerivative(f,n,x)
            % Computes the n-derivative of f at points x in R^d, with n a multi-index of size d
            % f: FunctionalBasisArray
            % n: 1-by-d array of integers
            % x: N-by-d array of doubles
            % y: N-by-1 array of doubles
            
            if ~ismethod(f.basis,'evalDerivative')
                error('Not implemented for the basis object.')
            end
            H = evalDerivative(f.basis,n,x);
            y = evalWithBasesEvals(f,H);
        end
        
        function df = derivative(f,n)
            % df = derivative(f,n)
            % Computes the n-derivative of f
            % f: FunctionalBasisArray
            % n: 1-by-d array of integers
            % df: FunctionalBasisArray
            
            df = f;
            df.basis = derivative(f.basis,n);
        end
        
        function [y,x] = random(f,varargin)
            % [y,x] = random(f,n,rv)
            % Computes evaluations of f at an array of points of size n,
            % drawn randomly according to the RandomVector rv if provided,
            % or to the standard RandomVector associated with each
            % polynomial if not
            % f: FunctionalBasisArray
            % n: i-by-j double
            % rv: RandomVector or RandomVariable (optional)
            % y: i-by-j double
            % x: d-by-1 cell containing i-by-j doubles, where d is the
            % dimension of the basis
            
            [fx,x] = random(f.basis,varargin{:});
            y = fx*reshape(f.data,cardinal(f.basis),prod(f.sz));
            y = reshape(y,[size(fx,1), f.sz]);
        end
        
        function rv = getRandomVector(f)
            % rv = getRandomVector(f)
            % Gets the random vector rv associated with the basis functions of f
            % f: FunctionalBasisArray
            % rv: RandomVector
            
            rv = getRandomVector(f.basis);
        end
        
        function s = storage(f)
            s = numel(f.data);            
        end
        
        function s = sparseStorage(f)
            s = nnz(f.data);            
        end
        
        
        function g = projection(f,basis,indices)
            % g = projection(f,basis,indices)
            % Projection of f on a functional basis using multi-indices
            % indices if provided, or the multi-indices associated with
            % the functional basis if not
            % f: FunctionalBasisArray
            % basis: FunctionalBasis (FullTensorProductFunctionalBasis or SparseTensorProductFunctionalBasis)
            % indices: MultiIndices (optional if basis is an instance of
            % class SparseTensorProductFunctionalBasis, indices = basis.indices by default)
            % g: FunctionalBasisArray
            
            if nargin<3 || isempty(indices)
                if isa(basis,'SparseTensorProductFunctionalBasis')
                    indices = basis.indices;
                else
                    error('Must specify a MultiIndices')
                end
            end
            
            if ndims(f.basis)==ndims(basis) && cardinal(f.basis)<=cardinal(basis)
                d = sparse(cardinal(basis),prod(f.sz));
                [~,ia,ib] = intersectIndices(f.basis.indices,indices);
                d(ib,:) = f.data(ia,:);
                if ndims(f.data)~=ndims(d)
                    d = reshape(full(d),[cardinal(basis),f.sz]);
                end
                if isa(basis,'FullTensorProductFunctionalBasis')
                    H = FullTensorProductFunctionalBasis(basis.bases);
                elseif isa(basis,'SparseTensorProductFunctionalBasis')
                    H = SparseTensorProductFunctionalBasis(basis.bases,indices);
                else
                    error('Method not implemented.')
                end
                g = FunctionalBasisArray(d,H,f.sz);
            else
                error('Method not implemented.')
            end
        end
        
        function plotMultiIndices(f,varargin)
            % plotMultiIndices(f)
            % Plots the multi-index set associated to f
            % f: FunctionalBasisArray
            
            plotMultiIndices(f.basis,varargin{:});
        end
        
        function s = subFunctionalBasis(p)
            % s = subFunctionalBasis(p)
            % Converts a FunctionalBasisArray p into a SubFunctionalbasis s
            % p: FunctionalBasisArray
            % s: SubFunctionalbasis
            
            s = SubFunctionalBasis(p.basis,p.data);
        end
        
        function s = getCoefficients(f)
            s = reshape(f.data,[cardinal(f.basis),f.sz]);
        end
    end
end