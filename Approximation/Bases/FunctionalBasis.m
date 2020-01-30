% Class FunctionalBasis

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

classdef FunctionalBasis
    
    properties
        measure % Measure
        isOrthonormal = false % L2-orthonormality with respect to the given measure
    end
    
    methods
        
        function n = numel(f)
            warning('numel is replaced by cardinal and will be removed in a future release.')
            n = cardinal(f);
        end
        
        function [fx,x] = random(f,n,measure)
            % [fx,x] = random(f,n,measure)
            % Evaluates the function f at n points x drawn randomly
            % according to the ProbabilityMeasure in measure if provided,
            % or in f.measure.
            % h: FunctionalBasis
            % n: integer
            % measure: ProbabilityMeasure (optional)
            % fx: n-by-cardinal(f) array of doubles
            % x: n-by-f.dim array of doubles
            
            if nargin == 1
                n = 1;
            end
            if nargin <= 2
                if isa(f.measure,'ProbabilityMeasure')
                    measure = f.measure;
                else
                    error('Must provide a ProbabilityMeasure')
                end
            end
            if nargin == 3 && ~isa(measure,'ProbabilityMeasure')
                error('Must provide a ProbabilityMeasure')
            end
            
            x = random(measure,n);
            fx = f.eval(x);
            fx = reshape(fx,[n, cardinal(f)]);
        end
        
        function s = storage(h)
            % s = storage(f)
            s=0;
        end
        
        function  P = adaptationPath(h)
            % P = adaptationPath(f)
            % Provides an adaptation path on a functional basis f
            % f: FunctionalBasis
            % P: n-by-m boolean array, where n is the dimension of the
            % functional basis f, and m is the number of elements in the adaptation path
            % column P(:,i) corresponds to a sparsity pattern
            
            n = cardinal(h);
            P = triu(true(n,n));
        end
        
        function [I,a] = interpolate(h,y,x)
            % [I,c] = interpolate(f,y,x)
            % Provides an interpolation on a functional basis f of a function
            % (or values of the function) y associated with a set of n
            % interpolation points x
            % f: FunctionalBasis
            % x: array of size n-by-d
            % y: function or values of the function at x
            % I: FunctionalBasisArray
            % c: set of coefficients of I on the FunctionalBasis
            %
            % For the simultaneous interpolation of k functions
            % y can be an array of size n-by-k
            % or a function such that y(x) is of size n-by-k
            % Then c is an array of size numel(h)-by-k
            
            if nargin==2
                x = interpolationPoints(h);
            end
            if ~isa(y,'double')
                y = y(x);
            end
            
            hx = eval(h,x);
            a = hx\y;
            I = FunctionalBasisArray(a,h,size(y,2));
            I.measure = h.measure;
        end
        
        function m = mean(~,varargin)
            % MEAN - Computes the mean of the basis h
            %
            % m = MEAN(h)
            % h: FunctionalBasis
            % m: n-by-1 double containing the means
            
            error('No generic implementation of the method mean.');
        end
        
        function m = expectation(~,varargin)
            try
                m = mean(h,varargin{:});
            catch
                error('No generic implementation of the method mean.');
            end
        end
        
        function m = conditionalExpectation(~,varargin)
            % y = conditionalExpectation(f,dims,XdimsC)
            % Computes the conditional expectation of f with respect to
            % the random variables dims (a subset of 1:d). The expectation
            % with respect to other variables (in the complementary set of
            % dims) is taken with respect the probability measure given by RandomVector XdimsC
            % if provided, or with respect the probability measure
            % associated with the corresponding bases of f.
            % f: FunctionalBasis
            % dims: 1-by-D double or 1-by-d logical
            % XdimsC: RandomVector containing (d-D) RandomVariable (optional)
            % m: FunctionalBasisArray
            
            error('No generic implementation of the method conditionalExpectation');
        end
        
        function h = kron(f,g)
            % h = kron(f,g)
            % For functional basis f_i, i=1...n, and g_j, j=1...m, returns a functional basis
            % h_k, k=1...nm
            % f: FunctionalBasis
            % g: FunctionalBasis
            
            error('Method not implemented.');
        end
        
        function u = projection(h,fun,G)
            % u = projection(h,fun,G)
            % Computes the projection of the function fun onto the
            % functional basis h using the integration rule G
            % h: FunctionalBasis
            % fun: function_handle or Function
            % G: IntegrationRule
            % u: FunctionalBasisArray
            %
            % For the projection of a function defined on R^d with values
            % in R^k, fun must be a function such that f(x) is of size n-by-k
            % when x is of size n-by-d
            
            if isa(fun,'function_handle')
                fun = fcnchk(fun);
            end
            
            N = length(G.weights);
            
            A = eval(h,G.points);
            W = spdiags(G.weights(:),0,N,N);
            
            y = fun(G.points);
            if h.isOrthonormal
                u = A'*W*y;
            else
                u = (A'*W*A) \ (A'*W*y);
            end
            u = FunctionalBasisArray(u,h,size(u,2));
        end
        
        
        function p = interpolationPoints(h,varargin)
            p = magicPoints(h,varargin{:});
        end
        
        function [points,I,output] = magicPoints(h,x,J)
            % [points,I,output] = magicPoints(f,x)
            % Provides the magic points associated with a functional basis f selected in
            % a given set of points x
            % Use magicIndices(F,numel(f)) on the matrix F of evaluations of f at
            % points x
            % f: FunctionalBasis
            % x: array of size N-by-d (N>=numel(f)) containing N points
            % I: location of points in x
            %
            % [points,I] = magicPoints(f,x,J)
            % Use magicIndices(F(:,J),numel(f),'left')
            
            if nargin<2 || isempty(x)
                if isa(h.measure,'DiscreteMeasure') || isa(h.measure,'DiscreteRandomVariables')
                    x = h.measure.values;
                else
                    x = random(h.measure,cardinal(h)*100);
                end
            end
            assert(size(x,1)>=cardinal(h),'the number of points must be higher than the number of basis functions')
            
            F = eval(h,x);
            if nargin==3
                I = magicIndices(F(:,J),cardinal(h),'left');
            else
                I = magicIndices(F);
            end
            points = x(I,:);
            
            % Estimation of Lebesgue constant
            hx = eval(h,points);
            A = F/hx;
            
            output.lebesgueConstant = max(sum(abs(A),2));
            %fprintf('Lebesgue constant = %3d\n',output.lebesgueConstant);
        end
        
        function D = domain(f)
            % D = domain(f)
            % f: FunctionalBasis
            % D: domain of the set of basis functions, support of the associated measure
            
            D = support(f.measure);
        end
        
        function ch = christoffel(f,x)
            % ch = christoffel(f)
            % or
            % ch = christoffel(f,x)
            
            if nargin==1
                if f.isOrthonormal
                    ch = @(x) sum(abs(f.eval(x)).^2,2);
                else
                    G = gramMatrix(f);
                    ch = @(x) sum((f.eval(x) / G) .* f.eval(x),2);
                end
            else              
                if f.isOrthonormal
                    ch = sum(abs(f.eval(x)).^2,2);
                else
                    G = gramMatrix(f);
                    fEval = f.eval(x);
                    ch = sum((fEval / G) .* fEval,2);
                end
            end
        end
        
        function f = orthonormalize(f)
            % f = orthonormalize(f)
            
            G = gramMatrix(f);
            if normest(G-eye(size(G,1)))>1e-15
                A = inv(chol(G,'lower'));
                f = SubFunctionalBasis(f,A');
            end
            f.isOrthonormal = true;
        end
        
        function nu = optimalSamplingMeasure(f)
            % nu = optimalSamplingMeasure(f)
            % nu is the measure with radon derivative w.r.t measure f.measure
            % equal to the christoffel function of f
            % f: FunctionalBasis
            % mu: ProbabilityMeasureWithRadonDerivative
            
            w = UserDefinedFunction(@(x) christoffel(f,x)/cardinal(f),ndims(f.measure));
            nu = ProbabilityMeasureWithRadonDerivative(f.measure,w);
        end
        
        function varargout = plot(h,indices,n,varargin)
            % varargout = plot(h,indices,n,varargin)
            % h: FunctionalBasis
            % indices: array (list of basis functions to be plotted, all if indices=[])
            % n: integer (number of points for the plot)
            % varargin: additional arguments for command plot
            
            if ndims(h)>1
                error('not implemented')
            end
            
            s = truncatedSupport(h.measure);
            if nargin<3
                n=10000;
            end
            if length(n)==1
                x = linspace(s(1),s(2),n)';
            else
                x=n(:);
            end
            
            if nargin<2 || isempty(indices)
                hx = h.eval(x);
            else
                hx = h.eval(x,indices);
            end
            
            plot(x,hx,varargin{:})
        end
    end
    
    methods (Abstract)
        % N = cardinal(f)
        % Number of basis functions
        N = cardinal(f)
        
        % n = ndims(f)
        % Dimension n for f defined in R^n
        n = ndims(f)
        
        % y = eval(f,x)
        % Evaluation of the basis functions at points x
        y = eval(f,x,varargin)
    end
end