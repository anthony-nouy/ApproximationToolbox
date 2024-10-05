% Class PolynomialFunctionalBasis

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

classdef PolynomialFunctionalBasis < FunctionalBasis
    
    properties
        basis
        indices
    end
    
    methods
        function p = PolynomialFunctionalBasis(basis,indices)
            % p = PolynomialFunctionalBasis(basis,indices)
            % basis : UnivariatePolynomials
            % indices : array containing the selected polynomials indices
            % p: PolynomialFunctionalBasis
            
            p.basis = basis;
            p.indices = indices;
            p.measure = basis.measure;
            
            if isa(basis,'OrthonormalPolynomials') || isa(basis,'ShiftedOrthonormalPolynomials')
                p.isOrthonormal = true;
            end
        end
        
        function ok = eq(p,q)
            % ok = eq(p,q)
            % Checks if the two objects p and q are identical
            % p: PolynomialFunctionalBasis
            % q: PolynomialFunctionalBasis
            % ok: boolean
            
            if ~isa(q,'PolynomialFunctionalBasis')
                ok = 0;
            else
                try
                    ok = all(p.basis == q.basis) & all(p.indices == q.indices);
                catch
                    ok = 0;
                end
            end
        end
        
        function n = cardinal(p)
            n = numel(p.indices);
        end
        
        
        function px = eval(p,x)
            % px = eval(p,x)
            % Computes evaluations of the polynomials of p.basis of degrees
            % in p.indices at points x
            % p: PolynomialFunctionalBasis
            % x: n-by-1 or 1-by-n double
            % px: n-by-d double, where d = size(p.indices,1)
            
            px = polyval(p.basis,p.indices,x);
        end
        
        function h = kron(p,q)
            % See FunctionalBasis/kron
            
            m = max(p.indices)+max(q.indices);
            b = PolynomialFunctionalBasis(p.basis,0:m);
            b.measure = p.measure;
            fun = @(x) reshape(repmat(eval(p,x),[1,1,cardinal(q)]).*...
                repmat(permute(eval(q,x),[1,3,2]),[1,cardinal(p),1]),...
                [size(x,1) , cardinal(p)*cardinal(q)]);
            rv = b.measure;
            Q = gaussIntegrationRule(rv,2*m);
            a = b.projection(fun,Q);
            h = SubFunctionalBasis(b,a.data);
        end
        
        function f = derivative(h,k,rv)
            % Computes the k-th order derivative of the functions of the basis h projected on h
            % h: PolynomialFunctionalBasis
            % k: integer
            % rv: Measure, optional if an h.basis is an
            % OrthonormalPolynomials
            % f: SubFunctionalBasis
            
            if nargin < 3
                if isempty(h.measure)
                    error('Must specify a Measure')
                else
                    rv = h.measure;
                end
            end
            
            m = ceil(max(h.indices) - (k-1)/2);
            I = gaussIntegrationRule(rv,m);
            f = h.projection(@(x) dnPolyval(h.basis,k,h.indices,x),I);
            
            f = subFunctionalBasis(f);
        end
        
        function g = gradient(h,rv)
            % Computes the first order derivative of the functions of the basis h projected on h
            % h: PolynomialFunctionalBasis
            % rv: Measure, optional if an h.basis is an
            % OrthonormalPolynomials
            % g: SubFunctionalBasis
            
            g = derivative(h,1,rv);
        end
        
        function H = hessian(h,rv)
            % Computes the second order derivative of the functions of the basis h projected on h
            % h: PolynomialFunctionalBasis
            % rv: Measure, optional if an h.basis is an
            % OrthonormalPolynomials
            % H: SubFunctionalBasis
            
            H = derivative(h,2,rv);
        end
        
        function y = evalDerivative(h,k,x)
            % y = evalDerivative(h,k,x)
            % Evaluates the k-th order derivative of the functions of the
            % basis h at points x
            % h: PolynomialFunctionalBasis
            % k: integer
            % x: N-by-1 or 1-by-N array of doubles
            % y: N-by-numel(h) array of doubles
            
            y = dnPolyval(h.basis,k,h.indices,x);
        end
        
        function M = gramMatrix(f,mu)
            % M = gramMatrix(f,mu)
            % Computes the Gram matrix of the basis f. The Gram
            % matrix is the matrix of the dot products between each
            % possible couple of basis functions of f. The dot product in
            % the dimension i is computed according to Measure mu
            % if provided, or according to the Measure in f if not.
            % f: PolynomialFunctionalBasis
            % mu: Measure (optional)
            % M: P-by-P double, where P is the number of multi-indices
            
            if nargin < 2
                if isempty(f.measure)
                    mu = f.basis.measure;
                else
                    mu = f.measure;
                end
            end      

            if (mu == f.basis.measure) && f.basis.isOrthonormal
                M = speye(f.cardinal());
                return
            end

            if isa(mu,'DiscreteMeasure') || isa(mu,'DiscreteRandomVariable')
                M = gramMatrix@FunctionalBasis(f,mu);
                return
            end

            if ismethod(f.basis,'moment')
                ind = f.indices;
                list = [repmat(ind',length(ind),1), reshape(repmat(ind,length(ind),1),numel(ind)^2,1)];
                M = reshape(moment(f.basis,list,mu),[length(ind) length(ind)]);
            else
                error('Not implemented');
            end
        end
        
        function H = one(p)
            [c,I] = one(p.basis);
            [ok,rep] = ismember(I,p.indices);
            if ~all(ok)
                error('constant 1 can not be represented')
            end
            H = zeros(numel(p.indices),1);
            H(rep) = c;
        end
        
        function m = mean(p,varargin)
            % m = mean(f,rv)
            % Gives the expectation of basis functions, according to the
            % randomVariable property of p if rv is not provided, or to the
            % RandomVariable rv if it is provided
            % rv: RandomVariable (optional)
            % m: n-by-1 double, where n is the length of the indices
            % property of p
            
            m = mean(p.basis,p.indices,varargin{:});
        end
        
        function rv = randomVariable(p)
            % rv = randomVariable(p)
            % Returns the RandomVariable associated to p if it exists
            % p: PolynomialFunctionalBasis
            % rv: RandomVariable or empty array
            
            if isprop(p.basis,'randomVariable')
                rv = p.basis.randomVariable;
            else
                warning('Empty random variable.')
                rv = [];
            end
            
        end
        
        function nu = optimalSamplingMeasure(f)
            % nu = optimalSamplingMeasure(f)
            % nu is the measure with randon derivative w.r.t measure f.measure
            % equal to the christoffel function of f
            % f: PolynomialFunctionalBasis
            % mu: ProbabilityMeasureWithRadonDerivative
            
            w = christoffel(f)/cardinal(f);
            nu = ProbabilityMeasureWithRadonDerivative(f.measure,w);
        end
        
        
        function ch = christoffel(f,x)
            % ch = christoffel(f)
            % Returns the inverse Christoffel function associated with a functional basis
            % f: PolynomialFunctionalBasis
            % ch: FunctionalBasisArray`
            %
            % ch = christoffel(f,x)
            % Evaluates the inverse Christoffel function at x
            % f: FunctionalBasis
            % x: n-by-1 array
            % ch: n-by-1 array
            
            if nargin==1
                m = 2*max(f.indices);
                b = PolynomialFunctionalBasis(f.basis,0:m);
                
                if f.isOrthonormal
                    fun = @(x) sum(eval(f,x).^2,2);
                else
                    G = gramMatrix(f);
                    fun = @(x) sum((f.eval(x) / G) .* f.eval(x),2);
                end
                Q = gaussIntegrationRule(f.measure,m+1);
                ch = b.projection(fun,Q);
            else
                ch = christoffel@FunctionalBasis(f,x);
            end
        end
        
        function s = domain(p)
            s = domain(p.basis);
        end
        
        function n = ndims(p)
            n = ndims(p.basis);
        end
    end
end