% Class RandomVector

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

classdef RandomVector < ProbabilityMeasure
    
    properties
        randomVariables
        copula = IndependentCopula()
    end
    
    methods
        function rv = RandomVector(varargin)
            % rv = RandomVector(r,d) or rv =
            % RandomVector({r1,r2,...}) or rv = RandomVector(r1,r2,...)
            % Constructor for the RandomVector class, takes as inputs a cell of RandomVariable, several RandomVariable or a
            % RandomVariable and the dimension of the RandomVector
            % The random variables are assumed independent and non
            % conditional
            % varargin: r, r1, r2, ...: RandomVariable, d: 1-by-1 double
            % rv: RandomVector
            
            if nargin==1 && isa(varargin{1}, 'RandomVector')
                rv = varargin{1};
            elseif nargin==2 && isa(varargin{1}, 'RandomVector') && isa(varargin{2}, 'Copula')
                rv = varargin{1};
                rv.copula = varargin{2};
            elseif nargin == 1 && isa(varargin{1}, 'RandomVariable')
                rv.randomVariables{1} = varargin{1};
            elseif nargin == 1 && isa(varargin{1},'cell')
                rv.randomVariables = varargin{1};
            elseif nargin == 2 && isa(varargin{1},'RandomVariable') && isa(varargin{2}, 'double')
                rv.randomVariables = cell(1,varargin{2});
                for i = 1:varargin{2}
                    rv.randomVariables{i} = varargin{1};
                end
            else
                rv.randomVariables = cell(1,nargin);
                for i = 1:nargin
                    rv.randomVariables{i} = varargin{i};
                end
            end
        end
        
        function n = ndims(rv)
            n = sum(cellfun(@ndims,rv.randomVariables));
        end
        
        function ok = eq(r1,r2)
            % ok = eq(r1,r2)
            % Checks if two RandomVector r1 and r2 are equal
            % r1: RandomVector
            % r2: RandomVector
            % ok: boolean
            
            if ~(isa(r1,'RandomVector') && isa(r2,'RandomVector'))
                ok = 0;
            elseif length(r1.randomVariables) ~= length(r2.randomVariables)
                ok = 0;
            else
                ok = 1;
                for i = 1:length(r1.randomVariables)
                    ok = ok & (r1.randomVariables{i} == r2.randomVariables{i});
                end
            end
        end
        
        function Y = marginal(X,ind)
            if ~isa(X.copula,'IndependentCopula')
                error('only works for independent copula');
            end
            Y = RandomVector(X.randomVariables(ind));
        end
        
        function ok = ne(r1,r2)
            % ok = ne(r1,r2)
            % Checks if two RandomVector r1 and r2 are not equal
            % r1: RandomVector
            % r2: RandomVector
            % ok: boolean
            
            ok = ~eq(r1,r2);
        end
        
        function rvstd = getStandardRandomVector(rv)
            % rvstd = getStandardRandomVector(rv)
            % Returns the standard random vector of rv
            % rv: RandomVector
            % rvstd: RandomVector
            
            Xstd = cell(1,numel(rv));
            for k=1:numel(rv)
                Xstd{k} = getStandardRandomVariable(rv.randomVariables{k});
            end
            rvstd = RandomVector(Xstd);
        end
        
        function G = isoProbabilisticGrid(rv,n)
            % G = isoProbabilisticGrid(rv,n)
            % Generates a grid of (n(1)-1)x...x(n(d)-1) points (x_{i_1}^1,...x_{i_d}^d) such that the N = (n(1))x...x(n(d)) sets [x_{i_1-1}^1,x_{i_1}^1]x ... x[x_{i_d-1}^1,x_{i_d}^1] have the same probability p = 1/N.
            % rv: RandomVector of ndims d, with IndependentCopula
            % n = 1-by-d integer
            % G: FullTensorGrid
            %
            % G = isoProbabilisticGrid(rv,p)
            % Specify the probability of each set
            % p : 1-by-1 double (<1)
            
            if ~isa(rv.copula,'IndependentCopula')
                error('The method only works for independent copulas.')
            end
            
            d = numel(rv);
            
            if numel(n)==1 && n<1
                p = n;
                n = ceil(p^(-1/d));
            end
            
            if numel(n)~=d
                n = repmat(n,1,d);
            end
            
            b = cell(1,d);
            for k =1:d
                b{k} = isoProbabilisticGrid(rv.randomVariables{k},n(k));
            end
            G = FullTensorGrid(b);
        end
        
        function A = lhsRandom(rv,n)
            % A = lhsRandom(X,n,p)
            % Latin Hypercube Sampling of the RandomVector X of n points
            % X: RandomVector
            % n: integer
            % A: 1-by-numel(rv) cell containing n-by-1 doubles
            
            A = lhsdesign(n,numel(rv));
            A = transfer(RandomVector(UniformRandomVariable(0,1),numel(rv)),rv,A);
            A = mat2cell(A,n,ones(1,numel(rv)));
        end
        
        function m = mean(rv)
            % m = mean(X)
            % Computes the means of the RandomVector X
            % X: RandomVector
            % m: 1-by-1 double
            
            m = cell(1,numel(rv));
            for i = 1:numel(rv)
                m{i} = mean(rv.randomVariables{i});
            end
        end
        
        function n = numel(rv)
            % n =  numel(rv)
            % Gets the number of RandomVariable in the RandomVector
            % rv: RandomVector
            % n: integer
            
            n = length(rv.randomVariables);
        end
        
        function p = pdf(rv,x)
            % px = pdf(rv,x)
            % Computes the probability density function of each RandomVariable in rv at points x, x must have ndims(rv) columns
            % rv: RandomVector
            % x: 1-by-n or n-by-1 double
            % p: 1-by-n or n-by-1 double
            
            p = zeros(size(x));
            for i = 1:numel(rv)
                p(:,i) = pdf(rv.randomVariables{i},x(:,i));
            end
            p = prod(p,2);
            
            if ~isa(rv.copula,'IndependentCopula')
                u = zeros(size(x));
                for i = 1:numel(rv)
                    u(:,i) = cdf(rv.randomVariables{i},x(:,i));
                end
                p = p.*pdf(rv.copula,u);
            end
        end
        
        function u = cdf(rv,x)
            % Fx = cdf(rv,x)
            % Computes the cumulative distribution function at points x, x must have ndims(rv) columns
            % rv: RandomVector
            % x: 1-by-n or n-by-1 double
            % Fx: 1-by-n or n-by-1 double
            
            u = zeros(size(x));
            for i = 1:numel(rv)
                u(:,i) = cdf(rv.randomVariables{i},x(:,i));
            end
            u = cdf(rv.copula,u);
        end
        
        function r = random(rv,n,varargin)
            % r = random(X,n)
            % Generates n random numbers according to the distributions of RandomVariable in the RandomVector rv
            % X: RandomVector
            % n: integer
            % r: n-by-numel(rv) double
            
            if nargin > 2
                warning('random should have only two input arguments.')
            end
            
            if ~isa(rv.copula,'IndependentCopula')
                error('only works for independent copula')
            end
            
            if nargin==1
                n=1;
            end
            
            if numel(n)>1
                error('n must be an integer.')
            end
            
            dims =  cellfun(@ndims, rv.randomVariables);
            r = zeros(n, sum(dims));
            for i = 1:length(rv.randomVariables)
                rep = sum(dims(1:i-1))+(1:dims(i));
                r(:,rep) = random(rv.randomVariables{i},n);
            end
        end
        
        function s = std(rv)
            % s = std(X)
            % Computes the standard deviation of the RandomVariable in the RandomVector rv
            % X: RandomVector
            % s: 1-by-numel(rv) cell
            
            s = cell(1,numel(rv));
            for i = 1:numel(rv)
                s{i} = std(rv.randomVariables{i});
            end
        end
        
        function y = transfer(X,Y,x)
            % y = transfer(X,Y,x)
            % Transfers from the RandomVector X to the RandomVector Y, at points x
            % X: RandomVariable
            % Y: RandomVariable
            % x: 1-by-numel(X) or numel(X)-by-1
            % y: 1-by-numel(X) or numel(X)-by-1
            
            if isa(x,'cell')
                x = [x{:}];
            end
            
            if isa(Y,'RandomVariable')
                Y = RandomVector(Y,numel(X));
            end
            
            if numel(X) ~= numel(Y)
                error('The two RandomVector must have the same dimension.');
            end
            
            y = zeros(size(x,1),numel(Y));
            
            for i = 1:numel(X)
                y(:,i) = transfer(X.randomVariables{i},Y.randomVariables{i},x(:,i));
            end
        end
        
        function v = variance(rv)
            % v = variance(X)
            % Computes the variance of the RandomVariable in the RandomVector rv
            % X: RandomVector
            % s: 1-by-numel(rv) cell
            
            v = cell(1,numel(rv));
            for i = 1:numel(rv)
                v{i} = variance(rv.randomVariables{i});
            end
        end

        function m = moment(rv,I)
            % function m = moment(X,I)
            % Returns the moments m_i(X) = E(X^i) of X with i listed in I
            % X: RandomVector
            % I: k-by-d array of integers, with d the dimension of the
            % random vector
            % m : k-by-1 vector with m(i) = m_{I(i,:)}(X) 

            if ~isa(rv.copula,'IndependentCopula')
                error('Not implemented for non IndependentCopula.')
            end

            m = ones(size(I,1),1);
            for k=1:numel(rv)
                mk = moment(rv.randomVariables{k},I(:,k));
                m = m.*mk(:);
            end

        end
        
        function s = support(rv)
            % s = support(rv)
            % s: cell array containing the supports of random variables
            
            if ~isa(rv.copula,'IndependentCopula')
                error('Not implemented for non IndependentCopula.')
            end
            s = cell(1,numel(rv));
            for k = 1:numel(rv)
                s{k} = support(rv.randomVariables{k});
            end
        end
        
        function s = truncatedSupport(rv)
            % s = truncatedSupport(rv)
            % s: cell array containing the truncated supports of random variables
            
            if ~isa(rv.copula,'IndependentCopula')
                error('Not implemented for non IndenpendentCopula.')
            end
            s=cell(1,numel(rv));
            for k=1:numel(rv)
                s{k} = truncatedSupport(rv.randomVariables{k});
            end
        end
        
        function p = orthonormalPolynomials(rv,n)
            p = cell(1,numel(rv));
            if nargin==2 && numel(n)==1
                n = repmat(n,1,numel(rv));
            end
            for k=1:numel(rv)
                if nargin==1
                    p{k} = orthonormalPolynomials(rv.randomVariables{k});
                else
                    p{k} = orthonormalPolynomials(rv.randomVariables{k},n(k));
                end
            end
        end
        
        function X = permute(X,p)
            % X = permute(X,p)
            % X: randomVector
            % p: array containing a permutation of 1:numel(X)
            
            X.randomVariables = X.randomVariables(p);
        end
        
        function sz = size(X)
            sz = [numel(X),1];
        end
    end
    
    methods (Static, Hidden)
        function initstate()
            % initstate()
            % Changes the state of the pseudo-random numbers generators
            
            pause(eps)
            rng(sum(100*clock));
        end
    end
end