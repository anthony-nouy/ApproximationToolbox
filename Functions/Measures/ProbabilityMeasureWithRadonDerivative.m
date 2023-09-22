% Class ProbabilityMeasureWithRadonDerivative

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

classdef ProbabilityMeasureWithRadonDerivative < ProbabilityMeasure
    
    properties
        dim
        mu % ProbabilityMeasure
        h % RadonDerivatives
    end
    
    methods
        function nu = ProbabilityMeasureWithRadonDerivative(mu,h)
            % ProbabilityMeasureWithRadonDerivative - Constructor of the class ProbabilityMeasureWithRadonDerivative
            %
            % nu = ProbabilityMeasureWithRadonDerivative(mu,h)
            % creates a Probability Measure nu with Radon derivative h with
            % respect to mu
            % mu: ProbabilityMeasure
            % h: Function
            % nu: ProbabilityMeasureWithRadonDerivative
            
            if isa(mu,'RandomVector') && isa(mu.copula,'IndependentCopula')
                nu.dim = ndims(mu);
            end
            nu.mu = mu;
            nu.h = h;
        end
        
        function px = pdf(nu,x)
            % px = pdf(nu,x)
            % nu: ProbabilityMeasureWithRadonDerivative
            % x: double
            % px: double
            
            if size(x,2)~=ndims(nu)
                error('Input argument should be of size n-by-dim.')
            end
            if isa(nu.mu,'ProbabilityMeasure')
                px = nu.h(x).*pdf(nu.mu,x);
            elseif isa(nu.mu,'LebesgueMeasure')
                px = nu.h(x).*isIn(nu.mu.support(),x);
            end
            
        end
        
        function Fx = cdf(nu,x,varargin)
            % Fx = cdf(nu,x,tol)
            % Compute cdf by numerical integration with tolerance tol (1e-12 by default)
            
            if size(x,2)~=ndims(nu)
                error('input argument should be of size n-by-dim')
            end
            
            m = UserDefinedProbabilityMeasure(nu.dim,'pdf',@(t) nu.pdf(t),'supp',support(nu));
            Fx = cdf(m,x,varargin{:});
        end
        
        function q = icdf(nu,u,varargin)
            % Fx = icdf(nu,x,tol)
            % Compute the inverse cdf by numerical integration and root finding with tolerance tol (1e-12 by default)
            
            u = u(:);
            m = UserDefinedProbabilityMeasure(nu.dim,'pdf',@(t) nu.pdf(t),'supp',support(nu));
            q = icdf(m,u,varargin{:});
        end
        
        function o = eq(p,q)
            o = false;
        end
        
        function s = support(nu)
            s = support(nu.mu);
        end
        
        function s = truncatedSupport(nu)
            s = truncatedSupport(nu.mu);
        end
        
        function r = random(nu,n,type,varargin)
            % r = random(nu,n,type)
            % nu: ProbabilityMeasureWithRadonDerivative
            % n: integer
            % type: 'slice' % or discrete
            % r: n-by-1 double
            
            if nargin > 2
                warning('random should have only two input arguments.')
            end
            if nargin==1
                n=1;
            end
            if numel(n)>1
                error('n must be an integer.')
            end
            if nargin==3
                if type == 0
                    r = slicesample(random(nu.mu),n,'pdf', @(x)abs(pdf(nu,x)),varargin{:});
                else
                    suppmeasure = truncatedSupport(nu.mu);
                    muD = discretizeSupport(nu.mu,2000,suppmeasure);
                    muD.probabilities = muD.probabilities.*nu.h(muD.values);
                    r = random(muD,n);
                end
            else
                ok = false;
                while ~ok
                    try   
                        suppmeasure = truncatedSupport(nu.mu);
                        muD = discretizeSupport(nu.mu,2000,suppmeasure);
                        muD.probabilities = muD.probabilities.*nu.h(muD.values);
                        r = random(muD,n);
                        ok = true;
                    end
                end
            end
        end
        
        function marg = marginal(nu, ind)
            hmarg = conditionalExpectation(nu.h,ind,nu.mu);
            if isa(nu.mu,'ProductMeasure')
                if numel(ind) == 1
                   mumarg =  nu.mu.measures{ind};
                else  
                   mumarg = ProductMeasure(nu.mu.measures(ind));
                end
            elseif isa(nu.mu,'RandomVector') && isa(nu.mu.copula,'IndependentCopula')
                mumarg = RandomVector(nu.mu.randomVariables(ind));
            else
                error('Method not implemented : mumarg should be the a conditional probability measure.')
            end
            marg = ProbabilityMeasureWithRadonDerivative(mumarg,hmarg);
        end
        
        function n = ndims(nu)
            n = ndims(nu.mu);
        end
        
        function xs = randomSequential(p,n,type)
            % p is a ProbabilityMeasureWithRadonDerivative or a cell containing ProbabilityMeasureWithRadonDerivative
            
            nbdim = ndims(p);
            marginals = cell(1,nbdim);
            marg = cell(1,nbdim);
            xs = zeros(n,nbdim);
            marginals{nbdim} = p;
            
            for i = (nbdim-1):-1:1
                marginals{i} = marginal(marginals{i+1},1:i);
            end
            if nbdim > 1
                for i = 1:nbdim
                    marg{i} = marginal(p,i);
                end
            else 
                marg{nbdim} = p;
            end
            
            for k=1:n
                if type == 0
                    xs(k,1) = slicesample(0,1,'pdf', @(x)abs(pdf(marg{1},x)));
                else
                    muD = discretizeSupport(marg{1}.mu,2000,truncatedSupport(marg{1}.mu));
                    muD.probabilities = muD.probabilities.*abs(marg{1}.h(muD.values));
                    xs(k,1) = random(muD);
                end
                x_sample = [];
                for q = 2:nbdim
                    x_sample = [x_sample, xs(k,q-1)];
                    if type == 0
                        f = eval(marginals{q}.h, x_sample,1:(q-1));
                        mu_num = @(y)pdf(marginals{q}, [x_sample,y]);
                        mu_den = pdf(marginals{q-1},x_sample);
                        mu = @(y)abs(mu_num(y)/mu_den);
                        xs(k,q) = slicesample(0,1,'pdf',mu);
                    else
                        muD = discretizeSupport(marg{q}.mu,1000,truncatedSupport(marg{q}.mu));
                        [N, M] = size(muD.values);
                        mu_num = pdf(marginals{q},[repmat(x_sample, N, 1), muD.values]);
                        mu_den = pdf(marginals{q-1},x_sample);
                        mu_ratio = abs(mu_num./mu_den);
                        muD.probabilities = muD.probabilities.*mu_ratio;
                        xs(k,q) = random(muD);
                    end
                end
            end
        end
        
        function varargout = plot(X,varargin)
            % varargout = plot(X,h,varargin)
            % Plots the desired quantity, chosen between 'pdf', 'cdf' of 'icdf'.
            % X: ProbabilityMeasureWithRadonDerivative
            % h: char ('pdf' or 'cdf' or 'icdf')
            
            p = inputParser;
            addRequired(p,'type',@ischar);
            addParamValue(p,'npts',100,@isscalar);
            addParamValue(p,'bar',false,@islogical);
            addParamValue(p,'options',{});
            parse(p,varargin{:});
            
            s = truncatedSupport(X);
            switch p.Results.type
                case 'cdf'
                    x = linspace(s(1),s(2),p.Results.npts)';
                    P = cdf(X,x);
                case 'icdf'
                    x = linspace(0,1,p.Results.npts)';
                    P = icdf(X,x);
                case 'pdf'
                    x = linspace(s(1),s(2),p.Results.npts)';
                    P = pdf(X,x);
                otherwise
                    error('wrong argument type')
            end
            
            if p.Results.bar
                bar(x,P,p.Results.options{:});
            else
                plot(x,P,p.Results.options{:});
            end
            
            if nargout >= 1
                varargout{1} = P;
            end
            if nargout >= 2
                varargout{2} = x;
            end
        end
    end
end
