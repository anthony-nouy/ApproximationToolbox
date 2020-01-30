% Class MixtureProbabilityMeasure

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

classdef MixtureProbabilityMeasure < ProbabilityMeasure
    
    properties
        measures
        probabilities
    end
    
    methods
        function mu = MixtureProbabilityMeasure(measures,probabilities)
            % mu = MixtureProbabilityMeasure(measures,probabilities)
            % Creates a mixture of probability measures mu = \sum_{k=1}^N p_k mu_k
            %
            % measures: 1-by-N cell array containing the probability measures mu_k
            % probabilities: 1-by-N array containing the probabilities p_k  (by default, p_k=1/N)
            
            mu.measures = measures(:);
            N = length(mu.measures);
            if nargin==1
                probabilities = repmat(1/N,N,1);
            elseif length(probabilities)~=N
                error('Arguments must have the same length.')
            end
            mu.probabilities = probabilities(:);
            if sum(mu.probabilities)~=1
                error('Probabilities should sum to 1.')
            end
        end
        
        function ok = eq(mu,nu)
            if ~strcmp(class(mu),class(nu))
                ok=false;
            elseif length(mu.measures)~=length(nu.measures)
                ok=false;
            else
                ok = true;
                for k=1:length(mu.measures)
                    ok = ok & eq(mu.measures{k},nu.measures{k});
                    if ~ok
                        return
                    end
                end
            end
        end
        
        function n = ndims(mu)
            n = cellfun(@ndims,mu.measures);
            n = unique(n);
            if length(n)>1
                error('All measures should have the same dimension.')
            end
        end
        
        function s = support(mu)
            % s = support(mu)
            % Returns the interval (or box) containing the support of the measure
            % s: 2-by-dim array if dim>1
            %    1-by-2 array if dim=1
            
            dim = ndims(mu);
            s=zeros(2,dim);
            discrete = true;
            for k=1:length(mu.probabilities)
                if ~isa(mu.measures{k},'DiscreteRandomVariable')
                    discrete = false;
                end
                sk = support(mu.measures{k});
                if discrete
                    s = union(s,sk);
                else
                    s = [min(min(s),min(sk));max(max(s),max(sk))];
                end
            end
            if dim==1
                s=s(:);
            end
        end
        
        function s = truncatedSupport(mu)
            % s = truncatedSupport(mu)
            % s: 2-by-dim array if dim>1
            %    1-by-2 array if dim=1
            
            dim = ndims(mu);
            s=zeros(2,dim);
            for k=1:length(mu.probabilities)
                sk = truncatedSupport(mu.measures{k});
                s = [min(min(s),min(sk));max(max(s),max(sk))];
            end
            if dim==1
                s=s(:);
            end
        end
        
        function G = gaussIntegrationRule(mu,n)
            % G = gaussIntegrationRule(mu,n)
            % Creates a Gauss integration rule for the mixture of probability measures, using n points per measure
            
            weights = [];
            points = [];
            for k=1:length(mu.measures)
                G = gaussIntegrationRule(mu.measures{k},n);
                weights = [weights;G.weights*mu.probabilities(k)];
                points = [points;G.points];
            end
            G = IntegrationRule(points,weights);
        end
        
        function px = pdf(mu,x)
            % px = pdf(mu,x)
            % Computes the probability density function of the measure at x
            %
            % mu: MixtureProbabilityMeasure
            % x: n-by-dim array
            % px: n-by-1 array
            
            px = zeros(size(x,1),1);
            for k=1:length(mu.measures)
                px = px + mu.probabilities(k) * pdf(mu.measures{k},x);
            end
        end
        
        function c = cdf(mu,x)
            % c = cdf(mu,x)
            % Computes the cumulative density function of the measure at x
            %
            % mu: MixtureProbabilityMeasure
            % x: n-by-dim array
            % c: n-by-1 array
            
            c = zeros(size(x,1),1);
            for k=1:length(mu.measures)
                c = c + mu.probabilities(k) * cdf(mu.measures{k},x);
            end
        end
        
        function c = icdf(mu,p)
            % x = icdf(mu,p)
            % Computes the inverse cumulative density function of X at p (Quantile),
            % mu: MixtureProbabilityMeasure
            % p: array
            % c: array
            
            nu = UserDefinedProbabilityMeasure(ndims(X),'cdffun',@(x) cdf(mu,x),'supp',support(mu));
            c = icdf(nu,p);
        end
        
        function m = mean(mu)
            % m = mean(mu)
            
            m = mu.probabilities(1)*mean(mu.measures{1});
            for k=2:length(mu.measures)
                m = m + mu.probabilities(k)*mean(mu.measures{k});
            end
        end
        
        function v = var(mu)
            %  v = var(mu)
            
            v = mu.probabilities(1)*var(mu.measures{1});
            for k=2:length(mu.measures)
                v = v + mu.probabilities(k)*var(mu.measures{k});
            end
        end
        
        
        function r = random(mu,n,varargin)
            % r = random(mu,n)
            % Generates n random numbers according to the distribution of the mixture probability measure mu
            % mu: MixtureProbabilityMeasure
            % n: integer
            % r: double of size n-by-dim
            
            if nargin==1
                n=1;
            end
            
            D = DiscreteRandomVariable(1:length(mu.measures),mu.probabilities);
            I = random(D,n);
            dim = ndims(mu);
            r = zeros(n,dim);
            for k=1:length(mu.measures)
                rep = (I==k);
                nk = nnz(rep);
                if nk>0
                    r(I==k,:) = random(mu.measures{k},nk);
                end
            end
        end
    end
end