% Class RandomVariable

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

classdef RandomVariable < ProbabilityMeasure
    
    properties
        name
        moments
    end
    
    methods
        
        function  X = RandomVariable(name)
            % RandomVariable - Constructor of the class RandomVariable
            %
            % X = RandomVariable(name)
            % Random variable named name, the moments property remains empty as long as the moments have not been computed using the method moment
            % name: char
            % X: RandomVariable
            
            X.name = name;
            X.moments = [];
        end
        
        function n = ndims(~)
            n = 1;
        end
        
        function P = cdf(X,x)
            % P = cdf(X,x)
            % Computes the cumulative distribution function of the random variable X at point(s) x
            % X: RandomVariable
            % x: double
            % P: double
            
            param = getParameters(X);
            P = cdf(X.name,x,param{:});
            P(P==1) = 1-eps;
            P(P==0) = eps;
        end
        
        function P = icdf(X,x)
            % P = cdf(X,x)
            % Computes the inverse cumulative distribution function of the random variable X at point(s) x
            % X: RandomVariable
            % x: double
            % P: double
            
            param = getParameters(X);
            P = icdf(X.name,x,param{:});
        end
        
        function g = isoProbabilisticGrid(X,n)
            % g = isoProbabilisticGrid(X,n)
            % Returns a set of n+1 points (x_0,...,x_{n}) such that the n sets (x0,x_1),[x_1,x_2) ... [x_{n-1},x_{n})  have all the same probability p = 1/n (with x0 = min(X) and x_{n+1}=max(X))
            %
            % X: RandomVariable
            % n: integer (n>=1)
            % g: (n+1)-by-1 double
            %
            % g = isoProbabilisticGrid(X,p)
            % Specifies the probability p of each set
            % p :double (p<1)
            % (p should be of the form p=1/n with n an integer, if not p is replaced by inv(ceil(inv(p))))
            
            if n<1
                n = ceil(1/n);
            end
            if n==1
                g= [];
            elseif n==2
                g = icdf(X,.5);
            else
                g = icdf(X,linspace(1/n,1-1/n,n-1)');
            end
            g = [min(X);g;max(X)];
        end
        
        function Xn = discretize(X,n)
            % Xn = discretize(X,n)
            % Returns a discrete random variable Xn taking n possible values x1,...xn, these values being the quantiles of X of probability 1/(2n) + i/n, i=0...n-1 and such that P(Xn \le xn) = 1/n
            % X: RandomVariable
            % Xn: DiscreteRandomVariable
            
            u = linspace(1/2/n,1-1/2/n,n)';
            x = icdf(X,u);
            Xn = DiscreteRandomVariable(x);
        end
        
        
        function ok = eq(r1,r2)
            % ok = eq(r1,r2)
            % Checks if two random variables r1 and r2 are equal
            % r1: RandomVariable
            % r2: RandomVariable
            % ok: boolean
            
            if ~(isa(r1,'RandomVariable') && isa(r2,'RandomVariable') )
                ok = 0;
            elseif ~strcmp(class(r1),class(r2))
                ok = 0;
            else
                ok = 1;
                param1 = getParameters(r1);
                param2 = getParameters(r2);
                for i=1:size(param1,2)
                    ok = ok & (param1{i} == param2{i});
                end
                ok = all(ok);
            end
        end
        
        function ok = ne(r1,r2)
            % ok = ne(r1,r2)
            % Checks if two random variables r1 and r2 are not equal
            % r1: RandomVariable
            % r2: RandomVariable
            % ok: boolean
            
            ok = not(eq(r1,r2));
        end
        
        function G = gaussIntegrationRule(X,n)
            % G = gaussIntegrationRule(X,n)
            % Returns the n-points gauss integration rule associated with the measure of X, using Golub-Welsch algorithm
            % X: RandomVariable
            % n: integer
            % G: IntegrationRule
            
            p = orthonormalPolynomials(X,n+1);
            flag = false;
            if isa(p,'ShiftedOrthonormalPolynomials')
                p = p.p;
                flag = true;
            end
            c = p.recurrenceCoefficients;
            if size(c,2)<n
                c = p.recurrence(p.measure,n-1);
            else
                c = c(:,1:n);
            end
            
            % Jacobi matrix
            if n == 1
                J = diag(c(1,:));
            else
                J = diag(c(1,:)) + diag(sqrt(c(2,2:end)),-1) + diag(sqrt(c(2,2:end)),1);
            end
            
            % Quadrature points are the eigenvalues of the Jacobi matrix,
            % weights are deduced from the eigenvectors
            [V, D] = eig(full(J));
            [points,I] = sort(diag(D));
            points = reshape(points,n,1);
            V = V(:,I);
            
            weights = reshape((V(1,:).^2)./sqrt(sum(V.^2,1)),1,n);
            
            if flag
                points = transfer(p.measure,X,points);
            end
            
            G = IntegrationRule(points,weights);
        end
        
        function A = lhsRandom(X,n,p)
            % A = lhsRandom(X,n,p)
            % Latin Hypercube Sampling of the random variable X of n points in dimension p
            % X: RandomVariable
            % n: integer
            % p: integer
            % A: n-by-p double
            
            if nargin == 2
                p = 1;
            end
            
            A = lhsdesign(n,p);
            A(:) = transfer(UniformRandomVariable(0,1),X,A(:));
        end
        
        function L = likelihood(X,x)
            % L = likelihood(X,x)
            % Computes the log-likelihood of X on sample x
            % X: RandomVariable
            % x: double
            % L: 1-by-1 double
            
            P = pdf(X,x);
            L = sum(log(P+eps));
        end
        
        function m = max(X)
            % m = max(X)
            % Computes the maximum value that can take the inverse cumulative distribution function of the random variable X
            % X: RandomVariable
            % m: 1-by-1 double
            
            m = max(support(X));
        end
        
        function m = mean(X)
            % m = mean(X)
            % Computes the mean of the random variable X
            % X: RandomVariable
            % m: 1-by-1 double
            
            m = randomVariableStatistics(X);
        end
        
        function m = min(X)
            % m = min(X)
            % Computes the minimum value that can take the inverse cumulative distribution function of the random variable X
            % X: RandomVariable
            % m: 1-by-1 double
            
            m = min(support(X));
        end
        
        function [m,X] = moment(X,list)
            % [m,X] = moment(X,list)
            % Computes the moments of X of orders contained in list, defined as E(x^list(i))
            % If a second output argument is asked, then the computed moments are stored in the random variable X
            % X: RandomVariable
            % list: double
            % m: double
            
            if length(X.moments)-1 >= max(list)
                m = X.moments(list+1);
            else
                m=zeros(1,length(list));
                n = ceil((max(list)+1)/2);
                G = gaussIntegrationRule(X,n);
                
                for i = 1:length(list)
                    p = @(x) x.^list(i);
                    m(i) = integrate(G,p);
                end
                
                if nargout > 1
                    X.moments = m;
                end
            end
        end
        
        
        function n = numberOfParameters(X)
            % n = numberOfParameters(X)
            % Computes the number of parameters that admits the random variable X
            % X: RandomVariable
            % n: integer
            
            n = length(getParameters(X));
        end
        
        function px = pdf(X,x)
            % px = pdf(X,x)
            % Computes the probability density function of the random variable X at points x
            % X: RandomVariable
            % x: double
            % px: double
            
            param = getParameters(X);
            px = pdf(X.name,x,param{:});
        end
        
        function r = random(X,n,varargin)
            % r = random(X,n)
            % Generates n random numbers according to the distribution of the RandomVariable X
            % X: RandomVariable
            % n: integer
            % r: n-by-1 double
            
            if nargin > 2
                warning('random should have only two input arguments.')
            end
            
            if nargin==1
                n=1;
            end
            
            param = getParameters(X);
            if numel(n)>1
                error('n must be an integer.')
            end
            
            r = random(X.name,param{:},[n,1]);
        end
        
        function s = std(X)
            % s = std(X)
            % Computes the standard deviation of the random variable X
            % X: RandomVariable
            % s: 1-by-1 double
            
            [~,v] = randomVariableStatistics(X);
            s = sqrt(v);
        end
        
        function y = transfer(X,Y,x)
            % y = transfer(X,Y,x)
            % Transfers from the random variable X to the random variable Y at points x
            % X: RandomVariable
            % Y: RandomVariable
            % x: double
            % y: double
            
            if ~isa(X,'RandomVariable') || ~isa(Y,'RandomVariable')
                error('First two arguments must be RandomVariable')
            end
            
            y = icdf(Y,cdf(X,x));
        end
        
        function s = truncatedSupport(X)
            % s = truncatedSupport(X)
            % Returns the truncated support of the random variable X
            % X: RandomVariable
            % s: 1-by-2 double
            
            s = support(X);
            
            if s(1) == -Inf
                s(1) = min(mean(X)-10*std(X));
            end
            if s(2) == Inf
                s(2) = max(mean(X)+10*std(X));
            end
        end
        
        function v = variance(X)
            % s = variance(X)
            % Computes the variance of the random variable X
            % X: RandomVariable
            % v: 1-by-1 double
            
            [~,v] = randomVariableStatistics(X);
        end
        
        function varargout = pdfPlot(X,varargin)
            % varargout = pdfPlot(X,varargin)
            % Plots the probability density function of the random variable X
            % X: RandomVariable
            % varargin: ('npts',n), n: integer, creates a plot with n
            % points, ('bar',bool), bool: boolean, uses the bar function
            % instead of the plot function, ('options',options), options:
            % cell, gives to the bar or plot function input arguments
            % varargout: if nargout >=1, the pdf is given, if nargout >=2,
            % the points of evaluation are also given
            
            varargout = cell(1,nargout);
            [varargout{:}]=plot(X,'pdf',varargin{:});
        end
        
        function varargout = cdfPlot(X,varargin)
            % varargout = cdfPlot(X,varargin)
            % Plots the cumulative density function of the random variable X
            % X: RandomVariable
            % varargin: ('npts',n), n: integer, creates a plot with n
            % points, ('bar',bool), bool: boolean, uses the bar function
            % instead of the plot function, ('options',options), options:
            % cell, gives to the bar or plot function input arguments
            % varargout: if nargout >=1, the pdf is given, if nargout >=2,
            % the points of evaluation are also given
            
            varargout = cell(1,nargout);
            [varargout{:}] = plot(X,'cdf',varargin{:});
        end
        
        function varargout = icdfPlot(X,varargin)
            % varargout = icdfPlot(X,varargin)
            % Plots the inverse cumulative distribution function of the random variable X
            % X: RandomVariable
            % varargin: ('npts',n)
            % n: integer, creates a plot with n points, ('bar',bool)
            % bool: boolean, uses the bar function instead of the plot function, ('options',options)
            % options: cell, gives to the bar or plot function input arguments
            % varargout: if nargout >=1, the pdf is given, if nargout >=2, the points of evaluation are also given
            
            varargout = cell(1,nargout);
            [varargout{:}]=plot(X,'icdf',varargin{:});
        end
        
        function varargout = plot(X,varargin)
            % varargout = plot(X,h,varargin)
            % Plots the desired quantity, chosen between 'pdf', 'cdf' of 'icdf'.
            % X: RandomVariable
            % h: char ('pdf' or 'cdf' or 'icdf')
            %
            % See also RandomVariable/pdfPlot, RandomVariable/cdfPlot, RandomVariable/icdfPlot
            
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