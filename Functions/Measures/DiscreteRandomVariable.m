% Class DiscreteRandomVariable

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

classdef DiscreteRandomVariable < RandomVariable
    
    properties
        values
        probabilities
    end
    
    methods
        function X = DiscreteRandomVariable(values,probabilities)
            % X = DiscreteRandomVariable(values,probabilities)
            % DiscreteRandomVariable taking a finite set of values in R^d
            % values: N-by-d array containing the set of values (x_1,...,x_N) taken by the random variables
            % probabilities: array containing the probabilities P(X=x_1)...P(X=x_N)
            % By default, uniform Law: P(X=x_i)=1/N
            
            X@RandomVariable('discrete');
            
            if nargin==2 && size(values,1)==1 && length(probabilities)==length(values)
                values=values(:);
            end
            
            X.values = values;
            N = size(X.values,1);
            if nargin==1
                probabilities = repmat(1/N,N,1);
            elseif numel(probabilities)~=N
                error('arguments must have the same size')
            elseif ~all(probabilities>=0)
                error('all probabilities should be >=0')
            end
            X.probabilities = probabilities(:);
            if abs(sum(X.probabilities)-1) > eps
%                 warning('Renormalizing probabilities.')
                X.probabilities = X.probabilities/sum(X.probabilities);
            end
        end
        
        function Xstd = getStandardRandomVariable(X)
            % Xstd = getStandardRandomVariable(X)
            % Returns the standard random variable with mean 0 and standard deviation 1
            % X: DiscreteRandomVariable
            % Xstd: DiscreteRandomVariable
            
            x=X.values;
            x = (x - repmat(mean(X),size(x,1),1))...
                ./ repmat(std(X),size(x,1),1);
            
            Xstd = DiscreteRandomVariable(x,X.probabilities);
        end
        
        function s = support(X)
            % s = support(X)
            % Returns two extreme points of the box in R^d containing the set of possible values
            % X: DiscreteRandomVariable in R^d
            % s: 2-by-d array
            
            s=[min(X.values,[],1);max(X.values,[],1)];
        end
        
        function varargout = plot(X,type,varargin)
            % varargout = plot(X,type)
            % Plots the desired quantity, chosen between 'pdf', 'cdf' of 'icdf'.
            % X: RandomVariable
            % type: char ('pdf' or 'cdf' or 'icdf')
            
            x = X.values(:)';
            y = X.probabilities(:)';
            delta = max(x)-min(x);
            ax = [min(x)-delta/10,max(x)+delta/10];
            
            switch type
                case 'cdf'
                    x = x(:)';
                    y = cumsum(y);
                    plot([ax(1),x;x,ax(2)],[0,y;0,y],varargin{:});
                    %hold on
                    %plot(x,y,'o');
                    %hold off
                    xlim(ax)
                    ylim([0,max(y)*1.1])
                case 'pdf'
                    plot([x;x],[zeros(1,length(x));y],varargin{:});
                    %hold on
                    %plot(x,y,'o');
                    %hold off
                    xlim(ax)
                    ylim([0,max(y)*1.1])
                case 'icdf'
                    fplot(@(u) reshape(icdf(X,u(:)),size(u)),[0,1]);
                otherwise
                    error('wrong argument type')
            end
            if nargout >= 1
                varargout{1} = x;
            end
            if nargout >= 2
                varargout{2} = y;
            end
        end
        
        function G = integrationRule(X)
            weights = X.probabilities;
            points = X.values;
            G = IntegrationRule(points(:),weights);
        end
        
        function p = getParameters(X)
            % p = getParameters(X)
            % Returns the parameters of the discrete random variable X in a cell array
            % X: DiscreteRandomVariable
            % p: 1-by-2 cell
            
            p = {X.values,X.probabilities};
        end
        
        function c = cdf(X,x)
            % c = cdf(X,x)
            % Computes the cumulative density function of X at x
            %
            % X: DiscreteRandomVariable
            % x: array
            % c: array of size size(x)
            
            c = zeros(size(x));
            N = numel(X.values);
            M = numel(x);
            x=repmat(reshape(x,1,M),N,1);
            v=repmat(X.values(:),1,M);
            w=repmat(X.probabilities(:),1,M);
            c(:) = sum(w(v<=x),1);
        end
        
        function c = icdf(X,p)
            % x = icdf(X,p)
            % Computes the inverse cumulative density function of X at p (Quantile)
            %
            % X: DiscreteRandomVariable
            % p: array
            % x: array of size size(p)
            
            c = zeros(size(p));
            v = [-Inf;X.values(:)];
            F = [0;X.probabilities(:)];
            N = numel(F);
            M = numel(p);
            p=repmat(reshape(p,1,M),N,1);
            F = cumsum(F);
            F = repmat(F(:),1,M);
            T=(F>=p);
            [I,~] = find(cumsum(T,1)==1);
            c(:) = v(I);
        end
        
        function m = mean(X)
            % m = mean(X)
            
            m= X.probabilities'*X.values;
        end
        
        function v = var(X)
            % v = var(X)
            
            v=X.probabilities*X.values.^2-mean(X).^2;
        end
        
        function [m,v] = randomVariableStatistics(X)
            % [m,v] = randomVariableStatistics(X)
            % Computes the mean m and the variance v of the random variable X
            
            m = mean(X);
            v = var(X);
        end
        
        function r = random(X,n,varargin)
            % r = random(X,n)
            % Generates n random numbers according to the distribution of the DiscreteRandomVariable X over R^d
            % X: DiscreteRandomVariable
            % n: integer
            % r: double of size n-by-d
            if nargin==1
                n=1;
            end
            Y = DiscreteRandomVariable((1:length(X.probabilities))',X.probabilities);
            I = icdf(Y,rand(n,1));
            r = X.values(I,:);
        end
    end
end