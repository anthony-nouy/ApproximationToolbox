% Class EmpiricalRandomVariable

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

classdef EmpiricalRandomVariable < RandomVariable
    
    properties
        sample
        bandwidth
    end
    
    methods
        function X = EmpiricalRandomVariable(sample)
            % X = EmpiricalRandomVariable(sample)
            % Random variable fitted using gaussian kernel smoothing, this class gives best results in case of normal distributions.
            % A sample must be provided, which is used to fit a probability density function using Scott's rule
            % sample: 1-by-n or n-by-1 double
            % X: EmpiricalRandomVariable with parameters the sample and the optimal bandwidth selected using Scott's rule
            
            X@RandomVariable('empirical');
            
            if nargin == 0
                error('Must specify a sample')
            end
            
            X.sample = sample;
            X.bandwidth = 3.5*std(sample)*numel(sample)^(-1/3); % Scott's rule [Scott, D.W. (1992) Multivariate Density Estimation. Theory, Practice and Visualization. New York: Wiley.]
        end
        
        function X = shift(X,b,s)
            X.sample = s*X.sample+b;
            X.bandwidth = X.bandwidth*s;
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
            
            % If Y is the standard random variable associated to X
            if isa(Y,'EmpiricalRandomVariable') && norm(Y.sample - (X.sample(:) - mean(X.sample(:)))/std(X.sample(:))) / norm(Y.sample) < eps
                y = (x-mean(X.sample))/std(X.sample);
            else
                y = icdf(Y,cdf(X,x));
            end
        end
        
        function Xstd = getStandardRandomVariable(X)
            % Xstd = getStandardRandomVariable(X)
            % Returns the standard random variable with mean 0 and standard deviation 1
            % X: EmpiricalRandomVariable
            % Xstd: EmpiricalRandomVariable
            
            x = reshape(X.sample,length(X.sample),1);
            x = (x - repmat(mean(x,1),size(x,1),1))...
                ./ repmat(std(x),size(x,1),1);
            
            Xstd = X;
            Xstd.sample = x;
            Xstd.bandwidth = X.bandwidth / std(X.sample);
        end
        
        function s = support(X)
            % s = support(X)
            % Returns the support of the empirical random variable X
            % X: EmpiricalRandomVariable
            % s: 1-by-2 double
            
            s=[-Inf,Inf];
        end
        
        function p = orthonormalPolynomials(X,varargin)
            % p = orthonormalPolynomials(X)
            % Returns the orthonormal polynomials according to the EmpiricalRandomVariable X
            % X: EmpiricalRandomVariable
            % p: EmpiricalPolynomials
            
            p = EmpiricalPolynomials(X,varargin{:});
            m = mean(X.sample);
            s = std(X.sample);
            if abs(m)>eps || abs(s-1)>eps
                warning('ShiftedOrthonormalPolynomials are created')
                p = ShiftedOrthonormalPolynomials(p,m,s);
            end
        end
        
        function p = getParameters(X)
            % p = getParameters(X)
            % Returns the parameters of the empirical random variable X in an array
            % X: EmpiricalRandomVariable
            % p: 1-by-2 cell
            
            p = {X.sample,X.bandwidth};
        end
        
        function c = cdf(X,x)
            % c = cdf(X,x)
            % Computes the cumulative density function of X, using Matlab's Statistics and Machine Learning Toolbox's function ksdensity
            % X: EmpiricalRandomVariable
            % x: 1-by-n or n-by-1 double
            % c: 1-by-n or n-by-1 double
            
            c = ksdensity(X.sample,x,'bandwidth',X.bandwidth,'function','cdf');
        end
        
        function c = icdf(X,x)
            % c = icdf(X,x)
            % Computes the inverse cumulative density function of X, using Matlab's Statistics and Machine Learning Toolbox's function ksdensity
            % X: EmpiricalRandomVariable
            % x: 1-by-n or n-by-1 double
            % c: 1-by-n or n-by-1 double
            
            c = ksdensity(X.sample,x,'bandwidth',X.bandwidth,'function','icdf');
        end
        
        function px = pdf(X,x)
            % px = pdf(X,x)
            % Computes the probability density function of the random variable X at points x
            % X: EmpiricalRandomVariable
            % x: 1-by-n or n-by-1 double
            % px: 1-by-n or n-by-1 double
            
            param = getParameters(X);
            xi = param{1};
            h = param{2};
            n = length(xi);
            
            X = repmat(x(:).',length(xi),1);
            XI = repmat(xi(:),1,length(x));
            px = 1/(n*h*sqrt(2*pi))*sum(exp(-0.5*(X-XI).^2/h^2),1);
        end
        
        function [m,v] = randomVariableStatistics(X)
            % [m,v] = randomVariableStatistics(X)
            % Computes the mean m and the variance v of the empirical random variable X
            % X: EmpiricalRandomVariable
            % m: 1-by-1 double
            % v: 1-by-1 double
            
            m = mean(X.sample);
            v = var(X.sample);
        end
        
        function varargout = cdfPlot(X,h,varargin)
            % varargout = cdfPlot(X,h,varargin)
            % Plots the cumulative distribution function of the random variable X
            % When h is set to 'histogram', an histogram is plotted with the bars height normalized to match a cdf
            % X: EmpiricalRandomVariable
            % h: char
            % varargin: ('npts',n)
            % n: integer, creates a plot with n points, ('bar',bool)
            % bool: boolean, uses the bar function
            % instead of the plot function, ('options',options)
            % options: cell, gives to the bar or plot function input arguments
            % varargout: if nargout >=1, the cdf is given, if nargout >=2, the points of evaluation are also given
            
            if exist('h','var') && strcmpi(h,'histogram')
                histogram(X.sample,'Normalization','cdf');
                hold all;
            end
            varargout = cell(1,nargout);
            [varargout{:}] = cdfPlot@RandomVariable(X,varargin{:});
        end
        
        function varargout = pdfPlot(X,h,varargin)
            % varargout = cdfPlot(X,h,varargin)
            % Plots the probabiloty density function of the random
            % variable X
            % When h is set to 'histogram', an histogram is plotted with
            % the bars height normalized to match a pdf
            % X: EmpiricalRandomVariable
            % h: char
            % varargin: ('npts',n)
            % n: integer, creates a plot with n points, ('bar',bool)
            % bool: boolean, uses the bar function instead of the plot function, ('options',options)
            % options: cell, gives to the bar or plot function input arguments
            % varargout: if nargout >=1, the pdf is given, if nargout >=2,
            % the points of evaluation are also given
            
            if exist('h','var') && strcmpi(h,'histogram')
                histogram(X.sample,'Normalization','pdf');
                hold all;
            end
            varargout = cell(1,nargout);
            [varargout{:}] = pdfPlot@RandomVariable(X,varargin{:});
        end
        
        function r = random(X,n)
            % r = random(X,n)
            % Generates random numbers according to the distribution of the EmpiricalRandomVariable X
            % X: EmpiricalRandomVariable
            % n: 1-by-1 integer
            % r: n-by-1 double
            
            r = icdf(X,rand(n,1));
        end
    end
end