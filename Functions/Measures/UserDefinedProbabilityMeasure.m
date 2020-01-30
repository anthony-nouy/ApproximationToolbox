% Class UserDefinedProbabilityMeasure

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

classdef UserDefinedProbabilityMeasure < ProbabilityMeasure
    
    properties
        dim
        pdffun = [] % Probability density function
        cdffun = [] % Cumulative density function
        icdffun = [] % Inverse cumulative density function
        supp = []; % Support
    end
    
    methods
        function mu = UserDefinedProbabilityMeasure(dim, varargin)
            mu.dim = dim;
            p = ImprovedInputParser;
            addParamValue(p,'pdffun',[]);
            addParamValue(p,'cdffun',[]);
            addParamValue(p,'icdffun',[]);
            addParamValue(p,'supp',[]);
            parse(p,varargin{:});
            mu = passMatchedArgsToProperties(p,mu);
        end
        
        function n = ndims(mu)
            n = mu.dim;
        end
        
        function px = pdf(mu,x)
            if ~isempty(mu.pdffun)
                px = mu.pdffun(x);
            else
                error('Should provide pdf function.')
            end
        end
        
        function Fx = cdf(mu,x,tol)
            if ~isempty(mu.cdffun)
                Fx = mu.cdffun(x);
            else
                if mu.dim>1
                    error('Method not implemented for dim>1.')
                end
                if nargin<3
                    tol = 1e-12;
                end
                x = x(:);
                s = mu.supp;
                if isempty(s)
                    error('Should provide the support.')
                end
                n=size(x,1);
                Fx = zeros(n,1);
                Fx(x >= max(s))=1;
                Fx(x < min(s))=0;
                rep = find(isIn(s,x));
                for k=1:length(rep)
                    Fx(rep(k))=quadgk(@(t) mu.pdf(t')',min(s),x(rep(k)),'RelTol',tol);
                end
            end
        end
        
        function q = icdf(mu,u,tol,s)
            % q = icdf(mu,u,tol,s)
            % Computes the inverse cdf of measure mu at u
            % tol: tolerance for fzero finding algorithm
            % s: for unbounded support, interval in which the values are searched
            
            if ~isempty(mu.icdffun)
                q = mu.icdffun(u);
            else
                if mu.dim>1
                    error('Method not implemented.')
                end
                if nargin<3 || isempty(tol)
                    tol = 1e-6;
                end
                if max(u)>1 || min(u)<0
                    error('Input argument should be between 0 and 1.')
                end
                if nargin<4
                    s = mu.supp;
                end
                if isempty(s)
                    error('Should provide the support.')
                end
                u=u(:);
                n = length(u);
                q=zeros(n,1);
                [us,I] = sort(u);
                x0 = [min(s),max(s)];
                options = optimset('TolFun',tol);
                for k=1:length(us)
                    [x,fval,flag,output]=fzero(@(x) mu.cdf(x')' - us(k),x0,options);
                    if flag~=1
                        q(I(k))=NaN;
                    else
                        q(I(k))=x;
                        x0 = [x,x0(2)];
                    end
                end
            end
        end
        
        function o = eq(p,q)
            o = false;
        end
        
        function s = support(mu)
            s = mu.supp;
        end
        
        function s = truncatedSupport(mu)
            s = mu.supp;
        end
        
        function xs = random(s,n)
            % Generate n samples given the ProbabilityMeasure
            % This method should be avoided when the dimension is high
            % n: number of samples
            % 'pdf': probability density function
            % Uses the matlab function slice sampling
            % For documentation see:
            % Neal, Radford M. Slice Sampling. Ann. Stat. Vol. 31, No. 3, pp. 705–767, 2003
            % x array of size n x dim
            
            if s.dim > 5
                warning('Use of slice sampling, may be not efficient in high dim.')
            end
            xs = slicesample(zeros(1,s.dim), n, 'pdf',s.pdffun);
        end
    end
end