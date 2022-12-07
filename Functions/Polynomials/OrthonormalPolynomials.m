% Class OrthonormalPolynomials

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

classdef OrthonormalPolynomials < UnivariatePolynomials
    

    methods
        function ok = isOrthonormal(p)
            % ok = isOrthonormal(p)
            % Checks the orthonormality of the basis created by the
            % functions of p
            % p: OrthonormalPolynomials
            % ok: boolean (false by default)
            
            ok = true;
        end
        
        function [c,I] = one(p)
            % [c,I] = one(p)
            % Returns c = 1 and I = 0
            % p: OrthonormalPolynomials
            % c: 1-by-1 double
            % I: 1-by-1 double
            
            c=1;
            I=0;
        end
        
        function ok = eq(p,q)
            % ok = eq(p,q)
            % Checks if the two objects p and q are identical
            % p: OrthonormalPolynomials
            % q: OrthonormalPolynomials
            % ok: boolean
            
            if ~isa(q,'OrthonormalPolynomials')
                ok = 0;
            else
                try
                    ok = all(full((p.measure == q.measure) & ...
                        all(p.isOrthonormal == q.isOrthonormal)));
                catch
                    ok = 0;
                end
            end
        end
        
        function s = domain(p)
            % s = domain(X)
            % Returns the support of the associated measure
            % p: OrthonormalPolynomials
            % s: 1-by-2 double
            
            s = support(p.measure);
        end
        
        function s = truncatedDomain(p)
            % s = truncatedDomain(X)
            % Returns the truncated support of the associated measure
            % p: OrthonormalPolynomials
            % s: 1-by-2 double
            
            s = truncatedSupport(p.measure);
        end
        
        function m = mean(p,list,measure)
            % m = mean(p,list,measure)
            % Returns the mean of the polynomials p of degree
            % contained in list, with a Measure given by measure if
            % provided, or to the measure property else.
            % p: OrthonormalPolynomials
            % list: 1-by-n or n-by-1 double
            % measure: Measure (optional)
            % m: n-by-1 double
            
            switch nargin
                case 2
                    m = moment(p,reshape(list,length(list),1)) / mass(p.measure);
                case 3
                    m = moment(p,reshape(list,length(list),1),measure) / mass(measure);
            end
        end
        
        function m = moment(p,list,rv)
            % m = moment(p,list,mu)
            % Computes the integral of products of polynomials of the
            % family p of degrees in list, using the gauss integration
            % rule. 
            % The integral is with respect to the Measure mu
            % which is taken as the Measure mu associated
            % to p if not provided in input
            % 
            % If list = [i1^1,...,ik^1 ; ... ; i1^n,...,ik^n], then 
            % m(l) = int p_{i1^l}(x)...p_{ik^l}(x) dmu(x)
            %
            % p: OrthonormalPolynomials
            % list: n-by-k double
            % rv: Measure (optional)
            % m: n-by-1 double
            
            m = zeros(size(list,1),1);
            
            if any(size(list,2) == [1 2]) && (nargin < 3 || rv == p.measure)
                switch size(list,2)
                    case 1
                        m(list == 0) = 1;
                    case 2
                        m(list(:,1) == list(:,2)) = 1;
                end
            else
                i = max(sum(list,2)); % Degree of the product
                n = ceil((i+1)/2); % Number of points for the quadrature
                
                if nargin < 3 || (nargin == 3 && ~isa(rv,'Measure'))
                    rv = p.measure;
                end
                
                G = gaussIntegrationRule(rv,n);
                
                for i = 1:size(list,1)
                    P = @(x) polyval(p,list(i,1),x);
                    for k = 2:size(list,2)
                        P = @(x) P(x).*polyval(p,list(i,k),x);
                    end
                    
                    m(i) = integrate(G,P);
                end
            end
        end
        
        function plot(p,d,varargin)
            % plot(p,d,varargin)
            % Plots the polynomial of degree d of the family p
            % p: OrthonormalPolynomials
            % d: array of integers
            % varargin: can contain 'xlim' to specify the limits of the x
            % axis, 'npts' to specify the number of points used for the
            % plot, 'options' to specify in a cell properties that are
            % passed to the plot function
            
            P = inputParser;
            addParamValue(P,'xlim',truncatedDomain(p),@isnumeric);
            addParamValue(P,'npts',200,@isscalar);
            addParamValue(P,'options',{});
            parse(P,varargin{:});
            
            x = linspace(P.Results.xlim(1),P.Results.xlim(2),P.Results.npts);
            px = polyval(p,d,x);
            
            plot(x,px,P.Results.options{:});
            xlim(P.Results.xlim);
            grid on
            xlabel('x')
            ylabel(['p_' num2str(d) '(x)'])
        end
        
        function c = polyCoeff(p,list)
            % c = polyCoeff(p,list)
            % Computes the coefficients of the monomials used to create the
            % polynomials of degree specified in list
            % p: OrthonormalPolynomials
            % list: 1-by-n or n-by-1 array of integer
            % c: n-by-max(list)+1 double
            
            i = max(list);
            
            recurr = recurrenceMonic(p,i);
            a = recurr(1,:);
            b = recurr(2,:);            
            
            c = zeros(i+1,i+1);
            c(1,1) = 1;
            
            if i > 0
                c(2,2:end) = c(1,1:end-1);
                c(2,:) = c(2,:) - a(1)*c(1,:);
                c(2,:) = c(2,:)/sqrt(b(2));
            end
            
            if i > 1
                for n = 2:i
                    c(n+1,2:end) = c(n,1:end-1);
                    c(n+1,:) = c(n+1,:) - a(n)*c(n,:) - sqrt(b(n))*c(n-1,:);
                    c(n+1,:)=c(n+1,:)/sqrt(b(n+1));
                end
            end
            
            c = sparse(c(list+1,:));
        end
        
        function px = polyval(P,list,x)
            % px = polyval(P,list,x)
            % Evaluates the polynomials of order contained in list at points x
            % P: OrthonormalPolynomials
            % list: 1-by-n or n-by-1 double
            % x: 1-by-m or m-by-1 double
            % px: m-by-n double
            
            if ~isempty(list)
                i = max(list);
                
                recurr = recurrenceMonic(P,i);
                a = recurr(1,:);
                b = recurr(2,:);

                px = zeros(length(x),i+1);
                px(:,1) = 1;
                if i>0
                    px(:,2) = (x(:) - a(1))/sqrt(b(2));
                    for n = 3:i+1
                        px(:,n) = (x(:) - a(n-1)).*px(:,n-1) - sqrt(b(n-1))*px(:,n-2);
                        px(:,n) = px(:,n)/sqrt(b(n));
                    end
                end

                px = px(:,list+1)/sqrt(mass(P.measure));
            else
                px = [];
            end
        end
        
        function px = dPolyval(P,list,x)
            % px = dPolyval(P,list,x)
            % Computes the first order derivative of polynomials of P of degrees in list at points x
            % P: OrthonormalPolynomials
            % list: d-by-1 or 1-by-d double
            % x: n-by-1 or 1-by-n double
            % px: n-by-d double
            
            i = max(list);
            

            recurr = recurrenceMonic(P,i);
            a = recurr(1,:);
            b = recurr(2,:);

            px = zeros(length(x),i+1);
            px(:,1) = 0;
            if i>0
                px(:,2) = 1/sqrt(b(2));
            end
            Pn = polyval(P,0:i,x);

            for n = 3:i+1
                px(:,n) = Pn(:,n-1) + (x(:) - a(n-1)).*px(:,n-1) - sqrt(b(n-1))*px(:,n-2);
                px(:,n) = px(:,n)/sqrt(b(n));
            end
            px = px(:,list+1);
        end
        
        function px = dnPolyval(P,n,list,x)
            % px = dnPolyval(p,list,x)
            % Computes the n-th order derivative of polynomials of p of
            % degrees in list at points x
            % p: OrthonormalPolynomials
            % n: integer
            % list: d-by-1 or 1-by-d  array of integers
            % x: N-by-1 or 1-by-N double
            % px: N-by-d double
            
            i = max(list);

            recurr = recurrenceMonic(P,i);
            a = recurr(1,:);
            b = recurr(2,:);
            
            px = polyval(P,0:i,x);
            
            for k = 1:n
                pxOld = px;
                
                px = zeros(length(x),i+1);
                px(:,1) = 0;
                if i>0
                    if k == 1
                        px(:,2) = 1/sqrt(b(2));
                    else
                        px(:,2) = 0;
                    end
                end
                for j = 3:i+1
                    px(:,j) = k*pxOld(:,j-1) + (x(:) - a(j-1)).*px(:,j-1) - sqrt(b(j-1))*px(:,j-2);
                    px(:,j) = px(:,j)/sqrt(b(j));
                end
                
            end
            px = px(:,list+1);
        end
        
        function [fx,x] = random(P,list,n,rv)
            % [fx,x] = random(P,list,n,rv)
            % Returns an array of size n of random evaluations of the
            % polynomials of P for which the degree is in list. If rv is
            % not provided, the random generation is performed using the
            % measure property of P.
            % P: OrthonormalPolynomials
            % list: p-by-1 of 1-by-p double
            % n: tuple of length d (n=1 by default)
            % rv: ProbabilityMeasure, optional
            % fx: (d+1)-dimensional array of size
            % n1-by...-by-nd-by-length(list)
            % x: d-dimensional array of size n1-by...-by-nd
            
            if nargin <= 2 || isempty(n)
                n=1;
            end
            if nargin <= 3
                rv = P.measure;
            end
            
            if ~isa(rv,'ProbabilityMeasure')
                error('Must provide a ProbabilityMeasure.')
            end
            
            x = random(rv,prod(n));
            
            fx = zeros(prod(n),length(list));
            for i = 1:length(list)
                fx(:,i) = polyval(P,list(i),x);
            end
            if numel(n)>1 && ~(numel(n)==2 && n(2)==1)
                fx = reshape(fx,[n(:)',length(list)]);
                x = reshape(x,n);
            end
        end
        
        function points = roots(p,n)
            % points = roots(p,n)
            % Returns the roots of the polynomial of degree n
            
                c = recurrenceMonic(p,n-1);
            
            % Jacobi matrix
            if n == 1
                J = diag(c(1,:));
            else
                J = diag(c(1,:)) + diag(sqrt(c(2,2:end)),-1) + diag(sqrt(c(2,2:end)),1);
            end
            
            d = eig(full(J));
            points = sort(d);
            points = reshape(points,n,1);
        end
    end
end
