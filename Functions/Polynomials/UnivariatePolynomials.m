% Class UnivariatePolynomials

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

classdef (Abstract) UnivariatePolynomials
    
    properties
        measure
    end
    
    methods
        function px = polyval(p,liste,x)
            % px = polyval(p,liste,x)
            % Evaluates the polynomials of order contained in liste at
            % points x
            % p: UnivariatePolynomials
            % liste: double
            % x: 1-by-n or n-by-1 double
            % px: n-by-length(liste) double
            
            L = fliplr(full(polyCoeff(p,0:max(liste))));
            
            switch min(size(x))
                case 0
                    px = zeros(0,length(liste));
                case 1
                    px = zeros(numel(x),size(L,1));
                    for i=1:size(L,1)
                        px(:,i) = polyval(L(i,:),x(:));
                    end
                    px = px(:,liste+1);
                otherwise
                    error('Method not implemented.')
            end
        end
        
        function c = dPolyCoeff(p,list)
            % c = dPolyCoeff(p,list)
            % Computes the coefficients of the monomials used to create the
            % first order derivative of the polynomials of degree specified
            % in list
            % p: UnivariatePolynomials
            % list: 1-by-n or n-by-1 array of integers
            % c: n-by-max(list)+1 double
            
            c = polyCoeff(p,list);
            c = c(:,2:end) .* repmat(1:max(list),length(list),1);
        end
        
        function c = dnPolyCoeff(p,n,list)
            % c = dnPolyCoeff(p,n,list)
            % Computes the coefficients of the monomials used to create the n-th order derivative of the polynomials of degree specified in list
            % p: UnivariatePolynomials
            % n: integer
            % list: 1-by-n or n-by-1 array of integers
            % c: n-by-max(list)+1 double
            
            c = polyCoeff(p,list);
            d = prod(repmat(0:max(list)-n,n,1) + repmat((1:n)',1,max(list)-n+1),1);
            c = c(:,n+1:end) .* repmat(d,length(list),1);
            
            if isempty(c)
                c = zeros(length(list),1);
            end
        end
        
        function px = dPolyval(p,list,x)
            % px = dPolyval(p,list,x)
            % Computes the first order derivative of polynomials of p of
            % degrees in list at points x
            % p: UnivariatePolynomials
            % list: d-by-1 or 1-by-d  array of integers
            % x: n-by-1 or 1-by-n double
            % px: n-by-d double
            
            px = dnPolyval(p,1,list,x);
        end
        
        function px = dnPolyval(p,n,list,x)
            % px = dnPolyval(p,list,x)
            % Computes the n-th order derivative of polynomials of p of
            % degrees in list at points x
            % p: UnivariatePolynomials
            % n: integer
            % list: d-by-1 or 1-by-d  array of integers
            % x: N-by-1 or 1-by-N double
            % px: N-by-d double
            
            L = fliplr(full(dnPolyCoeff(p,n,0:max(list))));
            
            switch min(size(x))
                case 0
                    px = zeros(0,length(list));
                case 1
                    px = zeros(numel(x),size(L,1));
                    for i=1:size(L,1)
                        px(:,i) = polyval(L(i,:),x(:));
                    end
                    px = px(:,list+1);
                otherwise
                    error('not implemented')
            end
        end
        
        function m = mean(p,list,measure)
            % MEAN - Mean of the polynomials of given degrees
            %
            % m = MEAN(p,list,measure)
            % Returns the mean of the polynomials p of degree
            % contained in list, with a Measure given by measure if
            % provided, or to the measure property else.
            % p: UnivariatePolynomials
            % list: 1-by-n or n-by-1 double
            % measure: Measure (optional)
            % m: n-by-1 double
            
            switch nargin
                case 2
                    assert(~isempty(p.measure),'Must provide a Measure.')
                    m = moment(p,reshape(list,length(list),1)) / mass(p.measure);
                case 3
                    m = moment(p,reshape(list,length(list),1),measure) / mass(measure);
            end
        end
        
        function m = moment(p,list,rv)
            % m = moment(p,I,X)
            % Computes the moments of the family of polynomials p_i(X) =
            % X^i, i in I, of a random variable X, using a gauss
            % integration rule.
            % p: UnivariatePolynomials
            % I: n-by-k array
            % X: RandomVariable
            % m: n-by-1 double
            %
            % If k = 1, returns the vector m = (E(p_i(X)): i in I),
            % if k = 2, returns the vector m = (E(p_i1(X)p_i2(X)): (i1,i2)
            % in I), etc.
            
            if nargin == 2
                rv = p.measure;
            end
            
            m = zeros(size(list,1),1);
            
            i = max(sum(list,2)); % Maximum degree of the product
            n = ceil((i+1)/2); % Number of points for the quadrature
            
            G = gaussIntegrationRule(rv,n); % Integration rule
            
            for i = 1:size(list,1)
                % Creation of the function to integrate
                P = @(x) polyval(p,list(i,1),x);
                for k = 2:size(list,2)
                    P = @(x) P(x).*polyval(p,list(i,k),x);
                end
                
                m(i) = integrate(G,P); % Integration
            end
        end
        
        function n = ndims(~)
            n = 1;
        end
    end
    
    methods (Abstract)
        % one -
        % [c,I] = one(p)
        % coefficients and corresponding indices for the decomposition of
        % the constant function 1
        [c,I] = one(p)
        
        % isOrthonormal -
        % ok = isOrthonormal(p)
        % Checks the orthonormality of the basis created by the
        % functions of p
        % p: UnivariatePolynomials
        % ok: boolean
        ok = isOrthonormal(p)
        
        % polyCoeff
        % c = polyCoeff(p,list)
        % Computes the coefficients of the monomials used to create the
        % polynomials of degree specified in list
        % p: UnivariatePolynomials
        % list: 1-by-n or n-by-1 array of integer
        % c: n-by-max(list)+1 double
        c = polyCoeff(p,list)
    end
end