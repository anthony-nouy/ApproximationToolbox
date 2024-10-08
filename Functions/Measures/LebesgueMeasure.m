% Class LebesgueMeasure

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

classdef LebesgueMeasure < Measure
    
    properties
        a
        b
    end
    
    methods
        function l = LebesgueMeasure(a,b)
            % Constructor for LebesgueMeasure on an interval [a,b]
            %
            % l = LebesgueMeasure(a,b)
            % or
            % l = LebesgueMeasure([a,b])
            
            if nargin==1
                l.a = a(1);
                l.b = a(2);
            else
                l.a = a;
                l.b = b;
            end
        end
        
        function l = shift(l,b,s)
            % l = shift(l,m,s)
            % if l is a Lebesgue measure on [a,b], returns the Lebesgue measure on interval [m+sa,m+sb]
            % l: LebesgueMeasure
            
            l.a = s*l.a+b;
            l.b = s*l.b+b;
        end
        
        function px = pdf(l,x)
            px = zeros(size(x,1));
            px((x>=l.a)&(x<=l.b)) = 1;
        end
        
        function n = ndims(l)
            n = 1;
        end
        
        function ok = eq(m1,m2)
            if ~(isa(m1,'LebesgueMeasure') && isa(m2,'LebesgueMeasure') )
                ok = 0;
            elseif ~strcmp(class(m1),class(m2))
                ok = 0;
            else
                ok = all(support(m1) == support(m2));
            end
        end
        
        function m = mass(l)
            m = l.b-l.a;
        end
        
        function s = support(l)
            s = [l.a,l.b];
        end
        
        function s = truncatedSupport(l)
            % s = truncatedSupport(X)
            % s: 1-by-2 double
            
            s = support(l);
            
            if s(1) == -Inf || s(2) == Inf
                error('truncated support not defined for Lebesgue measure on unbounded domain')
            end
        end


        function Y = discretize(X,n)
            % Discretize the LebesgueMeasure and returns a
            % discrete measure
            % 
            % Y = discretize(X,n)

            s = X.support();
            Y = DiscreteMeasure(discretize(UniformRandomVariable(s(1),s(2)),n));
            Y.weights = Y.weights * (s(2)-s(1));
        end
        
        function Y = discretizeSupport(X,varargin)
            % Discretize the LebesgueMeasure and returns a
            % discrete measure
            % 
            % Y = discretizeSupport(X,x)
            % For a n-by-1 array x, returns a discrete measure
            % taking n possible values x(1),...x(n)
            %
            % Y = discretizeSupport(X,n,[a,b])
            % For an integer n, returns a discrete measure
            % taking n possible values x(1),...x(n)
            s = X.support();
            Y = DiscreteMeasure(discretizeSupport(UniformRandomVariable(s(1),s(2)),varargin{:}));
            Y.weights = Y.weights * (s(2)-s(1));


        end
        
        function p = orthonormalPolynomials(l,varargin)
            % p = orthonormalPolynomials(l)
            % Returns the orthonormal polynomials according to the Lebesgue measure on an interval
            % l: LebesgueMeasure
            % p: ShiftedOrthonormalPolynomials
            
            p = LegendrePolynomialsLebesgue(varargin{:});
            if (l.a~=-1) || (l.b~=1)
                p = ShiftedOrthonormalPolynomials(p,(l.a+l.b)/2,(l.b-l.a)/2);
            end
            p.measure = l;
        end
        
        function G = gaussIntegrationRule(l,n)
            % G = gaussIntegrationRule(X,n)
            % Returns the n-points gauss integration rule associated with the Lebesgue measure on a bounded interval, using Golub-Welsch algorithm
            % l: LebesgueMeasure
            % n: integer
            % G: IntegrationRule
            
            X = UniformRandomVariable(l.a,l.b);
            G = gaussIntegrationRule(X,n);
            G.weights = G.weights*mass(l);
        end
        
        function x = random(l,n)
            % x = random(l,n)
            % Returns n samples from the uniform distribution over the
            % support of the measure, is this support is bounded.
            % l: LebesgueMeasure
            % n: integer
            if (l.b - l.a)==Inf
                error('the support of the Lebesgue measure should be bounded')
            end
            x = random(UniformRandomVariable(l.a,l.b),n);
            
        end

        function m = moment(l,I)
            assert(mass(l)<inf,'unbounded moments')
            m = moment(UniformRandomVariable(l.a,l.b),I)*(l.b-l.a);

        end
    end
end
