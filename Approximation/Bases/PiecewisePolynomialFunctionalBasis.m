% Class PiecewisePolynomialFunctionalBasis

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

classdef PiecewisePolynomialFunctionalBasis < FunctionalBasis
    
    properties
        points
        p
        s
    end
    
    methods
        function h = PiecewisePolynomialFunctionalBasis(points,p)
            % h = PiecewisePolynomialFunctionalBasis(points,p)
            % Constructor for the PiecewisePolynomialFunctionalBasis class, which
            % defines a functional basis compased of piecewise polynomials of degree p
            % on an interval [a,b]
            % The basis is orhtonormal with respect to the uniform probability measure
            % on (a,b)
            % points: a set of n+1 points [a=x0,...xn=b]
            % p: integer
            % h: PiecewisePolynomialFunctionalBasis
            
            h.points = points(:);
            if length(p)==1
                p = repmat(p,1,length(h.points)-1);
            end
            h.p = p(:)';
            h.isOrthonormal = true;
            h.measure = LebesgueMeasure(h.points(1),h.points(end));
        end
        
        function hx = eval(h,x,indices)
            % hx = eval(h,x,indices)
            % Evaluates the functional basis at the points x
            % h: PiecewisePolynomialFunctionalBasis
            % x: N-by-d array
            % hx: N-by-n array
            
            pos = elementNumber(h,x);
            p1 = h.points(1:end-1);
            p2 = h.points(2:end);
            l = p2-p1;
            u = (x-p1(pos))./l(pos);
            U = LebesgueMeasure(0,1);
            pol = PolynomialFunctionalBasis(orthonormalPolynomials(U),0:max(h.p));
            pu = pol.eval(u);
            hx = sparse([],[],[],length(x),cardinal(h),length(x)*max(h.p+1));
            for i=1:length(h.points)-1
                I=(pos==i);
                if ~isempty(I)
                    J = sum(h.p(1:i-1)+1)+(1:h.p(i)+1);
                    scale = l(i);%/(h.points(end)-h.points(1));
                    hx(I,J)=pu(I,1:h.p(i)+1)/sqrt(scale);
                end
            end
            
            if nargin==3
                hx = hx(:,indices);
            end
        end

        function hx = evalDerivative(h,k,x,indices)
            % hx = evalDerivative(h,k,x,indices)
            % Evaluates the k-th derivative of the functional basis at the points x
            % h: PiecewisePolynomialFunctionalBasis
            % x: N-by-d array
            % hx: N-by-n array
            
            pos = elementNumber(h,x);
            p1 = h.points(1:end-1);
            p2 = h.points(2:end);
            l = p2-p1;
            u = (x-p1(pos))./l(pos);
            U = LebesgueMeasure(0,1);
            pol = PolynomialFunctionalBasis(orthonormalPolynomials(U),0:max(h.p));
            pu = pol.evalDerivative(k,u);
            hx = sparse([],[],[],length(x),cardinal(h),length(x)*max(h.p+1));
            for i=1:length(h.points)-1
                I=(pos==i);
                if ~isempty(I)
                    J = sum(h.p(1:i-1)+1)+(1:h.p(i)+1);
                    scale = l(i);%/(h.points(end)-h.points(1));
                    hx(I,J)=pu(I,1:h.p(i)+1)/sqrt(scale)*l(i)^(-k);
                end
            end
            
            if nargin==4
                hx = hx(:,indices);
            end
        end


        
        function n = cardinal(h)
            n = sum(h.p+1);
        end
        
        function n = ndims(h)
            n=1;
        end
        
        function m = mean(h,varargin)
            p1 = h.points(1:end-1);
            p2 = h.points(2:end);
            l = p2-p1;
            m = zeros(cardinal(h),1);
            q = cumsum(([0,h.p(1:end-1)]+1));
            m(q) = sqrt(l);%/(h.points(end)-h.points(1));
        end
        

        function I = gaussIntegrationRule(h,n)
            % function I = gaussIntegrationRule(h,n) 
            % returns an Integration Rule that integrates extactly
            % piecewise polynomials of degree 2*n-1, using n-points
            % gauss integration rule per interval
            % h : PiecewisePolynomialFunctionalBasis
            % n : integer
            % I : IntegrationRule

            w = zeros(1,0);
            x = zeros(0,1);
            
            for k=1:length(h.points)-1
                supp = h.points(k:k+1);
                g = gaussIntegrationRule(LebesgueMeasure(supp(1),supp(2)),n);
                x = [x;g.points];
                w = [w,g.weights];
            end

            I = IntegrationRule(x,w);
        end


        function x = interpolationPoints(h,varargin)
            % x = interpolationPoints(h)
            % Interpolation points for PiecewisePolynomialFunctionalBasis
            % rescaled and shifted set of Chebychev points on each element
            
            pu = unique(h.p);
            u = cell(1,length(pu));
            for k=1:length(u)
                u{k} = (1+chebyshevPoints(pu(k)+1))/2;
            end
            
            p1 = h.points(1:end-1);
            p2 = h.points(2:end);
            l = p2-p1;
            x = zeros(cardinal(h),1);
            for i=1:length(h.points)-1
                ui = u{h.p(i)==pu};
                J = sum(h.p(1:i-1)+1)+(1:h.p(i)+1);
                x(J)=p1(i)+ui*l(i);
            end
        end
        
        function x = magicPoints(h,varargin)
            % x = magicPoints(h)
            % Magic points for PiecewisePolynomialFunctionalBasis
            % rescaled and shifted set of magic points
            % of a PolynomialFunctionalBasis on each element
            U = UniformRandomVariable(0,1);
            pol = PolynomialFunctionalBasis(orthonormalPolynomials(U),0:max(h.p));
            u = magicPoints(pol);
            
            p1 = h.points(1:end-1);
            p2 = h.points(2:end);
            l = p2-p1;
            x = zeros(cardinal(h),1);
            for i=1:length(h.points)-1
                J = sum(h.p(1:i-1)+1)+(1:h.p(i)+1);
                x(J)=p1(i)+u(1:h.p(i)+1)*l(i);
            end
        end
    end
    
    methods (Hidden)
        function pos = elementNumber(h,x)
            n = length(h.points)-1;
            X = repmat(x,1,n);
            P1 = repmat(h.points(1:end-1)',length(x),1);
            P2 = repmat(h.points(2:end)',length(x),1);
            I = find((X>=P1) & (X<P2));
            [I,J] = ind2sub([length(x),n],I);
            pos=zeros(length(x),1);
            pos(I)=J;
            pos(x>=h.points(end))=n;
            pos(x<h.points(1))=1;
            %if any(pos==0)
            %    error('points outside the interval of definition of the basis')
            %end
        end
    end
    
    methods (Static)
        function B = hp(a,b,h,p)
            % H = hp(a,b,h,p)
            % a,b: interval
            % h: mesh size
            % p: polynomial degree
            
            n = ceil((b-a)/h);
            points = linspace(a,b,n+1);
            B = PiecewisePolynomialFunctionalBasis(points,p);
        end
        
        function B = np(a,b,n,p)
            % H = np(a,b,n,p)
            % a,b: interval
            % n: number of elements
            % p: polynomial degree
            
            points = linspace(a,b,n+1);
            B = PiecewisePolynomialFunctionalBasis(points,p);
        end
        
        function B = singularityhpAdapted(a,b,s,h)
            % B = singularityhpAdapted(a,b,s,h)
            % Creates a PiecewisePolynomialFunctionalBasis B defined on the interval [a,b]
            % adapted in h and p around singularities
            % h is the size of elements near singularities
            % a,b: double
            % s: array of double (coordinates of singularities)
            % h: double (mesh size near singularity)
            
            e = s(:);
            if ~ismember(a,e)
                e = [a,e];
            end
            if ~ismember(b,e)
                e = [e,b];
            end
            n = length(e)-1;
            p = [];
            x = [];
            for i=1:n
                ai = e(i);
                bi = e(i+1);
                li = bi-ai;
                ne = ceil(log2(h^-1));
                if ismember(ai,s) && ismember(bi,s)
                    pi = [0:ne-1,ne-1:-1:0];
                    xi = [0,2.^-(ne:-1:2),1/2,1-2.^-(2:ne),1]';
                elseif ismember(ai,s)
                    %pi = [0:ne-1];
                    xi = [0,2.^-(ne:-1:1),1]';
                    pi = 0:length(xi)-2;
                elseif ismember(bi,s)
                    %pi = [ne-1:-1:0];
                    xi = [0,1-2.^-(1:ne),1]';
                    pi = length(xi)-2:-1:0;
                end
                
                if i<n
                    xi=xi(1:end-1);
                end
                xi = ai+xi*li;
                p = [p,pi];
                x=[x;xi];
            end
            B = PiecewisePolynomialFunctionalBasis(x,p);
        end
    end
end