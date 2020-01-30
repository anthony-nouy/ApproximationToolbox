% Class ShiftedOrthonormalPolynomials

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

classdef ShiftedOrthonormalPolynomials < UnivariatePolynomials
    
    properties
        p % OrthonormalPolynomials
        b % shift
        s % scaling
    end
    
    methods
        
        function sp = ShiftedOrthonormalPolynomials(p,b,s)
            % sp = ShiftedOrthonormalPolynomials(p,b,s)
            % p: OrthonormalPolynomials
            % b: shift
            % s: scaling
            sp.measure = shift(p.measure,b,s);
            sp.p = p;
            sp.s = s;
            sp.b = b;
        end
        
        function ok = isOrthonormal(p)
            % ok = isOrthonormal(p)
            % % Checks the orthonormality of the basis created by the
            % functions of p
            % p: ShiftedOrthonormalPolynomials
            % ok: boolean (false by default)
            
            ok = true;
        end
        
        function [c,I] = one(p)
            % [c,I] = one(p)
            % Returns c = 1 and I = 0
            % p: ShiftedOrthonormalPolynomials
            % c: 1-by-1 double
            % I: 1-by-1 double
            
            c=1;
            I=0;
        end
        
        function ok = eq(p,q)
            % ok = eq(p,q)
            % Checks if the two objects p and q are identical
            % p: ShiftedOrthonormalPolynomials
            % q: ShiftedOrthonormalPolynomials
            % ok: boolean
            
            if ~strcmp(class(p),class(q))
                ok = 0;
            else
                ok = (p.s==q.s) && (p.b==q.b) && (p.p == q.p);
            end
        end
        
        function D = domain(p)
            % D = domain(X)
            % Returns the support of the associated measure
            % p: ShiftedOrthonormalPolynomials
            % D: 1-by-2 double
            
            D = domain(p.p);
            D = p.b + p.s*D;
        end
        
        function D = truncatedDomain(p)
            % D = truncatedDomain(X)
            % Returns the truncated support of the associated measure
            % p: ShiftedOrthonormalPolynomials
            % D: 1-by-2 double
            
            D = truncatedDomain(p.p);
            D = p.b + p.s*D;
        end
        
        function m = mean(p,list,varargin)
            % m = mean(p,list)
            % Returns the mean of the polynomials of the family p of degree contained in list
            % p: ShiftedOrthonormalPolynomials
            % list: 1-by-n or n-by-1 double
            % m: n-by-1 double
            m = mean(p.p,list,varargin{:}) ;
        end
        
        function m = moment(p,list,varargin)
            % m = moment(p,list)
            % Computes the inner product between polynomials of the
            % same family p of degrees in list, using the gauss integration
            % rule. The degrees of the polynomials are stored in the rows
            % of list, hence every row of list is related to one moment.
            % p: ShiftedOrthonormalPolynomials
            % list: n-by-m double
            % m: n-by-1 double
            
            if nargin == 3
                if varargin{1} == p.measure
                    varargin = {};
                else
                    varargin{1} = shift(varargin{1},-p.b/p.s,1/p.s);
                end
            end
            m = moment(p.p,list,varargin{:});
        end
        
        function plot(p,d,varargin)
            % plot(p,d,varargin)
            % Plots the polynomial of degree d of the family p
            % p: ShiftedOrthonormalPolynomials
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
        
        function px = polyval(P,list,x)
            % px = polyval(P,list,x)
            % Evaluates the polynomials of order contained in list at
            % points x
            % P: ShiftedOrthonormalPolynomials
            % list: 1-by-n or n-by-1 double
            % x: 1-by-m or m-by-1 double
            % px: m-by-n double
            
            x = (x-P.b)/P.s;
            px = polyval(P.p,list,x);
        end
        
        function px = dPolyval(P,list,x)
            % px = dPolyval(P,list,x)
            % Computes the first order derivative of polynomials of P of
            % degrees in list at points x
            % P: ShiftedOrthonormalPolynomials
            % list: d-by-1 or 1-by-d double
            % x: n-by-1 or 1-by-n double
            % px: n-by-d double
            
            x = (x-P.b)/P.s;
            px = dPolyval(P.p,list,x)/P.s;
            
        end
        
        function px = dnPolyval(P,n,list,x)
            % px = dnPolyval(p,list,x)
            % Computes the n-th order derivative of polynomials of p of
            % degrees in list at points x
            % p: ShiftedOrthonormalPolynomials
            % n: integer
            % list: d-by-1 or 1-by-d  array of integers
            % x: N-by-1 or 1-by-N double
            % px: N-by-d double
            
            x = (x-P.b)/P.s;
            px = dPolyval(P.p,list,x)/(P.s^n);
            
        end
        
        function [fx,x] = random(P,varargin)
            % [fx,x] = random(P,list,n)
            % Returns an array of size n of random evaluations of the
            % polynomials of P for which the degree is in list.
            % P: ShiftedOrthonormalPolynomials
            % list: p-by-1 of 1-by-p double
            % n: tuple of length d (n=1 by default)
            % fx: (d+1)-dimensional array of size
            % n1-by...-by-nd-by-length(list)
            % x: d-dimensional array of size n1-by...-by-nd
            
            [fx,x] = random(P.p,varargin{:});
            x = P.b + P.s*x;
        end
        
        function x = roots(p,n)
            % points = roots(p,n)
            % Returns the roots of the polynomial of degree n
            
            x = roots(p.p,n);
            x = p.b + p.s*x;
        end
        
        function n = ndims(p)
            n=1;
        end
        
        function c = polyCoeff(p,list)
            error('Method not implemented.');
        end
    end
end