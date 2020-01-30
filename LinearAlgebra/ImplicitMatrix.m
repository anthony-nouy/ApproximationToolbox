% Class ImplicitMatrix

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

classdef ImplicitMatrix
    
    properties
        mtimes_right
        mtimes_left
        sz
        ctranspose_flag
        tolerance = 1e-6
        maxIterations = 100
    end
    
    methods (Access = public)
        function x = ImplicitMatrix(varargin)
            % ImplicitMatrix - Constructor of the class ImplicitMatrix
            %
            % x = ImplicitMatrix(sz,mtimes_right,mtimes_left)
            %
            % sz: size of the matrix
            % mtimes_right: multiplication operation on the right side
            % mtimes_left: multiplication operation on the left side
            % x: ImplicitMatrix
            % Or
            % x = ImplicitMatrix(A)
            % A: double matrix
            % Or
            % x = ImplicitMatrix(A)
            % A: ImplicitMatrix
            
            if length(varargin) == 1
                if  isa(varargin{1},'double')
                    x.mtimes_right = @(t) varargin{1}*t;
                    x.mtimes_left = @(t) t*varargin{1};
                    x.sz = size(varargin{1});
                    x.ctranspose_flag = 0;
                elseif isa(varargin{1},'ImplicitMatrix')
                    x.mtimes_right = varargin{1}.mtimes_right;
                    x.mtimes_left = varargin{1}.mtimes_left;
                    x.sz = size(varargin{1});
                    x.ctranspose_flag = varargin{1}.ctranspose_flag;
                end
            else
                x.sz = varargin{1};
                x.mtimes_right = varargin{2};
                x.mtimes_left = varargin{3};
                x.ctranspose_flag = 0;
            end
        end
        
        function y = mtimes(x,y)
            % z = mtimes(x,y)
            % x: double or ImplicitMatrix
            % y: double or ImplicitMatrix
            % z: double or ImplicitMatrix
            if isa(x,'ImplicitMatrix') && isa(y,'double')
                if length(y)==1
                    mr = @(t) y*(x*t);
                    ml = @(t) y*(t*x);
                    y  = ImplicitMatrix(x.sz,mr,ml);
                elseif x.ctranspose_flag==0
                    y=x.mtimes_right(y);
                else
                    y=x.mtimes_left(y')';
                end
            elseif isa(x,'double') && isa(y,'ImplicitMatrix')
                y = (y'*x')';
            elseif isa(x,'ImplicitMatrix') && isa(y,'ImplicitMatrix')
                mr = @(t) x*(y*t);
                ml = @(t) (t*x)*y;
                y  = ImplicitMatrix([size(x,1), size(y,2)],mr,ml);
            else
                error('Method not implemented.');
            end
        end
        
        function sz = size(x,i)
            % sz = size(x,i)
            
            if nargin==1
                sz=x.sz;
            else
                sz=x.sz(i);
            end
        end
        
        function x = ctranspose(x)
            % x = ctranspose(x)
            
            x.ctranspose_flag=~x.ctranspose_flag;
            x.sz=fliplr(x.sz);
        end
        
        function z = plus(x,y)
            % z = plus(x, y)
            % x: double or ImplicitMatrix
            % y: double or ImplicitMatrix
            % z: ImplicitMatrix
            
            mr = @(t) x*t + y*t;
            ml = @(t) t*x + t*y;
            z  = ImplicitMatrix(size(x),mr,ml);
        end
        
        function z = uminus(x)
            % z = uminus(x)
            % x: ImplicitMatrix
            % z: ImplicitMatrix, z = -x
            
            mr = @(t) - x*t;
            ml = @(t) - t*x;
            z  = ImplicitMatrix(x.sz,mr,ml);
        end
        
        function z = minus(x,y)
            % z = minus(x, y)
            % x: ImplicitMatrix
            % y: double or ImplicitMatrix
            % z: ImplicitMatrix, z = x - y
            
            z  = x + (-y);
        end
        
        function [z,flag] = mldivide(x,y)
            % z = mldivide(x,y)
            % x: ImplicitMatrix
            % y: double
            % z: double, z = x^{-1} y computed by GMRES method
            
            [z,flag] = gmres(@(t)x*t,y,10,x.tolerance,x.maxIterations);
        end
    end
    
    methods (Static)
        function [x,Q] = cholInverse(A)
            % x = cholInverse(A)
            % A: double, symmetric positive definite matrix
            % x: ImplicitMatrix such that x = A^{-1} computed from a Cholesky factorization
            % Q: s.t. Q'*Q = A
            
            perm = symamd(A);
            rev(perm) = [1:size(A,1)];
            L = chol(A(perm,perm),'lower');
            
            permute = @(x,prm) x(prm,:);
            
            mtimes_right = @(x) permute(L.' \ ( L \permute(x,perm)), rev) ;
            mtimes_left  = @(x) permute((L.' \ ( L \permute(x.',perm)) ), rev).';
            
            sz = size(A);
            x = ImplicitMatrix(sz,mtimes_right,mtimes_left);
            
            mtimes_right = @(x)  (permute(x,perm).'*L).';
            mtimes_left  = @(x)  permute((L*x.'),rev).';
            
            Q = ImplicitMatrix(sz,mtimes_right,mtimes_left);
        end
        
        function x = luInverse(A)
            % x = luInverse(A)
            % A: double, invertible matrix
            % x: ImplicitMatrix such that x = A^{-1} computed from a LU factorization
            
            [L,U,P,Q] = lu(A);
            
            mtimes_right = @(x) Q*(U\(L\(P*x)));
            mtimes_left = @(x) (P.'*(L.'\(U.'\(Q.'*x.')))).';
            
            sz = size(A);
            x = ImplicitMatrix(sz,mtimes_right,mtimes_left);
        end
        
        function x = iluInverse(A)
            % x = iluInverse(A)
            % A: double, invertible matrix
            % x: ImplicitMatrix such that x = A^{-1} computed from a ILU factorization
            
            [L,U] = ilu(A);
            
            mtimes_right = @(x) (U\(L\(x)));
            mtimes_left = @(x) ((L.'\(U.'\(x.')))).';
            
            sz = size(A);
            x = ImplicitMatrix(sz,mtimes_right,mtimes_left);
        end
    end
end