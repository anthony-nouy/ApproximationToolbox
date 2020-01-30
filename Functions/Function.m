% Class Function

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

classdef Function

    properties
        dim
        measure
        outputSize = [1,1]
        evaluationAtMultiplePoints = true
        store = false
    end
    
    properties (Hidden)
        xStored = []
        yStored = []
    end
    
    methods
        function [y,f] = subsref(f,s)
            % y = subsref(f,s)
            % Remplaces the built-in subsref
            % See also subsref
            
            if length(s) == 1 && strcmpi(s.type,'()')
                if f.store
                    [y,f] = f.storeEval(s.subs{:});
                else
                    y = f.eval(s.subs{:});
                end
            else
                y = builtin('subsref',f,s);
            end
        end
        
        function [y,f] = storeEval(f,x)
            if f.store && ~isempty(f.yStored)
                [I,J] = ismember(x,f.xStored,'rows');
                if prod(f.outputSize)~=1
                    f.yStored = f.yStored(:,:);
                end
                y = zeros(size(x,1),size(f.yStored,2));
                y(I,:) = f.yStored(J(I),:);
                xNew = x(~I,:);
                if ~isempty(xNew)
                    yNew = f.eval(xNew);
                    y(~I,:) = yNew;
                    f.xStored = [f.xStored;xNew];
                    f.yStored = [f.yStored;yNew];
                end
                y = reshape(y,[size(y,1),f.outputSize]);
                if prod(f.outputSize)~=1
                    f.yStored = reshape(f.yStored,[size(y,1),f.outputSize]);
                end
            else
                y = f.eval(x);
                f.store = true;
                f.xStored = x;
                f.yStored = y;
            end
        end
        
        function fplot(f,varargin)
            % fplot(f,varargin)
            % Replaces the built-in function fplot
            % See also fplot
            if nargin==1 && ~isempty(f.measure)
                varargin = {truncatedSupport(f.measure)};
            end
            g = @(x) f.eval(x.').';
            fplot(g,varargin{:})
        end
        
        function sz = size(f)
            sz = f.outputSize;
        end
        
        function varargout = surf(f,n,varargin)
            % h = surf(f,n,varargin)
            % f: Function
            % n: integer or array 1-by-2 giving the number of points in each dimension (1000 by default)
            if isempty(f.measure)
                error('property measure is empty.')
            end
            
            s = truncatedSupport(f.measure);
            
            if ndims(f)>2
                error('The function should be a bivariate function, use partial evaluation for higher-dimensional function.')
            end
            
            if nargin==1 || isempty(n)
                n=[1000,1000];
            elseif length(n)==1
                n = [n,n];
            end
            
            grids = cell(1,2);
            grids{1} = linspace(s{1}(1),s{1}(2),n(1))';
            grids{2} = linspace(s{2}(1),s{2}(2),n(2))';
            grids{1} = grids{1}(2:end-1);
            grids{2} = grids{2}(2:end-1);
            
            grid = FullTensorGrid(grids);
            fg = evalOnTensorGrid(f,grid);
            varargout = cell(1,nargout);
            
            [varargout{:}] = surf(grids{1},grids{2},fg.data,varargin{:});
        end
        
        function falpha = partialEvaluation(f,notalpha,xnotalpha)
            % falpha = partialEvaluation(f,notalpha,xnotalpha)
            % partial evaluation of a function f(x) = f(xalpha,xnotalpha)
            % returns a function falpha(.) = f(.,xnotalpha) for fixed
            % values xnotalpha of variables with indices notalpha
            % alpha = tuple of integers
            % xnotalpha: tuple of values of length length(alpha)
            % f: Function
            % falpha: UserDefinedFunction
            
            d = f.dim;
            if isempty(d)
                error('Function has empty property dim.')
            end
            
            alpha = setdiff(1:d,notalpha);
            [~,I]=ismember(1:d,[alpha,notalpha]);
            s.type = '()';
            s.subs = {':',I};
            falpha = UserDefinedFunction(@(xalpha) f.eval(subsref(array(FullTensorGrid({xalpha,xnotalpha})),s)),length(alpha));
            falpha.store = f.store;
            falpha.evaluationAtMultiplePoints = f.evaluationAtMultiplePoints;
            falpha.measure = marginal(f.measure,alpha);
        end
        
        function [fx,x] = random(f,n,measure)
            % [fx,x] = random(f,n,measure)
            % Evaluates the function f at n points x drawn randomly
            % according to the ProbabilityMeasure in measure if provided,
            % or in f.measure.
            % h: FunctionalBasis
            % n: integer
            % measure: ProbabilityMeasure (optional)
            % fx: n-by-f.outputSize array of doubles
            % x: n-by-f.dim array of doubles
            
            if nargin == 1
                n = 1;
            end
            if nargin <= 2
                if isa(f.measure,'ProbabilityMeasure')
                    measure = f.measure;
                else
                    error('Must provide a ProbabilityMeasure.')
                end
            end
            if nargin == 3 && ~isa(measure,'ProbabilityMeasure')
                error('Must provide a ProbabilityMeasure.')
            end
            
            x = random(measure,n);
            fx = f.eval(x);
        end
        
        function fx = evalOnTensorGrid(f,x)
            % fx = evalOnTensorGrid(f,x)
            % Evaluation of f on grid x
            % f: Function
            % x: TensorGrid
            % fx: AlgebraicTensor
            % if x is a FullTensorGrid, fx is a FullTensor
            % if x is a SparseTensorGrid, fx is a SparseTensor
            
            xa = array(x);
            fx = f.eval(xa);
            if isa(x,'SparseTensorGrid')
                if all(f.outputSize==1)
                    fx = SparseTensor(fx,x.indices,x.sz);
                else
                    error('Method not implemented.')
                end
            elseif isa(x,'FullTensorGrid')
                if all(f.outputSize==1)
                    sz = x.sz;
                    if f.dim>1
                        fx = reshape(fx,sz);
                    end
                else
                    sz = [x.sz,f.outputSize];
                    fx = reshape(fx,sz);
                end
                fx = FullTensor(fx,numel(sz),sz);
            else
                error('Second argument must be a TensorGrid.')
            end
        end
        
        
        function [errL2,errLinf] = testError(f,g,n,X)
            % [errL2,errLinf] = testError(f,g,numberOfSamples)
            % [errL2,errLinf] = testError(f,g,numberOfSamples,Measure)
            % [errL2,errLinf] = testError(f,gtest,xtest)
            
            if ~isa(f,'Function')
                [errL2,errLinf] = testError(g,f,n,X);
            elseif nargin==2
                [errL2,errLinf] = testError(f,g,1000);
            elseif nargin==4
                xtest = random(X,n);
                gxtest  = g.eval(xtest);
                [errL2,errLinf] = testError(f,gxtest,xtest);
            elseif nargin==3 && isa(g,'Function') && length(n)==1
                if ~isempty(f.measure)
                    X = f.measure;
                else
                    X = g.measure;
                end
                [errL2,errLinf] = testError(f,g,n,X);
            elseif nargin==3 && isnumeric(n)
                xtest = n;
                fxtest = f.eval(xtest);
                if isa(g,'Function')
                    gxtest = g.eval(xtest);
                else
                    assert(size(g,1)==size(n,1),'Number of evaluations does not match the number of points.')
                    gxtest = g;
                end
                errL2 = norm(fxtest(:,:)-gxtest(:,:),'fro')/norm(gxtest(:,:),'fro');
                errLinf = norm(sqrt(sum((fxtest(:,:)-gxtest(:,:)).^2,2)),'inf')/norm(sqrt(sum((gxtest(:,:).^2),2)),'inf');
            end
        end
    end
    
    methods (Abstract)
        y = eval(f,x);
    end
end