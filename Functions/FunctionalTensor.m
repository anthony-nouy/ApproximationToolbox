% Class FunctionalTensor

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

classdef FunctionalTensor < Function
    
    properties
        tensor
        bases
        fdims
        evaluatedBases = false
    end
    
    methods
        function f = FunctionalTensor(tensor,bases,fdims)
            % x = FunctionalTensor(tensor,bases,fdims)
            % tensor: object of type Tensor
            % bases: FullTensorProductFunctionalBasis
            % fdims: dimensions corresponding to bases
            
            if nargin==0
                tensor=[];
                bases=[];
                fdims=[];
            end
            
            if isa(tensor,'FunctionalTensor')
                tensor = tensor.tensor;
            end
            
            f.tensor = tensor;
            
            if ~isa(bases,'FunctionalBases') && ~iscell(bases)
                error('Must provide a FunctionalBases object, or a cell of bases evaluations.')
            end
            f.bases = bases;
            if nargin==2
                if tensor.order~=length(bases)
                    error('Bases must contain as many bases as the order of the tensor, with possible empty elements.')
                end
                fdims = 1:tensor.order;
            elseif nargin==3
                if numel(fdims)~=length(f.bases)
                    error('The number of functional dimensions must correspond to the number of bases in bases.')
                end
            end
            f.fdims = fdims;
            if iscell(bases)
                f.evaluatedBases = true;
            else
                f.measure = f.bases.measure;
                f.dim = sum(ndims(f.bases));
            end
        end
        
        function x = plus(x,y)
            if isa(x,'FunctionalTensor') && isa(y,'FunctionalTensor') && x.bases==y.bases
                x.tensor = x.tensor + y.tensor;
            else
                error('not implemented')
            end
        end
        
        function x = israndom(f)
            % x = israndom(f)
            % Determines if input f is random
            % f: FunctionalTensor
            % x: boolean
            
            if isa(f.bases.measure,'ProbabilityMeasure')
                x = 1;
            else
                x = 0;
            end
        end
        
        function m = mean(f,varargin)
            % m = mean(f,X)
            % Computes the expectation of the random variable f(X).
            % If X is not provided, uses
            % random variables associated with bases of f.
            %
            % f: FunctionalTensor
            % X: RandomVector (optional)
            % m: double or tensor if f is a tensor-valued function
            
            H = mean(f.bases,[],varargin{:});
            m = timesVector(f.tensor,H,f.fdims);
        end
        
        function m = expectation(f,varargin)
            % m = expectation(f,X)
            % Same as m = mean(f,X)
            %
            % See also FunctionalTensor/mean
            
            m = mean(f,varargin{:});
        end
        
        function v = variance(f,varargin)
            % v = variance(f,X)
            % Computes the variance of the random variable f(X).
            % If X is not provided, uses
            % random variables associated with bases of f.
            % f: FunctionalTensor
            % X: RandomVector (optional)
            % v: double or tensor if f is a tensor-valued function
            
            m = expectation(f,varargin{:});
            if numel(m)==1
                v = dotProductExpectation(f,f,[],varargin{:});
                v = v - m.^2;
            else
                error('not implemented')
            end
        end
        
        function s = std(f,varargin)
            % s = std(f,X)
            % Computes the standard deviation of the random variable f(X).
            % If X is not provided, uses
            % random variables associated with bases of f.
            % f: FuncFunctionalTensor
            % X: RandomVector (optional)
            % s: double or tensor if f is a tensor-valued function
            
            s = sqrt(variance(f,varargin{:}));
        end
        
        function c = dotProductExpectation(f,g,fdims,varargin)
            % c = dotProductExpectation(f,g)
            % Computes the expectation of f(X)g(X) where X is the random
            % vector associated with bases of f
            %
            % c = dotProductExpectation(f,g,[],X)
            % Computes the expectation of f(X)g(X) for the provided
            % RandomVector X
            %
            % c = dotProductExpectation(f,g,Xdims,X)
            % For tensor-valued functions of X (numel(X)<f.order), specify
            % the dimensions of f and g corresponding to RandomVector X
            %
            % f,g: FunctionalTensor
            % Xdims: D-by-1 or 1-by-D double (optional, 1:f.order by default)
            % X: RandomVector (optional)
            % c: double or tensor if f is a tensor-valued function
            
            if (nargin == 2 || isempty(fdims)) && f.tensor.order == g.tensor.order
                fdims = 1:f.tensor.order;
            elseif (nargin == 2 || isempty(fdims)) && f.tensor.order ~= g.tensor.order
                error('Tensors u and v do not have the same order, must specify dimensions')
            end
            
            if f.bases == g.bases
                M = gramMatrix(f.bases,fdims,varargin{:});
            else
                error('Not implemented');
            end
            
            f.tensor = timesMatrix(f.tensor,M,fdims);
            c = dot(f.tensor,g.tensor);
        end
        
        function n = norm(f,varargin)
            % n = norm(f,X)
            % Computes the L^2 norm of f(X).
            % If X is not provided, uses
            % random variables associated with bases of f.
            %
            % If f.evaluatedBases is true, without additional information,
            % return the canonical norm of f.tensor.
            %
            % f: FunctionalTensor
            % X: RandomVector (optional)
            % n: double or tensor if f is a tensor-valued function
            
            if ~f.evaluatedBases
                M = gramMatrix(f.bases,1:f.tensor.order,varargin{:});
            else
                M = cellfun(@(x) eye(size(x,2)), f.bases, 'UniformOutput', false);
            end
            g = f;
            f.tensor = timesMatrix(f.tensor,M,1:f.tensor.order);
            n = sqrt(dot(f.tensor,g.tensor));
        end
        
        function f = conditionalExpectation(f,dims,varargin)
            % y = conditionalExpectation(f,dims,XdimsC)
            % Computes the conditional expectation of f with respect to
            % the random variables dims (a subset of 1:d). The expectation
            % with respect to other variables (in the complementary set of
            % dims) is taken with respect the probability measure given by RandomVector XdimsC
            % if provided, or with respect the probability measure
            % associated with the corresponding bases of f.
            % f: FunctionalTensor
            % dims: D-by-1 or 1-by-D double
            %      or 1-by-d logical
            % rvC: RandomVector containing (d-D) RandomVariable (optional)
            % y: FunctionalTensor of order (d-D)
            if isa(dims,'logical')
                dims = find(dims);
            end
            d = f.tensor.order;
            if isempty(dims)
                f = expectation(f,varargin{:});
                return
            end
            
            dims = sort(dims);
            if length(f.fdims)~=d && ~all(f.fdims,1:d)
                error('not implemented for fdims different from 1:d')
            end
            dimsC = setdiff(1:length(f.bases),dims);
            if isempty(dimsC)
                return
            end
            H = mean(f.bases,dimsC,varargin{:});
            t = timesVector(f.tensor,H,dimsC);
            
            bases = keepBases(f.bases,dims);
            if isa(bases,'FunctionalBasesWithMappedVariables')
                warning('under developement')
                bases = keepMapping(bases,dims);
            end
            mes = f.measure;
            f = FunctionalTensor(t,bases);
            if ~isempty(mes)
                f.measure = marginal(mes,dims);
            end
        end
        
        function v = varianceConditionalExpectation(f,alpha)
            % v = varianceConditionalExpectation(f,alpha)
            % Computes the variance of the conditional expectation of f in dimensions in dims
            % f: FunctionalTensor
            % alpha: n-by-D double, where D is equal to the number of random variables
            %    or n-by-d logical
            % v: n-by-1 double
            
            m = expectation(f);
            v = zeros(size(alpha,1),1);
            for i = 1:size(alpha,1)
                u = alpha(i,:);
                if isa(u,'logical')
                    u = find(u);
                end
                if isempty(u)
                    v(i)=0;
                else
                    mu = conditionalExpectation(f,u);
                    v(i) = dotProductExpectation(mu,mu) - m.^2;
                end
            end
        end
        
        function y = eval(f,x,varargin)
            % y = eval(f,x)
            % Computes evaluations of f at points x
            % f: FunctionalTensor
            % x: array of size N-by-d or cell of length d
            %
            % y = eval(f,x,dims)
            % Computes partial evaluations of f at points x in dimensions in dims
            % dims contains the dimensions to evaluate
            % x are the coordinates of the variables associated with
            % dimensions dims
            % f: FunctionalTensor
            % dims: 1-by-n double
            % x: array of size N-by-n or cell of length n
            
            if f.evaluatedBases
                H = f.bases;
            elseif nargin >= 2
                H = eval(f.bases,x,varargin{:});
            else
                error('Must provide the evaluation points or the bases evaluations.')
            end
            y = evalWithBasesEvals(f,H,varargin{:});
        end
        
        function h = times(f,g)
            if isa(f,'FunctionalTensor') && isa(g,'FunctionalTensor')
                b = kron(f.bases,g.bases);
                t = kron(f.tensor,g.tensor);
                h = FunctionalTensor(t,b);
                if isa(h.tensor,'TreeBasedTensor') && h.tensor.ranks(h.tensor.tree.root)>1
                    if f.tensor.ranks(f.tensor.tree.root) ~=  g.tensor.ranks(g.tensor.tree.root)
                        error('wrong sizes')
                    else
                        c = h.tensor.tensors{h.tensor.tree.root};
                        n = f.tensor.ranks(f.tensor.tree.root);
                        s = cell(1,c.order);
                        s(:)={':'};
                        s{end}=(1:n:n^2);
                        c = subTensor(c,s{:});
                        h.tensor.tensors{h.tensor.tree.root}=c;
                        h.tensor.ranks(h.tensor.tree.root) = n;
                    end
                end
            elseif isa(f,'double')
                h = g;
                h.tensor = times(f,g.tensor);
                
            elseif isa(g,'double')
                h = f;
                h.tensor = times(f.tensor,g);
            else
                error('not implemented')
            end
        end
        
        function h = mtimes(f,g)
            if isa(f,'double')
                h=g;
                h.tensor = mtimes(f,g.tensor);
            elseif isa(g,'double')
                h=f;
                h.tensor = mtimes(f.tensor,g);
            else
                error('Method not implemented.')
            end
        end
        
        function g = parameterGradientEval(f,mu,x,varargin)
            if f.evaluatedBases
                H = f.bases;
            elseif nargin >= 3
                H = eval(f.bases,x);
            else
                error('Must provide the evaluation points or the bases evaluations.')
            end
            
            dims = 1:f.tensor.order;
            if isa(f.tensor,'TreeBasedTensor')
                % Compute fH, the TimesMatrixEvalDiag of f with H in all  
                % the dimensions except the ones associated with mu (if mu 
                % is a leaf node) or with the inactive children of mu (if
                % mu is an internal node). The tensor fH is used to compute 
                % the gradient of f with respect to f.tensor.tensors{mu}.
                t = f.tensor.tree;
                if t.isLeaf(mu)
                    dims(t.dim2ind == mu) = [];
                else
                    ch = nonzeros(t.children(:,mu));
                    ind = intersect(t.dim2ind,ch(~f.tensor.isActiveNode(ch)));
                    dims(ismember(t.dim2ind,ind)) = [];
                end
                
                if all(f.tensor.isActiveNode)
                    fH = timesMatrix(f.tensor,H(dims),dims);
                else
                    remainingDims = 1:f.tensor.order;
                    tensors = f.tensor.tensors;
                    dim2ind = t.dim2ind;

                    for leaf = intersect(t.dim2ind(dims), f.tensor.activeNodes)
                        dims = setdiff(dims, find(t.dim2ind == leaf));
                        tensors{leaf} = timesMatrix(f.tensor.tensors{leaf}, ...
                                H(t.dim2ind == leaf), 1);
                    end

                    for pa = unique(t.parent(setdiff(t.dim2ind(dims), ...
                            f.tensor.activeNodes)))
                        ind = intersect(t.dim2ind(dims), t.children(:, pa));
                        ind = setdiff(ind, f.tensor.activeNodes);
                        [~, dimsLoc] = ismember(ind, t.dim2ind);
                        if length(ind) > 1
                            tensors{pa} = timesMatrixEvalDiag(f.tensor.tensors{pa}, ...
                                H(dimsLoc), t.childNumber(ind));
                            remainingDims = setdiff(remainingDims, dimsLoc(2:end));
                            if all(~f.tensor.isActiveNode(nonzeros(t.children(:, pa))))
                                dim2ind(dimsLoc(1)) = t.parent(t.dim2ind(dimsLoc(1)));
                            else
                                dims = setdiff(dims, dimsLoc(1));
                            end
                            dim2ind(dimsLoc(2:end)) = 0;
                            perm = [t.childNumber(ind(1)), ...
                                setdiff(1:tensors{pa}.order, t.childNumber(ind(1)))];
                            tensors{pa} = ipermute(tensors{pa}, perm);
                        elseif length(ind) == 1
                            dims(dims == dimsLoc) = [];
                            tensors{pa} = timesMatrix(f.tensor.tensors{pa}, ...
                                H(dimsLoc), t.childNumber(ind));
                            dim2ind(dimsLoc) = t.dim2ind(dimsLoc);
                        end
                    end

                    keepind = fastSetdiff(1:t.nbNodes, t.dim2ind(dims));
                    a = t.adjacencyMatrix(keepind,keepind);
                    dim2ind = nonzeros(dim2ind).';

                    ind = setdiff(1:t.nbNodes, keepind);
                    I = zeros(1,t.nbNodes);
                    I(ind) = 1;
                    I = cumsum(I);
                    dim2ind = dim2ind - I(dim2ind);
                    mu = mu - I(mu);

                    t = DimensionTree(dim2ind,a);
                    fH = TreeBasedTensor(tensors(keepind),t);
                    fH = removeUniqueChildren(fH);
                    H = H(remainingDims);
                end
            else
                if mu <= f.tensor.order
                    dims(mu) = [];
                end
                fH = timesMatrix(f.tensor,H(dims),dims);
            end
            
            g = parameterGradientEvalDiag(fH, mu, H);
            if isa(f.tensor,'TreeBasedTensor') && ~t.isLeaf(mu)
                % If the order of the children has been modified in g, 
                % compute the inverse permutation.
                ch = nonzeros(t.children(:,mu));
                [~,I] = sort([ch(fH.isActiveNode(ch)) ; ch(~fH.isActiveNode(ch))]);
                J = []; 
                if mu ~= t.root 
                    J = fH.tensors{mu}.order+1; 
                end
                K = []; 
                if mu ~= t.root && f.tensor.ranks(t.root) > 1
                    K = g.order; 
                end
                g = permute(g,[1 ; I+1 ; J ; K]);
            end
        end
        
        function y = evalDerivative(f,n,x,varargin)
            % y = evalDerivative(f,n,x)
            % Computes evaluations of the n-derivative of f at points x
            % f: FunctionalTensor
            % n: 1-by-d array of integers
            % x: array of size N-by-d or cell of length d
            %
            % y = eval(f,n,x,dims)
            % Computes partial evaluations of the n-derivative of f at
            % points x in dimensions in dims
            % dims contains the dimensions to evaluate
            % x are the coordinates of the variables associated with
            % dimensions dims
            % f: FunctionalTensor
            % n: 1-by-length(dims) array of integers
            % dims: 1-by-m double
            % x: array of size N-by-m or cell of length m
            
            H = evalDerivative(f.bases,n,x,varargin{:});
            y = evalWithBasesEvals(f,H,varargin{:});
        end
        
        function df = derivative(f,n)
            % df = derivative(f,n)
            % Computes the n-derivative of f
            % f: FunctionalTensor
            % n: 1-by-d array of integers
            % df: FunctionalTensor
            
            df = f;
            df.bases = derivative(f.bases,n);
        end
        
%         function [H,x] = basesEval(f,x,dims)
%             % function [H,x] = basesEval(f,x,dims)
%             if nargin==2
%                 dims = f.fdims;
%             end
%             H = cell(numel(dims),1);
%             if ~isa(x,'cell')
%                 x = mat2cell(x,size(x,1),ones(1,size(x,2)));
%             end
%             for i=1:numel(dims)
%                 H{i} = eval(f.bases{dims(i)},x{i});
%             end
%         end
        
        function y = evalOnGrid(f,x,dims)
            % y = evalOnGrid(f,x)
            % Computes evaluations of f at points x
            % f: FunctionalTensor
            % x: cell array such that x{k} contains the grid associated
            % with variable x_k
            %
            % y = evalOnGrid(f,x,dims)
            % Computes evaluations of f at points x in dimensions in dims
            % dims indicates the dimensions corresponding to the cell array x
            % f: FunctionalTensor
            
            if nargin==2
                dims = 1:number(f.bases);
            end
            H = eval(f.bases,x,dims);
            
            if numel(dims)==f.tensor.order
                y=timesMatrix(f.tensor,H,dims);
            else
                f.tensor=timesMatrix(f.tensor,H,f.fdims(dims));
                f.bases = removeBases(f.bases,dims);
                f.fdims(dims) = [];
                if isempty(f.fdims)
                    y = f.tensor;
                else
                    y = f;
                end
            end
        end
        
%         function [H,x] = basesRandom(f,varargin)
%             % function [H,x] = basesRandom(f)
%
%             dims = f.fdims;
%             [H,x] = basesRandomDims(f,dims,varargin{:});
%         end
%
%         function [H,x] = basesRandomDims(f,dims,varargin)
%             % function [H,x] = basesRandomDims(f,dims)
%
%             H = cell(numel(dims),1);
%             x = cell(numel(dims),1);
%             for i=1:numel(dims)
%                 [H{i},x{i}] = random(f.bases{dims(i)},varargin{:});
%             end
%
%         end
        
        function [y,x] = random(f,varargin)
            % [y,x] = random(f)
            % Returns y=f(x) for a random sample of x
            %
            % [y,x] = random(f,N)
            % Returns y=f(x) for N random samples of x
            
            [y,x] = randomDims(f,1:length(f.fdims),varargin{:});
        end
        
        function [y,x] = randomDims(f,dims,varargin)
            % [y,x] = randomDims(f,dims)
            % Returns y=f(x) for a random sample of variables x_k with k in dims
            % x is a cell array of size numel(dims) containing the sample of x
            %
            % [y,x] = randomDims(f,dims,N)
            % Returns y=f(x) for N random samples of variables x_k with k in dims
            % x is a cell array of size numel(dims) containing the samples of x
            
            [H,x] = randomDims(f.bases,dims,varargin{:});
            y = evalWithBasesEvals(f,H,dims);
        end
        
        function rv = getRandomVector(f)
            % rv = getRandomVector(f)
            % Gets the random vector rv associated with the basis functions of f
            % f: FunctionalTensor
            % rv: RandomVector
            
            rv = getRandomVector(f.bases);
        end
        
%         function f = updateProperties(f)
%             isfdims = cellfun(@(c) isa(c,'FunctionalBasis'),f.bases);
%             f.fdims = find(isfdims);
%         end
        
        
        function y = evalWithBasesEvals(f,H,dims)
            % y = evalWithBasesEvals(f,H,dims)
            
            if nargin==2
                dims=1:length(f.bases);
            end
            if length(dims) == 1 && ~iscell(H)
                H = {H};
            end
            if numel(dims)==f.tensor.order
                y = timesMatrixEvalDiag(f.tensor,H);
            else
                y = f;
                fdimseval=y.fdims(dims);
                y.tensor = timesMatrixEvalDiag(y.tensor,H,fdimseval);
                
                fdimseval=sort(fdimseval);
                olddims=1:f.tensor.order;
                olddims(fdimseval(2:end))=[];
                if size(y.tensor,fdimseval(1))==1
                    y.tensor = squeeze(y.tensor,fdimseval(1));
                    olddims(fdimseval(1))=[];
                end
                y.bases = removeBases(y.bases,dims);
                y.fdims(dims)=[];
                [~,y.fdims]=ismember(y.fdims,olddims);
                if isempty(y.fdims)
                    y=y.tensor;
                    if y.order==1
                        y = double(y);
                    end
                end
            end
        end
        
        function n = storage(f)
            n = storage(f.tensor);
        end
    end
end