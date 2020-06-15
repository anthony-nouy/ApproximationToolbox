% Class MultiIndices

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

classdef MultiIndices
    
    properties
        array
    end
    
    methods
        function I = MultiIndices(array)
            % I = MultiIndices(array)
            % Constructor of MultiIndices
            % array: array of size nxd containing n multi-indices in N^d
            
            if nargin==0
                I.array = [];
            elseif nargin==1 && isa(array,'MultiIndices')
                I.array = array.array;
            else
                I.array = array;
            end
        end
        
        function ind = sub2ind(I,sz)
            % ind = sub2ind(I,sz)
            % I: MultiIndices with nxd array [I1,I2,...,Id]
            % ind = sub2ind(sz,I1,...,Id)
            
            I = cell(I);
            ind = sub2ind(sz,I{:});
        end
        
        function ok = eq(I,J)
            % ok = eq(I,J)
            % Checks if the two objects I and J are identical
            % I,J: MultIndices
            % ok: boolean
            
            if ~isa(J,'MultiIndices')
                ok = 0;
            else
                if cardinal(J)==1
                    J.array = repmat(J.array,cardinal(I),1);
                end
                ok = all(I.array == J.array,2);
            end
        end
        
        function s = le(I,J)
            % ok = le(I,J)
            % checks if I<=J
            % I: MultiIndices
            % J: MultiIndices containing 1 element or cardinal(I) elements
            % ok: boolean array of size cardinal(I)-by-1
            
            if ~isa(J,'MultiIndices')
                error('Must provide a MultiIndices.')
            end
            if cardinal(J)==1
                J.array = repmat(J.array,cardinal(I),1);
            end
            s = all(I.array <= J.array,2);
        end
        
        function n = numel(I)
            warning('numel is replaced by cardinal and will be removed in a future release.')
            n = cardinal(I);
        end
        
        function n = cardinal(I)
            n = size(I.array,1);
        end
        
        function c = cell(I)
            c = mat2cell(I.array,size(I.array,1),ones(1,size(I.array,2)));
        end
        
        function d = ndims(I)
            d = size(I.array,2);
        end
        
        function n = norm(I,p,k)
            % n = norm(I,p,k)
            % Computes the p-norm of multi-indices k in I
            % I: MultIndices
            % p: positive real scalar or Inf, p=2 by default
            % k: numbers of multi-indices in I, 1:cardinal(I) by default
            % n: array of length length(k)
            
            if nargin<3
                k=1:cardinal(I);
            end
            if nargin<2
                p=2;
            end
            if p==Inf
                n = max(I.array(k,:),[],2);
            else
                n = sum(I.array(k,:).^p,2).^(1/p);
            end
        end
        
        function n = weightedNorm(I,p,w,k)
            % n = norm(I,p,w,k)
            % Computes the weighted p-norm of multi-indices k in I
            % I: MultIndices
            % p: positive real scalar or Inf
            % w: array of length d=ndims(I) containing the weights
            % k: numbers of multi-indices in I, 1:cardinal(I) by default
            % n: array of length length(k)
            
            if nargin<4
                k=1:cardinal(I);
            end
            if p==Inf
                n = max(I.array(k,:).*repmat(w(:)',length(k),1),[],2);
            else
                n = sum((I.array(k,:).*repmat(w(:)',length(k),1)).^p,2).^(1/p);
            end
        end
        
        function I = sortByNorm(I,p,varargin)
            % I = sortByNorm(I,p,mode)
            % Sorts multi-indices by increasing or decreasing p-norm
            % I: MultIndices
            % p: positive real scalar or Inf
            % mode: 'ascend' or 'descend', 'ascend' by default
            
            n = norm(I,p);
            [~,j] = sort(n,varargin{:});
            I.array = I.array(j,:);
        end
        
        function I = sortByWeightedNorm(I,p,w,varargin)
            % I = sortByWeightedNorm(I,p,w,mode)
            % Sorts multi-indices by increasing or decreasing weighted p-norm
            % I: MultIndices
            % p: positive real scalar or Inf
            % w: array of length d=ndims(I) containing the weights
            % mode: 'ascend' or 'descend', 'ascend' by default
            
            n = weightedNorm(I,p,w);
            [~,j] = sort(n,varargin{:});
            I.array = I.array(j,:);
        end
        
        function I = plus(I,m)
            I.array = I.array+m;
        end
        
        function I = minus(I,m)
            I.array = I.array-m;
        end
        
        function I = sort(I,varargin)
            % I = sort(I)
            % Sorts multi-indices using sortrows(I.array,d:-1:1)
            % I: MultIndices
            %
            % I = sort(I,varargin)
            % Sorts multi-indices using sortrows with arguments varargin
            
            if nargin==1
                I.array = sortrows(I.array,size(I.array,2):-1:1);
            else
                I.array = sortrows(I.array,varargin{:});
            end
        end
        
        function I = addIndices(I,J,varargin)
            % I = addIndices(I,J)
            % Union of multi-indices of I and J
            % I,J: MultiIndices
            
            I.array = unique([I.array;J.array],'rows','stable');
        end
        
        function I = removeIndices(I,J,varargin)
            % I = removeIndices(I,J)
            % Removes multi-indices J from I
            % I: MultiIndices
            % J: MultiIndices or array containing the numbers of multi-indices to remove
            
            if isa(J,'MultiIndices')
                I.array = setdiff(I.array,J.array,'rows','stable');
            else
                I.array(J,:) = [];
            end
        end
        
        function I = removeDims(I,d)
            % I = removeDims(I,d)
            % Removes the dimensions d in I
            % I: MultiIndices
            % d: array containing the dimensions to remove
            
            I.array(:,d) = [];
            I.array = unique(I.array,'rows','stable');
        end
        
        function [I,iI,iJ] = intersectIndices(I,J,varargin)
            % [K,iI,iJ] = intersectIndices(I,J)
            % Intesection of multi-indices of I and J
            % I,J,K: MultiIndices
            % iI,iJ: index vectors such that K = I(iI,:) and K = J(iJ,:)
            
            [I.array,iI,iJ] = intersect(I.array,J.array,'rows','stable');
        end
        
        function I = keepIndices(I,k)
            % I = keepIndices(I,k)
            % Keeps the multi-indices k in I
            % I: MultIndices
            % k: array containing the numbers of multi-indices to keep
            
            I.array = I.array(k,:);
        end
        
        function I = keepDims(I,d)
            % I = keepDims(I,d)
            % Keeps the dimensions d in I
            % I: MultIndices
            % d: array containing the dimensions to keep
            
            I.array = I.array(:,d);
            I.array = unique(I.array,'rows','stable');
        end
        
        function I = getIndices(I,k)
            % ind = getIndices(I,k)
            % Gets the multi-indices k in I
            % I: MultiIndices with nxd array
            % k: array containing the numbers of multi-indices to select
            % ind: array of size length(k)xd
            
            I = I.array(k,:);
        end
        
        function r = isDownwardClosed(I,varargin)
            % r = isDownwardClosed(I)
            % Checks whether or not the multi-index set is downward closed  (or lower or monotone)
            % I: MultIndices
            % r: boolean
            %
            % r = isDownwardClosed(I,m)
            % m: lowest index for all dimensions, m=0 by default
            
            r = true;
            indtest = 1:cardinal(I);
            while ~isempty(indtest)
                p = I.array(indtest(end),:);
                Ip = MultiIndices.boundedBy(p,varargin{:});
                [ok,rep] = ismember(Ip.array,I.array,'rows');
                if ~all(ok)
                    r = false;
                    return
                end
                indtest = setdiff(indtest,rep);
            end
        end
        
        function env = envelope(I,u)
            % e = envelope(I,u)
            % Computes the monotone envelope (or monotone majorant) e of a bounded sequence u
            % I: MultiIndices with nxd array
            % u: array of length n containing the bounded sequence
            % e: array of length n containing the monotone envelope
            % corresponding to the sequence defined by
            % e_i = \max_{j\ge i} |u_j|
            
            ind = I.array;
            n = cardinal(I);
            
            if length(u)~=n
                error('The length of the sequence does not coincide with the number of multi-indices.')
            end
            
            env = u;
            for i = 1:n
                ind_sup = all(ind>=repmat(ind(i,:),[n 1]),2);
                env(i) = max(abs(u(ind_sup)));
            end
        end
        
        function Imax = getMaximalIndices(I)
            % Imax = getMaximalIndices(I)
            % Returns the set of maximal multi-indices contained in the downward closed multi-index set I
            % I,Imax: MultiIndices
            
            d = ndims(I);
            n = cardinal(I);
            neighbours = repmat(permute(I.array,[1,3,2]),[1,d,1]) + repmat(permute(eye(d),[3,1,2]),[n,1,1]);
            neighbours = reshape(neighbours,n*d,d);
            [ok,~] = ismember(neighbours,I.array,'rows');
            ok = reshape(ok,[n,d]);
            ind_max = I.array(~any(ok,2),:);
            Imax = MultiIndices(ind_max);
        end
        
        function Imarg = getMargin(I)
            % Imarg = getMargin(I)
            % Returns the margin of the multi-index set I defined by the set of multi-indices i not in I such that \exists k \in N^\ast s.t. i_k \neq 0 \implies i - e_k \in I where e_k is the k-th Kronecker sequence
            % I,Imarg: MultiIndices
            
            d = ndims(I);
            n = cardinal(I);
            neighbours = repmat(permute(I.array,[1,3,2]),[1,d,1]) + repmat(permute(eye(d),[3,1,2]),[n,1,1]);
            neighbours = reshape(neighbours,n*d,d);
            ind_marg = setdiff(neighbours,I.array,'rows');
            
            ind_marg = unique(ind_marg,'rows');
            Imarg = MultiIndices(ind_marg);
        end
        
        function Imarg = getReducedMargin(I)
            % Imarg = getReducedMargin(I)
            % Returns the reduced margin of the multi-index set I defined by the set of multi-indices i not in I such that \forall k \in N^\ast s.t. i_k \neq 0 \implies i - e_k \in I where e_k is the k-th Kronecker sequence
            % I,Imarg: MultiIndices
            
            Imarg = getMargin(I);
            d = ndims(I);
            
            neighbours = repmat(permute(Imarg.array,[1,3,2]),[1,d,1]) - repmat(permute(eye(d),[3,1,2]),[cardinal(Imarg),1,1]);
            
            n = size(neighbours,1);
            neighbours = reshape(neighbours,n*d,d);
            ok = ismember(neighbours,I.array,'rows');
            isout = any(neighbours<0,2);
            ok = ok | isout;
            ok = reshape(ok,[n,d]);
            keep = all(ok,2);
            ind_marg_red = Imarg.array(keep,:);
            
            Imarg = MultiIndices(ind_marg_red);
        end
        
        function plot(I,varargin)
            % plot(I)
            % Plots the multi-index set I
            % I: MultiIndices
            
            plotMultiIndices(I,varargin{:});
        end
    end
    
    methods (Static)
        function I = withBoundedNorm(d,p,m)
            % I = withBoundedNorm(d,p,m)
            % Creates the set of multi-indices in N^d with p-norm bounded by m, p>0
            % d: positive integer scalar
            % p: positive real scalar or Inf
            % m: positive real scalar
            % I: MultiIndices
            
            if p==Inf
                I = MultiIndices.productSet((0:m),d);
            elseif p==1
                I = MultiIndices(zeros(1,d));
                for i=1:m
                    I = I.addIndices(getMargin(I));
                end
            else
                I = MultiIndices(zeros(1,d));
                add = true;
                while add
                    N = getMargin(I);
                    n = norm(N,p);
                    k = find(n<=m);
                    if isempty(k)
                        add=false;
                    else
                        [~,j] = sort(n(k));
                        N = keepIndices(N,k(j));
                        I = I.addIndices(N);
                    end
                end
                
            end
            
        end
        
        function I = withBoundedWeightedNorm(d,p,m,w)
            % I = withBoundedWeightedNorm(d,p,m,w)
            % Creates the set of multi-indices in N^d with weighted p-norm bounded by m, p>0
            % d: positive integer scalar
            % p: positive real scalar or Inf
            % m: positive real scalar
            % w: array of length d containing the weights defining the weighted p-norm of a multi-index i
            % |i|_{p,w} = (\sum_{k=1}^d w_k^p i_k^p)^(1/p) for 0<p<inf
            % |i|_{Inf,w} = \max_k  w_k i_k
            % I: MultiIndices
            
            I = cell(1,d);
            for k=1:d
                I{k} = 0:floor(m/w(k));
            end
            I = MultiIndices.productSet(I);
            n = weightedNorm(I,p,w);
            k = find(n<=m);
            [~,j] = sort(n(k));
            I = keepIndices(I,k(j));
            
        end
        
        function I = boundedBy(m,m0)
            % I = boundedBy(m)
            % Creates the set of multi-indices bounded by m
            % m: array of length d containing the highest indices in each dimension
            % I: MultiIndices containing the product set (0:m(1))x...x(0:m(d))
            %
            % I = boundedBy(m,m0)
            % m0: lowest index for all dimensions, m0=0 by default
            % I: MultiIndices containing the product set (m0:m(1))x...x(m0:m(d))
            
            d = length(m);
            L = cell(1,d);
            if nargin==1
                m0 = 0;
            end
            for k=1:d
                L{k} = (m0:m(k))';
            end
            I = MultiIndices.productSet(L);
        end
        
        function I = productSet(L,d)
            % I = productSet(L)
            % Creates the set of multi-indices obtained by a product of sets of indices
            %
            % If L is a cell of length d containing arrays of integers of size m_kx1, 1<=k<=d, I is a MultiIndices containing the product set L{1}xL{2}x...xL{d}
            %
            % If L is an array of size nx1 containing a set of indices, I is a MultiIndices containing the product set LxLx...xL (d times)
            
            if nargin==1 && isa(L,'cell')
                d = length(L);
            elseif nargin==2 && isa(L,'double')
                L = repmat({L(:)},1,d);
            else
                error('Wrong argument.')
            end
            
            N = cellfun(@length,L);
            I = cell(1,d);
            [I{:}] = ind2sub(N,(1:prod(N))');
            for i=1:d
                L{i} = L{i}(:);
                I{i} = L{i}(I{i});
            end
            I = [I{:}];
            I = MultiIndices(I);
            
        end
        
        function I = ind2sub(sz,ind)
            % I = ind2sub(sz,ind)
            % Creates the set of multi-indices with array [I1,...,Id] such that [I1,...,Id] = ind2sub(sz,ind)
            % I: MultiIndices
            
            d = length(sz);
            I = cell(1,d);
            [I{:}] = ind2sub(sz,ind(:));
            I = [I{:}];
            I = MultiIndices(I);
        end
    end
end