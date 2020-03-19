% Class TreeBasedTensor: algebraic tensors in tree-based tensor format
%
% References:
% - Nouy, A. (2017). Low-rank methods for high-dimensional approximation
% and model order reduction. Model reduction and approximation, P. Benner,
% A. Cohen, M. Ohlberger, and K. Willcox, eds., SIAM, Philadelphia, PA, 171-226.
% - Falcó, A., Hackbusch, W., & Nouy, A. (2018). Tree-based tensor formats.
% SeMA Journal, 1-15
% - Grelier, E., Nouy, A., & Chevreuil, M. (2018). Learning with tree-based
% tensor formats. arXiv preprint arXiv:1811.04455
% - Nouy, A. (2019). Higher-order principal component analysis for the
% approximation of tensors in tree-based low-rank formats. Numerische
% Mathematik, 141(3), 743-789

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

classdef TreeBasedTensor < AlgebraicTensor
    
    properties
        % TENSORS - Parameters of the representation
        tensors
        % RANKS - Tree-based rank
        ranks
        % ORDER - Order of the tensor
        order
        % SZ - Size of the tensor
        sz
        % TREE - Dimension tree
        tree
        % ISORTH - True if the representation of the tensor is orthogonal
        isOrth = false
        % ISACTIVENODE - Logical array indicating if the nodes are active
        isActiveNode
        % ORTHNODE - Node with respect to which the representation is orthogonalized (0 is the root node)
        orthNode = 0
    end
    
    methods
        function x = TreeBasedTensor(C,varargin)
            % TREEBASEDTENSOR - Constructor for the class TreeBasedTensor
            %
            % x = TREEBASEDTENSOR(C,T)
            % C: cell of FullTensor (parameters associated with the nodes of the tree)
            % T: DimensionTree
            % x: TREEBASEDTENSOR
            
            if nargin==1 && isa(C,'TreeBasedTensor')
                x.tensors = C.tensors;
                x.ranks = C.ranks;
                x.sz = C.sz;
                x.order = C.order;
                x.tree = C.tree;
                x.isOrth = C.isOrth;
                x.isActiveNode = C.isActiveNode;
                x.orthNode = C.orthNode;
            elseif isa(C,'cell') && isa(varargin{1},'DimensionTree')
                x.tree = varargin{1};
                x.tensors = C;
                x.order = numel(x.tree.dim2ind);
                x = updateProperties(x);
            else
                error('Constructor not implemented.');
            end
        end
        
        function [ok,c] = isAdmissibleRank(x,varargin)
            % ISADMISSIBLERANK - Checks if a given tuple is an admissible tree-based rank
            %
            % ok = ISADMISSIBLERANK(x)
            % Checks if x.ranks is an admissible rank for the tree-based format
            %
            % function [ok,c] = ISADMISSIBLERANK(x,r)
            % Checks if the tuple r is an admissible rank for the tree-based format
            %
            % x: TreeBasedTensor
            % r: 1-by-n double of integers, with n the number of nodes of x.tree
            % ok: logical
            % c: 1-by-n cell of logical, with n the number of nodes of x.tree, each cell giving the admissibiliy of the children ranks
            %
            % Note: could be modified to take into account the ranks of inactive nodes
            
            T = x.tree;
            nodes = fastIntersect(T.internalNodes,x.activeNodes);
            c = cell(1,T.nbNodes);
            
            if nargin == 1
                r = x.ranks;
            else
                r = varargin{1};
            end
            
            if r(T.parent==0)~=1
                ok = false;
                warning('The root rank should be 1.')
                return
            else
                ok=true;
            end
            
            for i = nodes
                ch = nonzeros(T.children(:,i));
                
                if all(x.isActiveNode(ch))
                    c{i}(1) = (r(i) <= prod(r(ch)));
                    for mu=1:length(ch)
                        nomu = fastSetdiff(1:length(ch),mu);
                        c{i}(mu+1) = (r(ch(mu)) <= r(i)*prod(r(ch(nomu))));
                    end
                end
                ok = ok & all(c{i});
            end
            
            leafNodes = fastIntersect(find(T.isLeaf),x.activeNodes);
            for i = leafNodes
                c{i} = r(i) <= x.tensors{i}.sz(1);
                ok = ok & c{i};
            end
        end
        
        function n = storage(x)
            n = sum(cellfun(@numel,x.tensors));
        end
        
        function n = sparseStorage(x)
            n = sum(cellfun(@nnz,x.tensors));
        end
        
        function n = sparseLeavesStorage(x)
            n = sum(cellfun(@numel,x.tensors(x.tree.internalNodes)))+...
                sum(cellfun(@nnz,x.tensors(x.tree.dim2ind)));
        end
        
        function x = plus(x,y)
            t = x.tree;
            
            for nod=1:t.nbNodes
                if t.isLeaf(nod) && x.isActiveNode(nod)
                    x.tensors{nod} = FullTensor([x.tensors{nod}.data,y.tensors{nod}.data]);
                elseif ~t.isLeaf(nod)
                    ch = nonzeros(t.children(:,nod));
                    nonActiveLeafChildren = t.isLeaf(ch) & ~x.isActiveNode(ch);
                    if any(nonActiveLeafChildren)
                        otherDims = 1:x.tensors{nod}.order;
                        otherDims(nonActiveLeafChildren) = [];
                        x.tensors{nod} = cat(x.tensors{nod},y.tensors{nod},otherDims);
                    else
                        x.tensors{nod} = cat(x.tensors{nod},y.tensors{nod});
                    end
                    x.tensors{nod}.isOrth = false;
                    x.tensors{nod}.orthDim = [];
                end
                
            end
            x.isOrth = false;
            
            x = updateProperties(x);
        end
        
        function x = uminus(x)
            x.tensors{x.tree.root} = -x.tensors{x.tree.root};
        end
        
        function x = minus(x,y)
            x = x + (-y);
        end
        
        function x = subTensor(x,varargin)
            assert(nargin == 1+x.order, 'Wrong number of input arguments.');
            t = x.tree;
            for k = 1:x.order
                nod = t.dim2ind(k);
                if ~x.isActiveNode(nod)
                    i = t.parent(nod);
                    chNum = t.childNumber(nod);
                    x.tensors{i} = evalAtIndices(x.tensors{i},varargin{k},chNum);
                else
                    x.tensors{nod} = evalAtIndices(x.tensors{nod},varargin{k},1);
                end
            end
            x = updateProperties(x);
        end
        
        function xf = full(x)
            t = x.tree;
            maxLvl = max(t.level);
            xf = x.tensors;
            for l = maxLvl-1:-1:0
                nodLvl = setdiff(nodesWithLevel(t,l),t.dim2ind);
                for nod = nodLvl
                    chNod = nonzeros(t.children(:,nod));
                    dims = [];
                    for k = 1:length(chNod)
                        ch = chNod(k);
                        if x.isActiveNode(ch)
                            xf{nod} = timesTensor(xf{ch},xf{nod},...
                                xf{ch}.order,length(dims)+1);
                            xf{ch}=[];
                            dims = [t.dims{ch},dims];
                        else
                            dims = [dims,t.dims{ch}];
                        end
                    end
                    
                    t.dims{nod} = dims;
                    
                end
            end
            xf = xf{t.root};
            if xf.order>1
                xf = ipermute(xf,dims);
            end
        end
        
        function dT = computeDistances(x,alpha)
            % COMPUTEDISTANCES - Distances from all nodes of the tree to a given node
            %
            % dT = COMPUTEDISTANCES(x,alpha)
            % alpha: integer (node index)
            % dT: 1-by-length(x.tree.nbNodes) double such that dT(j) gives the distance between nodes i and j
            
            t = x.tree;
            r = x.ranks;
            r(~x.isActiveNode) = x.sz(~x.isActiveNode(t.dim2ind));
            
            dT = zeros(1,t.nbNodes);
            for beta = 1:t.nbNodes
                if ~(ismember(beta,t.ascendants(alpha)) || ...
                        ismember(beta,t.descendants(alpha)) || ...
                        t.parent(alpha) == t.parent(beta))
                    commonAs = fastIntersect(ascendants(t,alpha), ascendants(t,beta));
                    gamma = commonAs(t.level(commonAs) == max(t.level(commonAs)));
                    as = unique([t.ascendants(alpha), t.ascendants(beta)]);
                    dT(beta) = r(gamma)*prod(r(fastSetdiff(...
                        nonzeros(t.children(:,fastSetdiff(as,t.ascendants(gamma)))),as)));
                end
            end
        end
        
        function xStar = optimizeDimensionTree(x,tol,N)
            % OPTIMIZEDIMENSIONTREE - Optimization over the set of trees to obtain a representation of the tensor with lower complexity
            %
            % xStar = OPTIMIZEDIMENSIONTREE(x,tol,N)
            % x,xStar: TreeBasedTensor
            % tol: double (tolerance)
            % N: integer (number of iterations)
            
            x0 = x;
            t = x.tree;
            xStar = x;
            
            nodes = fastSetdiff(1:t.nbNodes,t.root);
            
            probaN = 1./((1:x.order).^2);
            RVN = DiscreteRandomVariable((1:x.order).',probaN/sum(probaN));
            
            Cstar = storage(x0);
            sigmaStar = [];
            mPerm = 0;
            mStar = 0;
            ind = true;
            
            for iter = 1:N
                m = random(RVN);
                if mPerm < m || ind
                    mPerm = m;
                    ind = false;
                    xSigma = x0;
                    for i = 1:mStar
                        xSigma = permuteNodes(xSigma,sigmaStar(i,:),tol/(m+mStar));
                    end
                end
                
                sigma = zeros(m,t.nbNodes);
                x = xSigma;
                for i = 1:m
                    probaA = x.ranks(x.tree.parent(nodes)).^2;
                    a = random(DiscreteRandomVariable(nodes(:),probaA/sum(probaA)));
                    
                    sigma(i,:) = 1:t.nbNodes;
                    dT = computeDistances(x,a);
                    if any(dT ~= 0)
                        [~,candidates,candidateDistances] = find(dT);
                        probaB = 1./(candidateDistances.^2);
                        b = random(DiscreteRandomVariable(candidates(:),probaB/sum(probaB)));
                        
                        sigma(i,[a , b]) = sigma(i,[b, a]);
                        x = permuteNodes(x,sigma(i,:),tol/(m+mStar));
                    end
                end
                
                C = storage(x);
                if C < Cstar && isAdmissibleRank(x)
                    ind = true;
                    Cstar = C;
                    xStar = x;
                    sigmaStar = [sigmaStar ; sigma];
                    mStar = size(sigmaStar,1);
                end
            end
        end
        
        function [xStar,sigmaStar] = optimizeLeavesPermutation(x,tol,N)
            % OPTIMIZELEAVESPERMUTATION - Optimization over a set of trees obtained by permutations to obtain a representation of the tensor with lower complexity
            %
            % [xStar,sigmaStar] = OPTIMIZELEAVESPERMUTATION(x,tol,N)
            % x,xStar: TreeBasedTensor
            % tol: double (tolerance)
            % N : integer (number of iterations)
            % sigmaStar: 1-by-d double (permutation of (1,...,d))
            
            x0 = x;
            t = x0.tree;
            sigmaStar = 1:x.order;
            r = x0.ranks;
            r(~x0.isActiveNode) = x0.sz(~x0.isActiveNode(t.dim2ind));
            
            ind = find(t.isLeaf);
            dT = zeros(t.nbNodes);
            for i = 1:length(ind)
                for j = i + 1:length(ind)
                    alpha = ind(i); beta = ind(j);
                    commonAs = fastIntersect(ascendants(t,alpha), ascendants(t,beta));
                    gamma = commonAs(t.level(commonAs) == max(t.level(commonAs)));
                    
                    if t.parent(alpha) ~= t.parent(beta)
                        dT(alpha,beta) = r(gamma)*prod(r(setdiff(...
                            unique(nonzeros(t.children(:,setdiff([t.ascendants(alpha), ...
                            t.ascendants(beta)],t.ascendants(gamma))))), ...
                            [t.ascendants(alpha), t.ascendants(beta)])));
                    end
                end
            end
            dT = dT + dT.';
            
            probaN = 1./(1:(x.order-1)).^2;
            RVN = DiscreteRandomVariable((1:(x.order-1)).',probaN/sum(probaN));
            
            probaA = x.ranks(t.parent(t.dim2ind));
            RVA = DiscreteRandomVariable(t.dim2ind(:),probaA/sum(probaA));
            
            m = random(RVN,N);
            
            Cstar = storage(x0);
            C0 = Cstar;
            for iter = 1:N
                a = RVA.random(m(iter));
                
                dim2ind = t.dim2ind;
                sigma = 1:length(dim2ind);
                for j = 1:length(a)
                    [~,candidates,candidateDistances] = find(dT(a(j),:));
                    probaB = 1./(candidateDistances).^2;
                    b = random(DiscreteRandomVariable(candidates(:),probaB/sum(probaB)));
                    
                    [~,loc] = ismember([a(j) , b],dim2ind);
                    sigma(loc) = sigma([loc(2) loc(1)]);
                    dim2ind(loc) = dim2ind([loc(2) loc(1)]);
                end
                
                x = permuteLeaves(x0,sigma,tol);
                C = storage(x);
                if C <= Cstar && isAdmissibleRank(x)
                    % If we found a permutation leading to a smaller
                    % storage complexity
                    if C < Cstar
                        Cstar = C;
                        sigmaStar = sigma;
                        % Else if we already found at least one better
                        % permutation and this one leads to the same storage
                    elseif Cstar < C0 && ~ismember(sigma,sigmaStar,'rows')
                        sigmaStar = [sigmaStar ; sigma];
                    end
                end
            end
            
            % Random selection of one of the best permutations
            l = size(sigmaStar,1);
            if l > 1
                I = randi(l);
                sigmaStar = sigmaStar(I,:);
            end
            xStar = permuteLeaves(x0,sigmaStar,tol);
        end
        
        function x = permuteNodes(x,perm,tol)
            % PERMUTENODES - Permutation of nodes
            %
            % x = PERMUTENODES(x,perm,tol)
            % Permutations of the nodes given a permutation perm of the set of nodes and a tolerance tol (for SVD-based truncations)
            %
            % x: TreeBasedTensor
            % perm: 1-by-x.tree.nbNodes double (permutation of (1,...,x.tree.nbNodes))
            % tol: double (relative precision for SVD truncations)
            
            t = x.tree;
            if all(perm == 1:t.nbNodes)
                return
            end

            if nargin < 3 || isempty(tol)
                tol = 1e-15;
            end
            if t.nbNodes ~= length(perm)
                error('The permutation vector size must be equal to the number of nodes.')
            elseif ~all(sort(perm) == 1:t.nbNodes)
                error('Invalid permutation of the nodes.')
            elseif nnz(perm ~= 1:t.nbNodes) ~= 2
                error('Must permute two nodes at most.')
            end
            
            if ~all(1:length(perm) == perm)
                nodesToPermute = find(1:length(perm) ~= perm);
                a = nodesToPermute(1); b = nodesToPermute(2);
                if ismember(b,t.ascendants(a)) || ismember(b,t.descendants(a))
                    error('Cannot permute the nodes a and b if b is an ascendant or descendant of a.')
                elseif t.parent(a) ~= t.parent(b)
                    asA = ascendants(t,a); asB = ascendants(t,b);
                    commonAs = fastIntersect(asA, asB);
                    gamma = commonAs(t.level(commonAs) == max(t.level(commonAs)));
                    commonAs(commonAs == gamma) = [];
                    subnod = fastSetdiff(unique([asA, asB]),commonAs);
                    
                    tr = Truncator;
                    tr.tolerance = max(1e-15,tol/sqrt(length(subnod)-1));
                    
                    x = orthAtNode(x,gamma);
                    C = x.tensors;
                    L = max(t.level(subnod));
                    if t.root == gamma
                        S = nonzeros(t.children(:,gamma));
                    else
                        S = [nonzeros(t.children(:,gamma)) ; gamma];
                    end
                    for l = t.level(gamma)+1:L
                        nodLvl = nodesWithLevel(t,l);
                        for nod = fastIntersect(nodLvl,subnod)
                            C{gamma} = timesTensor(C{nod},C{gamma},C{nod}.order,find(S == nod));
                            C{nod} = [];
                            S = [nonzeros(t.children(:,nod)) ; setdiff(S,nod,'stable')];
                        end
                    end
                    
                    perm = 1:length(S);
                    perm([find(S == a),find(S == b)]) = perm([find(S == b),find(S == a)]);
                    C{gamma} = permute(C{gamma},perm);
                    S = S(perm);
                    
                    t.adjacencyMatrix(t.parent(a),a) = 0;
                    t.adjacencyMatrix(t.parent(a),b) = 1;
                    t.adjacencyMatrix(t.parent(b),b) = 0;
                    t.adjacencyMatrix(t.parent(b),a) = 1;
                    t = precomputeProperties(t);
                    t = updateDimsFromLeaves(t);
                    
                    for l = L:-1:t.level(gamma)+1
                        nodLvl = nodesWithLevel(t,l);
                        for nod = fastIntersect(nodLvl,subnod)
                            [~,ind] = ismember(nonzeros(t.children(:,nod)),S);
                            nind = setdiff((1:length(S))',ind);
                            
                            w = permute(C{gamma},[ind ; nind]);
                            w = reshape(w,[prod(C{gamma}.sz(ind)), prod(C{gamma}.sz(nind))]);
                            tr.maxRank = min(w.sz);
                            w = tr.truncate(w);
                            S = [S(nind) ; nod];
                            r = w.core.sz(1);
                            x.ranks(nod) = r;
                            C{nod} = reshape(FullTensor(w.space.spaces{1}),[C{gamma}.sz(ind),r]);
                            C{gamma} = reshape(FullTensor(w.space.spaces{2}),[C{gamma}.sz(nind),r]);
                            
                            perm = 1:C{gamma}.order;
                            perm = [perm(1:t.childNumber(nod)-1), C{gamma}.order, perm(t.childNumber(nod):C{gamma}.order-1)];
                            C{gamma} = permute(C{gamma},perm);
                            C{gamma} = timesMatrix(C{gamma},diag(w.core.data),t.childNumber(nod));
                            S = S(perm);
                        end
                    end
                    
                    [~,perm] = ismember(nonzeros(t.children(:,gamma)),S);
                    if gamma~= t.root, perm = [perm ; C{gamma}.order]; end
                    C{gamma} = permute(C{gamma},perm);
                                        
                    x = TreeBasedTensor(C,t,x.isActiveNode);
                    
                    % Ensure that the tensor x is rank admissible
                    tr = Truncator('tolerance',eps,'maxRank',x.ranks);
                    x = tr.truncate(x);
                end
            end
        end
        
        function x = permuteLeaves(x,perm,tol)
            % PERMUTELEAVES - Permutation of leaves
            %
            % x = PERMUTELEAVES(x,perm,tol)
            % Permutations of the leaves given a permutation perm and a tolerance tol (for SVD-based truncations)
            %
            % x: TreeBasedTensor
            % perm: 1-by-d double (permutation of (1,...,d))
            % tol: double (relative precision for SVD truncations)
            
            if nargin < 3 || isempty(tol)
                tol = 1e-15;
            end
            
            if ~all(1:length(perm) == perm)
                if nnz(1:length(perm) ~= perm) ~= 2
                    % Decomposition of the permutation into a sequence of
                    % elementary permutations of two leaf nodes
                    i = 0;
                    init = 1:length(perm);
                    while ~all(init == perm)
                        i = i + 1;
                        S = find(init ~= perm,1);
                        S = [find(perm == init(S)), find(perm == perm(S))];
                        elemPerm{i} = init;
                        elemPerm{i}(S) = elemPerm{i}(flip(S));
                        perm(S) = perm(flip(S));
                    end
                    
                    for i = 1:length(elemPerm)
                        x = permuteLeaves(x,elemPerm{i},tol/length(elemPerm));
                    end
                else
                    t = x.tree;
                    
                    dim = find(1:length(perm) ~= perm);
                    nu = t.dim2ind(dim(1)); mu = t.dim2ind(dim(2));
                    
                    if t.parent(nu) == t.parent(mu)
                        gamma = t.parent(nu);
                        C = x.tensors;
                        ch = nonzeros(t.children(:,gamma));
                        
                        perm = 1:C{gamma}.order;
                        perm([find(ch == nu),find(ch == mu)]) = perm([find(ch == mu),find(ch == nu)]);
                        C{gamma} = permute(C{gamma},perm);
                        C([nu mu]) = C([mu nu]);
                    else
                        asNu = ascendants(t,nu); asMu = ascendants(t,mu);
                        commonAs = fastIntersect(asNu, asMu);
                        gamma = commonAs(t.level(commonAs) == max(t.level(commonAs)));
                        
                        commonAs(commonAs == gamma) = [];
                        subnod = fastSetdiff(unique([asNu, asMu]),commonAs);
                        
                        tr = Truncator;
                        tr.tolerance = max(1e-15,tol/sqrt(length(subnod)-1));
                        
                        x = orthAtNode(x,gamma);
                        C = x.tensors;
                        L = max(t.level(subnod));
                        if t.root == gamma
                            S = nonzeros(t.children(:,gamma));
                        else
                            S = [nonzeros(t.children(:,gamma)) ; gamma];
                        end
                        for l = t.level(gamma)+1:L
                            nodLvl = nodesWithLevel(t,l);
                            for nod = fastIntersect(nodLvl,subnod)
                                C{gamma} = timesTensor(C{nod},C{gamma},C{nod}.order,find(S == nod));
                                C{nod} = [];
                                S = [nonzeros(t.children(:,nod)) ; setdiff(S,nod,'stable')];
                            end
                        end
                        
                        perm = 1:length(S);
                        perm([find(S == nu),find(S == mu)]) = perm([find(S == mu),find(S == nu)]);
                        C{gamma} = permute(C{gamma},perm);
                        C([nu mu]) = C([mu nu]);
                        
                        for l = L:-1:t.level(gamma)+1
                            nodLvl = nodesWithLevel(t,l);
                            for nod = fastIntersect(nodLvl,subnod)
                                [~,ind] = ismember(nonzeros(t.children(:,nod)),S);
                                nind = setdiff((1:length(S))',ind);
                                
                                w = permute(C{gamma},[ind ; nind]);
                                w = reshape(w,[prod(C{gamma}.sz(ind)), prod(C{gamma}.sz(nind))]);
                                tr.maxRank = min(w.sz);
                                w = tr.truncate(w);
                                S = [S(nind) ; nod];
                                r = w.core.sz(1);
                                x.ranks(nod) = r;
                                C{nod} = reshape(FullTensor(w.space.spaces{1}),[C{gamma}.sz(ind),r]);
                                C{gamma} = reshape(FullTensor(w.space.spaces{2}),[C{gamma}.sz(nind),r]);
                                
                                perm = 1:C{gamma}.order;
                                perm = [perm(1:t.childNumber(nod)-1), C{gamma}.order, perm(t.childNumber(nod):C{gamma}.order-1)];
                                C{gamma} = permute(C{gamma},perm);
                                C{gamma} = timesMatrix(C{gamma},diag(w.core.data),t.childNumber(nod));
                                S = S(perm);
                            end
                        end
                        
                        [~,perm] = ismember(nonzeros(t.children(:,gamma)),S);
                        if gamma~= t.root, perm = [perm ; C{gamma}.order]; end
                        C{gamma} = permute(C{gamma},perm);
                    end
                    t.dim2ind(dim) = t.dim2ind(flip(dim));
                    t = updateDimsFromLeaves(t);
                    x = TreeBasedTensor(C,t,x.isActiveNode);
                end
            end
        end
        
        function x = inactivateNodes(x,listnod)
            % INACTIVATENODES - Inactivation of a list of nodes
            %
            % x = INACTIVATENODES(x,listnod)
            % Inactivates the nodes whose indices are given in listnode
            % x: TreeBasedTensor
            % listnod: 1-by-n double (list of nodes to inactivate)
            
            t = x.tree;
            for l = max(t.level):-1:1
                nodLvl = nodesWithLevel(t,l);
                for nod = nodLvl
                    if ismember(nod,listnod) && x.isActiveNode(nod)
                        pnod = t.parent(nod);
                        chNum = t.childNumber(nod);
                        x.tensors{pnod} = timesTensor(x.tensors{nod},x.tensors{pnod},x.tensors{nod}.order,chNum);
                        x.tensors{pnod} = ipermute(x.tensors{pnod},[chNum,setdiff(1:x.tensors{pnod}.order,chNum)]);
                        x.tensors{nod} = [];
                    end
                end
            end
            x = updateProperties(x);
        end
        
        function [xd,s] = evalDiag(x,dims)
            if nargin==1
                dims = 1:x.order;
            else
                error('Method not implemented.')
            end
            if ~all(sort(dims) == (1:x.order))
                error('Method not implemented.')
            end
            t = x.tree;
            s = x.tensors;
            nodes = t.internalNodes;
            for l = max(t.level)-1:-1:0
                for nod = fastIntersect(nodesWithLevel(t,l),nodes)
                    chNod = nonzeros(t.children(:,nod));
                    ischa = x.isActiveNode(chNod);
                    repcha = find(ischa);
                    
                    if all(~ischa)
                        s{nod} = evalDiag(s{nod},1:length(chNod));
                    else
                        repchna = find(~ischa);
                        sch = s{chNod(repcha(1))};
                        for k = 2:length(repcha)
                            ch = chNod(repcha(k));
                            sch = outerProductEvalDiag(sch,s{ch},1,1);
                        end
                        if ~isempty(repchna)
                            s{nod} = timesTensorEvalDiag(sch,s{nod},2:sch.order,repcha,1,repchna);
                        else
                            s{nod} = timesTensor(sch,s{nod},2:sch.order,repcha);
                        end
                    end
                    if nargout == 1
                        s(chNod(repcha)) = {[]};
                    end
                end
            end
            
            xd = s{t.root};
            if isa(xd,'FullTensor') && xd.order==1
                xd=xd.data;
            end
        end
        
        function s = evalDiagNodesBelow(x,nodes,s)
            % EVALDIAGNODESBELOW - Extraction of the diagonal below some nodes of a tensor
            %
            % s = EVALDIAGNODESBELOW(x)
            % Extracts the diagonal of the tensor x
            % See also evalDiag
            %
            % s = EVALDIAGNODESBELOW(x,nodes,s)
            % Extracts the diagonal of the tensor x below the nodes of x in nodes
            %
            % x,s: TreeBasedTensor
            % nodes: 1-by-length(x.activeNodes()) double
            
            if nargin==1
                nodes = 1:x.tree.nbNodes;
                s = x.tensors;
            elseif nargin==2 && ~isempty(setdiff(x.activeNodes(),nodes))
                error('A third argument must be provided.')
            end
            
            t = x.tree;
            nodes = fastIntersect(nodes,t.internalNodes);
            for l = max(t.level)-1:-1:0
                for nod = fastIntersect(nodesWithLevel(t,l),nodes)
                    chNod = nonzeros(t.children(:,nod));
                    ischa = x.isActiveNode(chNod);
                    
                    if all(~ischa)
                        s{nod} = evalDiag(s{nod},1:length(chNod));
                    else
                        repcha = find(ischa);
                        repchna = find(~ischa);
                        sch = s{chNod(repcha(1))};
                        for k = 2:length(repcha)
                            ch = chNod(repcha(k));
                            sch = outerProductEvalDiag(sch,s{ch},1,1);
                        end
                        if ~isempty(repchna)
                            s{nod} = timesTensorEvalDiag(sch,s{nod},2:sch.order,repcha,1,repchna);
                        else
                            s{nod} = timesTensor(sch,s{nod},2:sch.order,repcha);
                        end
                        s(chNod(repcha)) = {[]};
                    end
                end
            end
        end
        
        function x = cat(x,y)
            for i = 1:x.tree.nbNodes
                if x.isActiveNode(i)
                    x.tensors{i} = cat(x.tensors{i},y.tensors{i});
                end
            end
            x.sz = x.sz + y.sz;
            x.sz(1) = 1;
            x.isOrth = false;
            x.orthNode = 0;
        end
        
        function x = kron(x,y)
            for i = 1:x.tree.nbNodes
                if x.isActiveNode(i)
                    x.tensors{i} = kron(x.tensors{i},y.tensors{i});
                end
            end
            x.isOrth = false;
            x.orthNode = 0;
            x = updateProperties(x);
        end
        
        function x = mtimes(a,b)
            if isa(a,'double')
                x = b;
                if numel(a) == 1
                    x.tensors{1} = x.tensors{1}*a;
                elseif size(a,2) == x.ranks(x.tree.root) && size(a,2)>1
                    x.tensors{1} = timesMatrix(x.tensors{1},a,x.tensors{1}.order);
                    x.ranks(x.tree.root) = size(a,1);
                else
                    error('Wrong sizes.')
                end
            elseif isa(b,'double')
                x = a;
                if numel(b) == 1
                    x.tensors{1} = x.tensors{1}*b;
                else
                    error('Wrong sizes.')
                end
            else
                error('Method mtimes not implemented.')
            end
        end
        
        function z = times(x,y)
            if isa(x,'double') || isa(y,'double')
                z = mtimes(x,y);
            else
                error('Method times not implemented.')
            end
        end
        
        function x = mrdivide(x,c)
            % MRDIVIDE - Division of tensor by a scalar
            % function x = MRDIVIDE(x,c)
            % x : TreeBasedTensor
            % c : scalar
            %
            % See also: MRDIVIDE
            
            if isa(c,'double') && numel(c)==1
                x.tensors{1} = x.tensors{1}/c;
            else
                error('Method mrdivide not implemented.')
            end
        end
        
        function s = dot(x,y)
            M = cellfun(@(z) speye(z),num2cell(x.sz),'UniformOutput',false);
            s = dotWithRankOneMetric(x,y,M);
        end
        
        function n = norm(x)
            if ~x.isOrth
                x = orth(x);
            end
            n = norm(x.tensors{x.orthNode});
        end
        
        function [G,x] = gramians(x,varargin)
            % GRAMIANS - Computation of the Gramm matrices of the bases of minimal subspaces associated with all nodes of the tree
            %
            % function G = GRAMIANS(x,node)
            % x: TreeBasedTensor
            % node: 1-by-n double
            % G: 1-by-n cell
            
            if ~x.isOrth || x.orthNode ~= x.tree.root
                x = orth(x);
                if nargout == 1
                    error('x has been modified but not returned.')
                end
            end
            
            t = x.tree;
            G = cell(1,t.nbNodes);
            G{t.root} = 1;
            lvls = t.level;
            list = x.activeNodes;
            
            if nargin == 1
                maxLvl = max(lvls);
            else
                alpha = varargin{1}(:).';
                maxLvl = max(lvls(alpha));
                ascendants = unique(cell2mat(arrayfun(@(x) t.ascendants(x),alpha,'UniformOutput',false)));
                list = list(ismember(list,[alpha, ascendants]));
            end
            
            for l = 0:maxLvl
                nodLvl = list(ismember(list,nodesWithLevel(t,l)));
                nodLvl(ismember(nodLvl,t.dim2ind)) = [];
                
                for nod = nodLvl
                    B = x.tensors{nod};
                    if nod ~= t.root
                        B = timesMatrix(B,G{nod},x.tensors{nod}.order);
                    end
                    children = list(ismember(list,t.children(:,nod)));
                    indlist = 1:x.tensors{nod}.order;
                    for cind = children(:).'
                        clist = indlist;
                        clist(t.childNumber(cind)) = [];
                        G{cind} = timesTensor(x.tensors{nod},B,clist,clist);
                        G{cind} = G{cind}.data;
                    end
                end
            end
            
            if nargin ~= 1 && length(alpha)==1
                G = G{alpha};
            elseif nargin ~= 1
                G = G(alpha);
            end
        end
        
        function x = timesVector(x,V,varargin)
            if isa(V,'double')
                V = {V};
            end
            V = cellfun(@(v) v',V,'uniformoutput',false);
            x = timesMatrix(x,V,varargin{:});
            x = squeeze(x,varargin{:});
        end
        
        function [x] = timesDiagMatrix(x,M,order)
            if any(x.isActiveNode(x.tree.isLeaf))
                error('Method not implemented.')
            end
            
            M = mat2cell(M,size(M,1),ones(1,size(M,2)));
            M = cellfun(@(d) diag(d) , M ,'UniformOutput',0);
            for mu = 1:numel(order)
                i = x.tree.childNumber(x.tree.dim2ind(order(mu)));
                p = x.tree.parent(x.tree.dim2ind(order(mu)));
                x.tensors{p} = timesMatrix(x.tensors{p},M(mu),i);
            end
            
            x.isOrth = false;
            x.orthNode = 0;
        end
        
        function [c] = timesMatrix(c,M,order)
            if nargin==2 && length(M)==c.order
                order = 1:c.order;
            end
            
            t = c.tree;
            for mu = order
                nod = t.dim2ind(mu);
                
                if ~c.isActiveNode(nod)
                    pnod = t.parent(nod);
                    childNb = t.childNumber(nod);
                    c.tensors{pnod} = timesMatrix(c.tensors{pnod},M(mu==order),childNb);
                    c.sz(mu) = c.tensors{pnod}.sz(childNb);
                else
                    c.tensors{nod} = timesMatrix(c.tensors{nod},M(mu==order),1);
                    c.sz(mu) = c.tensors{nod}.sz(1);
                end
            end
            
            c.isOrth = false;
            c.orthNode = 0;
        end
        
        function y = timesMatrixEvalDiag(c,H,dims)
            % TIMESMATRIXEVALDIAG - Contractions with matrices and evaluation of the diagonal
            %
            % y = TIMESMATRIXEVALDIAG(c,H)
            % Equivalent to the method evalDiag applied to the method timesMatrix, but less memory and time consuming
            %
            % c: TreeBasedTensor
            % H: 1-by-c.order or c.order-by-1 cell
            % y: n-by-1 array
            %
            % y = TIMESMATRIXEVALDIAG(c,H,dims)
            % c: TreeBasedTensor
            % H: 1-by-N or N-by-1 cell
            % dims: 1-by-N array
            % y: n-by-1 array
            %
            % See also: TIMESMATRIX, EVALDIAG
            
            if nargin==2
                t = c.tree;
                ind = ismember(t.dim2ind,c.activeNodes);
                v = timesMatrix(c,H(ind),find(ind));
                v = v.tensors;
                
                nodes = t.internalNodes;
                for l = max(t.level)-1:-1:0
                    nodLvl = fastIntersect(nodesWithLevel(t,l),nodes);
                    nodLvl = fastIntersect(nodLvl,c.activeNodes);
                    for nod = nodLvl
                        chNod = nonzeros(t.children(:,nod));
                        aChNod = fastIntersect(chNod,c.activeNodes);
                        naChNod = chNod; naChNod(c.isActiveNode(naChNod)) = [];
                        
                        if ~isempty(aChNod)
                            saCh = v{aChNod(1)};
                            for k = 2:length(aChNod)
                                saCh = outerProductEvalDiag(saCh,v{aChNod(k)},1,1);
                            end
                            v{nod} = timesTensor(saCh,v{nod},2:saCh.order,t.childNumber(aChNod));
                        end
                        
                        if ~isempty(naChNod)
                            snaCh = FullTensor(H{c.tree.dim2ind == naChNod(1)});
                            for i = 2:length(naChNod)
                                snaCh = outerProductEvalDiag(snaCh,FullTensor(H{c.tree.dim2ind == naChNod(i)}),1,1);
                            end
                            
                            if ~isempty(aChNod)
                                v{nod} = timesTensorEvalDiag(snaCh,v{nod},2:snaCh.order,2:snaCh.order,1,1);
                            else
                                v{nod} = timesTensor(snaCh,v{nod},2:snaCh.order,t.childNumber(naChNod));
                            end
                        end
                    end
                end
                y = v{t.root}.data;
            else
                y = timesMatrix(c,H,dims);
                if all(y.sz(dims)==1)
                    if length(dims)>1
                        y = squeeze(y,dims(2:end));
                    end
                else
                    error('Method not implemented.')
                end
            end
        end
        
        function x = squeeze(x,varargin)
            d = x.order;
            if nargin == 1
                dims = 1:d;
                dims(x.sz ~= 1) = [];
            else
                dims = varargin{1};
            end
            dims = sort(dims);
            remainingDims = 1:d;
            remainingDims(dims) = [];
            
            if isempty(remainingDims)
                x = full(x);
                x = x.data;
            else
                ind = x.tree.dim2ind(dims);
                t = x.tree;
                maxLvl = max(t.level);
                xf = x.tensors;
                for l = maxLvl-1:-1:0
                    nodLvl = fastSetdiff(nodesWithLevel(t,l),t.dim2ind);
                    for nod = nodLvl
                        chNod = nonzeros(t.children(:,nod));
                        chNodR = ismember(chNod,ind);
                        for k = 1:length(chNod)
                            ch = chNod(k);
                            if x.isActiveNode(ch) && chNodR(k)
                                if xf{ch}.order==1
                                    xf{ch}.data=xf{ch}.data.';
                                end
                                xf{nod} = timesMatrix(xf{nod},xf{ch}.data,k);
                                xf{ch}=[];
                            end
                        end
                        
                        if any(chNodR)
                            xf{nod} = squeeze(xf{nod},find(chNodR));
                        end
                        if all(chNodR)
                            ind = union(ind,nod);
                        end
                    end
                end
                
                keepind = fastSetdiff(1:t.nbNodes, ind);
                xf = xf(keepind);
                a = t.adjacencyMatrix(keepind,keepind);
                [~,dim2ind]=ismember(t.dim2ind(remainingDims),keepind);
                t = DimensionTree(dim2ind,a);
                x = TreeBasedTensor(xf,t);
                x = removeUniqueChildren(x);
            end
        end
        
        function x = removeUniqueChildren(x)
            % REMOVEUNIQUECHILDREN - Removal of the unique children (nodes with no brothers)
            %
            % x = REMOVEUNIQUECHILDREN(x)
            % x: TreeBasedTensor
            
            t = x.tree;
            nbChildren = sum(t.children~=0,1);
            parentsWithUniqueChild = nbChildren==1;
            uniqueChildren = t.children(1,parentsWithUniqueChild);
            a = t.adjacencyMatrix;
            c = x.tensors;
            
            maxLvl = max(t.level);
            dim2ind = t.dim2ind;
            for l = maxLvl:-1:1
                nodLvl = nodesWithLevel(t,l);
                for nod=intersect(uniqueChildren,nodLvl)
                    pnod = find(a(:,nod));
                    if x.isActiveNode(nod)
                        c{pnod} = timesTensor(c{nod},c{pnod},c{nod}.order,1);
                        a(pnod,:)=a(nod,:);
                        a(nod,:) = false;
                    end
                    c{nod}=[];
                    dim2ind(dim2ind==nod)=pnod;
                end
            end
            
            keepind = fastSetdiff(1:t.nbNodes, uniqueChildren);
            [~,dim2ind] = ismember(dim2ind,keepind);
            c = c(keepind);
            a = a(keepind,keepind);
            
            t = DimensionTree(dim2ind,a);
            x = TreeBasedTensor(c,t);
        end
        
        function x = removeNodesWithOneChild(x)
            % REMOVENODESWITHONECHILD - Removal of the nodes with one child
            %
            % x = REMOVENODESWITHONECHILD(x)
            % x: TreeBasedTensor
            
            warning('This implementation does not respect the order of children.')
            
            t = x.tree;
            nbChildren = sum(t.children~=0,1);
            ind = find(nbChildren==1);
            a = t.adjacencyMatrix;
            c = x.tensors;
            maxLvl = max(t.level);
            
            for l = 0:maxLvl-1
                nodLvl = nodesWithLevel(t,l);
                for nod=intersect(ind,nodLvl)
                    ch = find(a(nod,:));
                    c{ch} = timesTensor(c{ch},c{nod},c{ch}.order,1);
                    c{nod}=[];
                    pnod = find(a(:,nod));
                    if pnod~=0
                        a(pnod,nod)=false;
                        a(pnod,ch)=true;
                    end
                    a(nod,ch) = false;
                end
            end
            keepind = setdiff(1:t.nbNodes,ind);
            c = c(keepind);
            a = a(keepind,keepind);
            [~,dim2ind]=ismember(t.dim2ind,keepind);
            t = DimensionTree(dim2ind,a);
            x = TreeBasedTensor(c,t);
        end
        
        function x = orth(x)
            % ORTH - Orthogonalization of the representation
            %
            % x = ORTH(x)
            % x: TreeBasedTensor
            %
            % All core tensors except the root core represents orthonormal bases of principal subspaces
            
            t = x.tree;
            lvls = t.level;
            maxLvl = max(lvls);
            activeNodes = x.activeNodes();
            
            for l = maxLvl:-1:1
                nodLvl = fastIntersect(nodesWithLevel(t,l),activeNodes);
                for i = 1:numel(nodLvl)
                    nod = nodLvl(i);
                    if ~x.tensors{nod}.isOrth || ...
                            x.tensors{nod}.orthDim ~= x.tensors{nod}.order
                        [x.tensors{nod},R] = orth(x.tensors{nod},x.tensors{nod}.order);
                        pnod = t.parent(nod);
                        x.tensors{pnod} = timesMatrix(x.tensors{pnod},R,t.childNumber(nod));
                        x.tensors{pnod}.isOrth = false;
                    end
                end
            end
            x = updateProperties(x);
            x.isOrth = true;
            x.orthNode = t.root;
        end
        
        function x = orthAtNode(x,nod)
            % ORTHATNODE - Orthogonalization of the representation with respect to a given node
            %
            % x = ORTHATNODE(x,nod)
            % x: TreeBasedTensor
            % nod: 1-by-1 double
            %
            % All core tensors except the one of node nod represents orthonormal bases of principal subspaces the core tensor of node nod is such that the tensor x(i_alpha,i_alpha^c) = sum_k u_k(i_alpha) w_k(i_alpha^c) where w_k is a set of orthonormal vectors
            
            if nod == x.tree.root
                x = orth(x);
            elseif x.isActiveNode(nod)
                [G,x] = x.gramians(nod);
                [U,S] = svd(G);
                S = sqrt(abs(diag(S)));
                L = U*diag(S);
                
                rep = rank(L);
                if rep ~= size(G,1)
                    L(:,rep+1:end) = [];
                    S = pinv(L);
                else
                    S = inv(L);
                end
                x.tensors{nod} = timesMatrix(x.tensors{nod},L',x.tensors{nod}.order);
                x.tensors{nod}.isOrth = false;
                pnod = x.tree.parent(nod);
                chNum = find(nonzeros(x.tree.children(:,pnod))==nod);
                x.tensors{pnod} = timesMatrix(x.tensors{pnod},S,chNum);
                x.tensors{pnod}.isOrth = false;
                x.orthNode = nod;
                x.isOrth = false;
                x = updateProperties(x);
            elseif ~x.isActiveNode(nod)
                error('Non active node.');
            end
        end
        
        function [c] = dotWithRankOneMetric(x,y,M)
            t = x.tree;
            N = cell(1,t.nbNodes);
            N(t.dim2ind) = M;
            lvls = t.level;
            maxLvl = max(lvls);
            for nod = t.dim2ind
                if x.isActiveNode(nod)
                    B = timesMatrix(y.tensors{nod},N(nod),1);
                    N{nod} = timesTensor(x.tensors{nod},B,1,1);
                    N{nod} = reshape(N{nod},[x.tensors{nod}.sz(end), ...
                        y.tensors{nod}.sz(end)]);
                    N{nod} = N{nod}.data;
                end
            end
            
            for l = maxLvl-1:-1:0
                nodLvl = fastIntersect(t.nodesWithLevel(l),t.internalNodes);
                for i = 1:numel(nodLvl)
                    nod = nodLvl(i);
                    ch = t.children(:,nod);
                    ch = ch(1:find(ch,1,'last'));
                    if t.parent(nod) ~= 0
                        ord = 1:x.tensors{nod}.order-1;
                    else
                        ord = 1:x.tensors{nod}.order;
                    end
                    B = timesMatrix(y.tensors{nod},N(ch),ord);
                    N{nod} = timesTensor(x.tensors{nod},B,ord,ord);
                    if l ~= 0
                        N{nod} = reshape(N{nod},[x.tensors{nod}.sz(end), ...
                            y.tensors{nod}.sz(end)]);
                    end
                    N{nod} = N{nod}.data;
                end
            end
            c = N{nodLvl};
        end
        
        function c = timesTensorTimesMatrixExceptDim(x,y,M,order)
            ind = x.tree.dim2ind(order);
            c = reduceDotWithRankOneMetricAtNode(x,y,M,ind);
            c = c{1};
        end
        
        function C = reduceDotWithRankOneMetricAtNode(x,y,M,ind)
            % REDUCEDOTWITHRANKONEMETRICATNODE -
            %
            % C = REDUCEDOTWITHRANKONEMETRICATNODE(x,y,M,ind)
            % x,y: TreeBasedTensor
            % M: 1-by-x.order cell
            % ind: 1-by-1 integer
            % C: 1-by-x.tensors{ind}.order cell or 1-by-1 cell
            
            if any(x.isActiveNode(x.tree.isLeaf))
                error('Method not implemented for active leaves.')
            end
            
            t = x.tree;
            lvl = t.level(ind);
            maxLvl = max(t.level);
            isRoot = false;
            isLeaf = false;
            if lvl == 0
                isRoot = true;
            end
            if all(t.children(:,ind) == 0)
                isLeaf = true;
            end
            N = cell(1,t.nbNodes);
            N(t.dim2ind) = M;
            if ~isLeaf
                C = cell(1,x.tensors{ind}.order);
            else
                C = {[]};
            end
            % Adjacency matrix trick to find the descendants of ind
            A = sparse(double(t.adjacencyMatrix));
            desc = t.nodesIndices(logical(A(ind,:)));
            B = A;
            while nnz(B) ~= 0
                B = B*A;
                desc = [desc t.nodesIndices(logical(B(ind,:)))];
            end
            isDesc = false(1,t.nbNodes);
            isDesc(desc) = true;
            isNotLeaf = true(1,t.nbNodes);
            isNotLeaf(t.dim2ind) = false;
            if ~isLeaf
                % Contract under the node
                for l = maxLvl-1:-1:lvl+1
                    indLvl = false(1,t.nbNodes);
                    indLvl(t.level==l) = true;
                    indLvl = indLvl & isNotLeaf & isDesc; % Remove the leaves and keep the descendants
                    nodLvl = t.nodesIndices(indLvl);
                    for i = 1:numel(nodLvl)
                        nod = nodLvl(i);
                        ch = t.children(:,nod);
                        ch = ch(1:find(ch,1,'last'));
                        if t.parent(nod) ~= 0
                            ord = 1:x.tensors{nod}.order-1;
                        else
                            ord = 1:x.tensors{nod}.order;
                        end
                        B = timesMatrix(y.tensors{nod}, N(ch), ord);
                        N{nod} = timesTensor(x.tensors{nod},B,ord,ord);
                        N{nod} = reshape(N{nod},[x.tensors{nod}.sz(3),...
                            y.tensors{nod}.sz(3)]);
                        N{nod} = N{nod}.data;
                    end
                end
            end
            if ~isRoot
                % Contract over the node
                % Compute the ascendants (parents)
                asc = zeros(lvl,1);
                asc(1) = t.parent(ind);
                for i = 2:lvl
                    asc(i) = t.parent(asc(i-1));
                end
                isAsc = false(1,t.nbNodes);
                isAsc(asc) = true;
                
                % Contract from the leaves to the root, stop to ascendants
                for l = maxLvl-1:-1:0 % Remove the deepest leaves
                    % Get all the nodes of the level
                    indLvl = false(1,t.nbNodes);
                    indLvl(t.level==l) = true;
                    % Remove leaves, parents and descendants
                    indLvl = indLvl & isNotLeaf & (~isDesc) & ~isAsc;
                    nodLvl = t.nodesIndices(indLvl);
                    
                    for i = 1:numel(nodLvl)
                        nod = nodLvl(i);
                        ch = t.children(:,nod);
                        ch = ch(1:find(ch,1,'last'));
                        if t.parent(nod) ~= 0
                            ord = 1:x.tensors{nod}.order-1;
                        else
                            ord = 1:x.tensors{nod}.order;
                        end
                        B = timesMatrix(y.tensors{nod},N(ch),ord);
                        N{nod} = timesTensor(x.tensors{nod},B,ord,ord);
                        N{nod} = reshape(N{nod},...
                            [x.tensors{nod}.sz(end), ...
                            y.tensors{nod}.sz(end)]);
                        N{nod} = N{nod}.data;
                    end
                end
                % Contract back from the root, following ascendants
                asc = flipud(asc);
                nod = asc(1); % this is the root node
                ch = t.children(:,nod);
                ch = ch(1:find(ch,1,'last'));
                if numel(asc) > 1
                    j = t.childNumber(asc(2));
                else
                    j = t.childNumber(ind);
                end
                ch(j) = [];
                ord = 1:x.tensors{nod}.order;
                ord(j) = [];
                B = timesMatrix(y.tensors{nod},N(ch),ord);
                N{nod} = timesTensor(x.tensors{nod},B,ord,ord);
                N{nod} = reshape(N{nod},[x.tensors{nod}.sz(j), ...
                    y.tensors{nod}.sz(j)]);
                N{nod} = N{nod}.data;
                if numel(asc) > 1
                    for i = 2:numel(asc)
                        nod = asc(i);
                        ch = t.children(:,nod);
                        ch = ch(1:find(ch,1,'last'));
                        if i == numel(asc)
                            j = t.childNumber(ind);
                        else
                            j = t.childNumber(asc(i+1));
                        end
                        ch(j) = [];
                        ord = 1:x.tensors{nod}.order;
                        ord(j) = [];
                        Npi = N{asc(i-1)};
                        B = timesMatrix(y.tensors{nod},{N{ch},Npi},ord);
                        N{nod} = timesTensor(x.tensors{nod},B,ord,ord);
                        N{nod} = reshape(N{nod},[x.tensors{nod}.sz(j), ...
                            y.tensors{nod}.sz(j)]);
                        N{nod} = N{nod}.data;
                    end
                end
            end
            % Fill C
            ch = t.children(:,ind);
            ch = ch(1:find(ch,1,'last'));
            if ~isLeaf
                C(1:numel(ch)) = N(ch);
            end
            if ~isRoot
                C(end) = N(t.parent(ind));
            end
        end
        
        function x = updateProperties(x)
            % UPDATEPROPERTIES - Update of all properties
            %
            % x = UPDATEPROPERTIES(x)
            % x: TreeBasedTensor
            
            x.ranks = zeros(1,x.tree.nbNodes);
            x.isActiveNode = reshape(~cellfun(@isempty,x.tensors),1,x.tree.nbNodes);
            if ~all(x.isActiveNode(x.tree.internalNodes))
                error('Method not implemented for this format.')
            end
            for nod = 1:x.tree.nbNodes
                if ~isempty(x.tensors{nod}) && ~isa(x.tensors{nod},'FullTensor')
                    sznod = size(x.tensors{nod});
                    ch = nonzeros(x.tree.children(:,nod));
                    if nod == x.tree.root
                        ord = length(ch);
                    elseif x.tree.isLeaf(nod)
                        ord = 2;
                    else
                        ord = length(ch)+1;
                    end
                    sznod = [sznod,ones(1,ord-length(sznod))];
                    x.tensors{nod}=FullTensor(x.tensors{nod},length(sznod),sznod);
                end
            end
            
            for nod = x.tree.internalNodes
                p = x.tree.parent(nod);
                ch = nonzeros(x.tree.children(:,nod));
                if p ~= 0 || (x.tensors{nod}.order == length(ch)+1)
                    x.ranks(nod) = x.tensors{nod}.sz(end);
                else
                    x.ranks(nod) = 1;
                end
            end
            
            x.sz = zeros(1,x.order);
            for nod = x.tree.dim2ind
                mu = find(x.tree.dim2ind==nod);
                if isempty(x.tensors{nod})
                    pnod = x.tree.parent(nod);
                    j = x.tree.childNumber(nod);
                    x.sz(mu) = x.tensors{pnod}.sz(j);
                else
                    x.sz(mu) = x.tensors{nod}.sz(1);
                    x.ranks(nod) = x.tensors{nod}.sz(end);
                end
            end
        end
        
        function a = activeNodes(x)
            % ACTIVENODES - List of active nodes
            %
            % x = ACTIVENODES(x)
            % x: TreeBasedTensor
            % a: 1-by-length(ACTIVENODES(x)) integer
            
            a = x.tree.nodesIndices(x.isActiveNode);
        end
        
        function a = nonActiveNodes(x)
            % NONACTIVENODES - List of non active nodes
            %
            % a = NONACTIVENODES(x)
            % x: TreeBasedTensor
            % a: 1-by-length(NONACTIVENODES(x)) integer
            
            a = x.tree.nodesIndices(~x.isActiveNode);
        end
        
        function a = activeDims(x)
            % ACTIVEDIMS - List of active dimensions
            %
            % a = ACTIVEDIMS(x)
            % x: TreeBasedTensor
            % a: 1-by-length(ACTIVEDIMS(x)) integer
            
            a = find(ismember(x.tree.dim2ind, activeNodes(x)));
        end
        
        function a = nonActiveDims(x)
            % NONACTIVEDIMS - List of non active dimensions
            %
            % a = NONACTIVEDIMS(x)
            % x: TreeBasedTensor
            % a: 1-by-length(NONACTIVEDIMS(x)) integer
            
            a = setdiff(1:x.order, activeDims(x));
        end
        
        function a = isActiveDim(x,mu)
            % ISACTIVEDIM - True if the given dimension is active
            %
            % a = ISACTIVEDIM(x,mu)
            % Returns true if the node x.dim2ind(mu) is active
            % x: TreeBasedTensor
            % mu: 1-by-1 integer
            % a: 1-by-1 logical
            
            a = ismember(x.tree.dim2ind(mu), activeNodes(x));
        end
        
        function sv = singularValues(x)
            % SINGULARVALUES - Tree-based singular values of a tensor
            %
            % sv = SINGULARVALUES(x)
            % Returns the singular values associated with alpha-matricizations of the tensor, for all alpha in the dimension tree
            %
            % x: TreeBasedTensor
            % sv: cell array of length x.tree.nbNodes
            % sv{i} contains the alpha-singular values of the tensor, where alpha the set of dimensions associated with the i-th node of the tree
            
            x = orth(x);
            G = x.gramians();
            sv = cell(size(G));
            for i = 1:numel(G)
                sv{i} = sqrt(svd(G{i}));
            end
        end
        
        function r = representationRank(x)
            % REPRESENTATIONRANK - Representation tree-based rank of the tensor
            %
            % r = REPRESENTATIONRANK(x)
            % x: TreeBasedTensor
            % r: 1-by-x.tree.nbNodes integer
            
            r  = x.ranks;
        end
        
        function r = rank(x)
            % RANK - Tree-based rank of the tensor (computed by SVD)
            %
            % r = RANK(x)
            % x: TreeBasedTensor
            % r: 1-by-x.tree.nbNodes integer
            
            sv = singularValues(x);
            r = cellfun(@nnz,sv);
        end
        
        function plot(x,nodesLabels)
            % PLOT - Graph of the tree with a label at each node
            %
            % PLOT(x,label)
            % x: TreeBasedTensor
            % label: array or cell of length(x.tree.nbNodes) containing labels of nodes or function_handle
            % If label is a function_handle, the label of node i is label(x.tensors{i})
            
            if nargin==1
                nodesLabels = 1:x.tree.nbNodes;
            elseif nargin==2 && isa(nodesLabels,'function_handle')
                nodesLabels = cellfun(nodesLabels,x.tensors,'uniformoutput',false);
            end
            
            plotNodes(x.tree,x.activeNodes(),'ok','markerfacecolor','k','markersize',7)
            hold on
            plotNodes(x.tree,x.nonActiveNodes(),'ok','markerfacecolor','w','markersize',7)
            plotEdges(x.tree)
            plotLabelsAtNodes(x.tree,nodesLabels)
            hold off
        end
        
        function NN = neuralNetwork(x)
            nbNeurons = 0;
            neuronsGroup = [];
            neuronsIndexInGroup = [];
            neuronsLayer = [];
            neuronsParentGroup = [];
            depth = max(x.tree.level)+2;
            widths = zeros(1,depth+2);
            widths(1) = sum(x.sz);
            for k=1:x.order
                nalpha = x.sz(k);
                neuronsGroup = [neuronsGroup , -x.tree.dim2ind(k)*ones(1,nalpha)];
                neuronsLayer = [neuronsLayer , ones(1,nalpha)];
                nbNeurons = nbNeurons + nalpha;
                neuronsParentGroup = [neuronsParentGroup, x.tree.dim2ind(k)*ones(1,nalpha)];
                neuronsIndexInGroup = [neuronsIndexInGroup , 1:nalpha];
            end
            for alpha=1:x.tree.nbNodes
                l = length(unique(x.tree.level(x.tree.descendants(alpha))))+2;
                ralpha = x.ranks(alpha);
                widths(l) = widths(l)+ralpha;
                neuronsGroup = [neuronsGroup , alpha*ones(1,ralpha)];
                neuronsLayer = [neuronsLayer , (l)*ones(1,ralpha)];
                nbNeurons = nbNeurons + ralpha;
                neuronsParentGroup = [neuronsParentGroup , x.tree.parent(alpha)*ones(1,ralpha)];
                neuronsIndexInGroup = [neuronsIndexInGroup , 1:ralpha];
            end
            
            NN.nbNeurons = nbNeurons;
            NN.widths = widths;
            NN.depth = depth;
            NN.tree = x.tree;
            NN.neuronsGroup = neuronsGroup;
            NN.neuronsLayer = neuronsLayer;
            NN.neuronsParentGroup = neuronsParentGroup;
            NN.neuronsIndexInGroup = neuronsIndexInGroup;
            NN.tree = x.tree;
            
        end
        
        function varargout = plotNeuralNetwork(x,varargin)
            network = neuralNetwork(x);
            
            [xnodes,ynodes]=network.tree.treelayout;
            xn=zeros(network.nbNeurons,1);
            yn=zeros(network.nbNeurons,1);
            Hsize = max(xnodes)-min(xnodes);
            Wsize = max(ynodes)-min(ynodes);
            h = Hsize/max(network.widths);
            yshift = Wsize/(network.depth-2)/10;
            ybottom = min(ynodes) + yshift;
            for k=1:network.nbNeurons
                alpha = network.neuronsGroup(k);
                nalpha = nnz(network.neuronsGroup==alpha);
                i = network.neuronsIndexInGroup(k);
                xn(k) = xnodes(abs(network.neuronsGroup(k))) + h*(-nalpha/2 + i);
                if network.neuronsGroup(k)<0
                    yn(k) = ybottom;
                else
                    yn(k) = ynodes(abs(network.neuronsGroup(k))) + Wsize/(network.depth-2) + yshift;
                end
            end
            
            hneurons = plot(xn,yn,'ko',varargin{:});
            hold on
            edges = zeros(1,2);
            for k=1:network.nbNeurons
                p = find(network.neuronsGroup == network.neuronsParentGroup(k));
                xnp = xn(p);
                ynp = yn(p);
                X = [xn(k)*ones(length(p),1),xnp(:)];
                Y = [yn(k)*ones(length(p),1),ynp(:)];
                hedges = plot(X',Y','k-',varargin{:});
                edges = [edges;k*ones(length(p),1),p(:)];
            end
            
            if nargout>0
                varargout{1}.network = network;
                varargout{1}.hneurons = hneurons;
                varargout{1}.hedges = hedges;
                varargout{1}.xn=xn;
                varargout{1}.yn=yn;               
                varargout{1}.edges = edges;
            end
            
        end
        
        function v = evalDiagBelow(f,mu)
            % EVALDIAGBELOW - Evaluation of the diagonal of the tensor of the function v^\alpha of the representation
            % f = \sum_{k=1}^{r_\alpha} v^\alpha_k w^\alpha_k
            % (optionally for a node \alpha = mu)
            %
            % v = EVALDIAGBELOW(f,mu)
            % f: TreeBasedTensor
            % mu: 1-by-1 integer
            % v: 1-by-f.tree.nbNodes cell
            
            t = f.tree;
            v = f.tensors;
            
            if nargin == 2 && mu ~= t.root
                excludeList = [mu ascendants(t,mu)];
            else
                excludeList = [];
            end
            v(excludeList(2:end)) = {[]};
            
            nodes = t.internalNodes;
            for l = max(t.level)-1:-1:0
                nodLvl = fastIntersect(fastIntersect(nodesWithLevel(t,l),nodes),f.activeNodes);
                nodLvl = fastSetdiff(nodLvl,excludeList);
                
                for nod = nodLvl
                    ch = nonzeros(t.children(:,nod));
                    aChNod =  ch(f.isActiveNode(ch));
                    naChNod = ch(~f.isActiveNode(ch));
                    
                    if ~isempty(aChNod)
                        vaChNod = v{aChNod(1)};
                        for i = 2:length(aChNod)
                            vaChNod = outerProductEvalDiag(vaChNod,v{aChNod(i)},1,1);
                        end
                        
                        if ~isempty(naChNod) && nod ~= t.root
                            v{nod} = timesTensorEvalDiag(v{nod},vaChNod,t.childNumber(aChNod),2:vaChNod.order,t.childNumber(naChNod),1);
                        else
                            v{nod} = timesTensor(vaChNod,v{nod},2:vaChNod.order,t.childNumber(aChNod));
                        end
                    else
                        v{nod} = evalDiag(v{nod},t.childNumber(naChNod));
                    end
                end
            end
        end
        
        function w = evalDiagAbove(f,v,mu)
            % EVALDIAGABOVE - Evaluation of the diagonal of the tensor of the function w^\alpha of the representation
            % f = \sum_{k=1}^{r_\alpha} v^\alpha_k w^\alpha_k
            % (optionally for a node \alpha = mu)
            %
            % w = EVALDIAGABOVE(f,v,mu)
            % f: TreeBasedTensor
            % v: 1-by-f.tree.nbNodes cell
            % mu: 1-by-1 integer
            % w: 1-by-f.tree.nbNodes cell or 1-by-1 cell if mu is provided
            
            if nargin == 1 || isempty(v)
                v = evalDiagBelow(f);
            end
            
            t = f.tree;
            w = cell(1,t.nbNodes);
            if nargin == 3
                includeList = [mu t.ascendants(mu)];
                if mu == t.root
                    w{t.root} = FullTensor.ones(v{t.root}.sz);
                end
            else
                includeList = 1:t.nbNodes;
            end
            
            for l = 1:max(t.level)
                nodLvl = fastIntersect(nodesWithLevel(t,l),f.activeNodes);
                nodLvl = fastIntersect(nodLvl,includeList);
                for nod = nodLvl
                    pa = t.parent(nod);
                    ch = nonzeros(t.children(:,pa)); ch(ch == nod) = [];
                    aChNod =  ch(f.isActiveNode(ch));
                    naChNod = ch(~f.isActiveNode(ch));
                    
                    if ~isempty(aChNod)
                        vaChNod = v{aChNod(1)};
                        for i = 2:length(aChNod)
                            vaChNod = outerProductEvalDiag(vaChNod,v{aChNod(i)},1,1);
                        end
                        if ~isempty(naChNod)
                            w{nod} = timesTensorEvalDiag(vaChNod,f.tensors{pa},2:vaChNod.order,t.childNumber(aChNod),1,t.childNumber(naChNod));
                        else
                            w{nod} = timesTensor(vaChNod,f.tensors{pa},2:vaChNod.order,t.childNumber(aChNod));
                        end
                    elseif length(t.childNumber(naChNod)) ~= 1
                        w{nod} = evalDiag(f.tensors{pa},t.childNumber(naChNod));
                        ind = 1:w{nod}.order; ind(t.childNumber(naChNod(1))) = [];
                        w{nod} = permute(w{nod}, [t.childNumber(naChNod(1)) ind]);
                    else
                        ind = 1:f.tensors{pa}.order; ind(t.childNumber(naChNod)) = [];
                        w{nod} = permute(f.tensors{pa},[t.childNumber(naChNod) ind]);
                    end
                    if pa ~= t.root
                        w{nod} = timesTensorEvalDiag(w{pa},w{nod},2,3,1,1);
                    end
                end
            end
            if nargin == 3
                w = w{mu};
            end
        end
        
        function [g,ind] = parameterGradientEvalDiag(f,mu,H)
            % PARAMETERGRADIENTEVALDIAG - Diagonal of the gradient of the tensor with respect to a given parameter
            %
            % [g,ind] = PARAMETERGRADIENTEVALDIAG(f,mu,H)
            % f: TreeBasedTensor
            % mu: integer (index of node of the dimension tree)
            % H: 1-by-f.order cell
            % g: FullTensor
            % ind: length(f.tree.children(mu))-by-1 integer
            
            u = evalDiagBelow(f,mu);
            w = evalDiagAbove(f,u,mu);
            t = f.tree;
            
            ch = nonzeros(f.tree.children(:,mu));
            aCh = ch(f.isActiveNode(ch));
            naCh = ch(~f.isActiveNode(ch));
            if t.isLeaf(mu)
                g = FullTensor(ones(w.sz(1),1), 1, w.sz(1));
                if nargin == 3
                    g = outerProductEvalDiag(g,FullTensor(H{t.dim2ind == mu}),1,1);
                else
                    g = outerProductEvalDiag(g,FullTensor(eye(u{mu}.sz(1))),[],[],true);
                end
                ind = [find(t.dim2ind == mu) ; 2];
            else
                if ~isempty(aCh)
                    g = u{aCh(1)};
                    for i = 2:length(aCh)
                        g = outerProductEvalDiag(g,u{aCh(i)},1,1);
                    end
                end
                if ~isempty(naCh)
                    gna = FullTensor(ones(w.sz(1),1), 1, w.sz(1));
                    for i = 1:length(naCh)
                        if nargin == 3
                            gna = outerProductEvalDiag(gna,FullTensor(H{t.dim2ind == naCh(i)}),1,1);
                        else
                            gna = outerProductEvalDiag(gna,FullTensor(eye(f.tensors{mu}.sz(t.childNumber(naCh(i))))),[],[],true);
                        end
                    end
                    
                    if ~isempty(aCh)
                        g = outerProductEvalDiag(g,gna,1,1);
                    else
                        g = gna;
                    end
                end
                ind = [find(ismember(t.dim2ind,naCh)) ; length(aCh) + 1 + (1:length(naCh))];
            end
            if mu ~= t.root
                g = outerProductEvalDiag(g,w,1,1);
            end
        end
        
        function t = tTTensor(x)
            % TTTENSOR - Conversion to TTTensor
            %
            % function x = TTTENSOR(x)
            % Converts x into a TTTensor
            % x: TreeBasedTensor, with a Tensor-Train structure
            % t: TTTensor
            
            cores = x.tensors(x.isActiveNode);
            cores{1}.sz = [cores{1}.sz 1];
            cores{1} = reshape(cores{1},cores{1}.sz);
            cores{end}.sz = [1 cores{end}.sz];
            cores{end} = reshape(cores{end},cores{end}.sz);
            cores = flip(cores);
            t = TTTensor(cores(:));
        end
    end
    
    methods (Static)
        function C = create(generator,tree,ranks,sz,isActiveNode)
            % CREATE - Creation of a tree-based tensor from a generator
            %
            % function x = CREATE(generator,tree,ranks,sz)
            % Build a TreeBasedTensor with active leaves using the function generator (randn, ones...)
            % tree: DimensionTree
            % ranks: 1-by-tree.nbNodes integer or 'random'
            % sz : 1-by-d integer, where d is the order of the tensor, or 'random'
            % C: TreeBasedTensor with active leaves
            %
            % function x = CREATE(generator,tree,ranks)
            % Build a TreeBasedTensor with non active leaves using the function generator (randn, ones...)
            % The size of the tensor is given by ranks(tree.dim2ind)
            %
            % function x = CREATE(generator,tree,ranks,sz,isActiveNode)
            % Build a TreeBasedTensor with active nodes given by activeNodes
            % isActiveNode: 1-by-tree.nbNodes logical or 'random'
            
            
            if nargin<=2 || (isa(ranks,'char') && strcmpi(ranks,'random'))
                ranks = randi(5,1,tree.nbNodes);
                ranks(tree.root)=1;
            end
            
            if nargin<=3
                sz = ranks(tree.dim2ind);
            elseif isa(sz,'char') && strcmpi(sz,'random')
                sz = randi(10,1,length(tree.dim2ind));
            end
            
            if nargin<=4
                isActiveNode = true(1,tree.nbNodes);
            elseif isa(isActiveNode,'char') && strcmpi(isActiveNode,'random')
                isActiveNode = TreeBasedTensor.randomIsActiveNode(tree);
            end
            
            a = cell(1,tree.nbNodes);
            
            for i=1:tree.nbNodes
                
                if isActiveNode(i) && ~tree.isLeaf(i)
                    ch = nonzeros(tree.children(:,i));
                    
                    szi = [];
                    for k=ch(:)'
                        if isActiveNode(k)
                            szi=[szi,ranks(k)];
                        elseif tree.isLeaf(k)
                            szi=[szi,sz(tree.dim2ind==k)];
                        elseif (~tree.isLeaf(k)) && (~isActiveNode(k))
                            error('Inactive nodes should be leaves.')
                        end
                    end
                    if (i~= tree.root) || (i==tree.root && ranks(i)>1)
                        szi = [szi,ranks(i)];
                    end
                    a{i} = FullTensor.create(generator,szi);
                elseif tree.isLeaf(i) && isActiveNode(i)
                    szi = [sz(tree.dim2ind==i),ranks(i)];
                    a{i} = FullTensor.create(generator,szi);
                elseif ~isActiveNode(i) && ~tree.isLeaf(i)
                    error('Inactive nodes should be leaves.')
                end
            end
            C = TreeBasedTensor(a,tree);
        end
        
        function C = randn(varargin)
            % RANDN - Creation of a tensor of size sz and tree-based rank ranks with i.i.d. entries generated with randn
            %
            % x = RANDN(tree,ranks,sz,isActiveNode)
            % tree: DimensionTree
            % ranks: 1-by-tree.nbNodes integer
            % sz: 1-by-d integer
            % isActiveNodes: 1-by-tree.nbNodes integer
            
            C = TreeBasedTensor.create(@randn,varargin{:});
        end
        
        function C = rand(varargin)
            % RAND - Creation of a tensor of size sz and tree-based rank ranks with i.i.d. entries generated with rand
            %
            % x = rand(tree,ranks,sz,isActiveNode)
            % tree: DimensionTree
            % ranks: 1-by-tree.nbNodes integer
            % sz: 1-by-d integer
            % isActiveNodes: 1-by-tree.nbNodes integer
            
            C = TreeBasedTensor.create(@rand,varargin{:});
        end
        
        function C = ones(varargin)
            % ONES - Creation of a tensor of size sz and tree-based rank ranks with entries equal to 1
            %
            % x = ones(tree,ranks,sz,isActiveNode)
            % tree: DimensionTree
            % ranks: 1-by-tree.nbNodes integer
            % sz: 1-by-d integer
            % isActiveNodes: 1-by-tree.nbNodes integer
            
            C = TreeBasedTensor.create(@ones,varargin{:});
        end
        
        function C = zeros(varargin)
            % ZEROS - Creation of a tensor of size sz and tree-based rank ranks with zero entries
            % x = zeros(tree,ranks,sz,isActiveNode)
            % tree: DimensionTree
            % ranks: 1-by-tree.nbNodes integer
            % sz: 1-by-d integer
            % isActiveNodes: 1-by-tree.nbNodes integer
            
            C = TreeBasedTensor.create(@zeros,varargin{:});
        end
    end
    
    methods (Static,Hidden)
        function isActiveNode = randomIsActiveNode(tree)
            % RANDOMISACTIVENODE - Random list of active nodes
            %
            % isActiveNode = RANDOMISACTIVENODE(tree)
            % tree: DimensionTree
            % isActiveNode: 1-by-tree.nbNodes integer
            
            choice =randi(3);
            if choice==1
                isActiveNode = true(1,tree.nbNodes);
            elseif choice==2
                isActiveNode = ~tree.isLeaf;
            elseif choice==3
                d=length(tree.dim2ind);
                p = randperm(d,randi(d));
                isActiveNode = true(1,tree.nbNodes);
                isActiveNode(tree.dim2ind(p))=false;
            end
        end
    end
end