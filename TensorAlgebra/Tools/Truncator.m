% Class Truncator

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

classdef Truncator
    
    properties
        tolerance
        maxRank
    end
    
    properties (Hidden)
        hsvdType = 2 % 1 for root to leaves, 2 for leaves to root
    end
    
    methods
        function t = Truncator(varargin)
            p = ImprovedInputParser();
            addParamValue(p,'tolerance',1e-8);
            addParamValue(p,'maxRank',50);
            parse(p,varargin{:});
            t = passMatchedArgsToProperties(p,t);
        end
        
        function y = truncate(t,x)
            if isa(x,'TuckerLikeTensor') && isa(x.space,'TSpaceOperators')
                % Process as TSpaceVectors (exploit sparsity patterns)
                isOperator=true;
                operatorSize = x.space.sz ;
                [x,pattern] = vectorize(x) ;
            else
                isOperator = false;
            end
            if ndims(x) == 2
                y = t.svd(x);
            elseif ndims(x) > 2
                if isa(x,'FullTensor')
                    y = t.hosvd(x);
                elseif isa(x,'TuckerLikeTensor')
                    switch class(x.core)
                        case 'DiagonalTensor'
                            I = TuckerLikeTensor.eye(x.sz);
                            localSolver = RankOneALSLinearSolver(I,x, ...
                                'maxIterations',30,...
                                'stagnation',1e-5,...
                                'display',false);
                            gs = GreedyLinearSolver(I,x,...
                                'maxIterations', x.sz(1),...
                                'localSolver',localSolver,...
                                'display',false);
                            y = gs.solve();
                            if norm(y-x)/norm(x) > t.tolerance
                                warning('Truncator:truncate',['The approximation ' ...
                                    'is not accurate enough.'])
                            end
                        case 'FullTensor'
                            y = t.hosvd(x);
                        case 'TreeBasedTensor'
                            y = t.hsvd(x);
                        case 'TTTensor'
                            y = t.ttsvd(x);
                        otherwise
                            error('Unknown format')
                    end
                elseif isa(x,'TTTensor')
                    y = t.ttsvd(x);
                elseif isa(x,'TreeBasedTensor')
                    y = t.hsvd(x);
                else
                    error('Method not implemented.')
                end
            else
                error('Wrong order of tensor x.')
            end
            if isOperator % Go back to operator format
                y = unvectorize(y,operatorSize,1:y.order,pattern) ;
            end
        end
        
        function y = truncSvd(t,x,tol,p)
            % y = truncSvd(t,x)
            % Truncated SVD of the matrix x with relative precision
            % t.tolerance and maximal rank t.maxRank
            %
            % y = truncSvd(t,x,tol)
            % use the argument tol for tolerance, instead of t.tolerance
            %
            % y = truncSvd(t,x,tol,p)
            % Truncated SVD of the matrix x with relative precision tol
            % in schatten-p norm (1<=p<=Inf, p=2 for Frobenius), p=2 by default
            
            if nargin == 2 || isempty(tol)
                tol = t.tolerance;
            end
            if nargin<=3
                p=2;
            end
            [L,D,R]=svd(x,'econ');
            d = diag(D);
            if p==Inf
                err = d/max(d);
            else
                if verLessThan('matlab','8.2') % compatibility (<R2013b)
                    err = (flipdim(cumsum(flipdim(d.^p,1)),1)/sum(d.^p)).^(1/p);
                else
                    err = (flip(cumsum(flip(d.^p)))/sum(d.^p)).^(1/p);
                end
            end
            err = [err(2:end);0];
            m = find(err<tol);
            if isempty(m)
                m = min(size(x));
            else
                m = min(m);
            end
            %if m>t.maxRank
            %    warning('maxRank reached, tolerance not achieved')
            %end
            m = min(m,t.maxRank);
            L = L(:,1:m);
            D = diag(D(1:m,1:m));
            R = R(:,1:m);
            ys = TSpaceVectors({L;R});
            y = CanonicalTensor(ys,D);
        end
        
        function y = svd(t,x)
            if ndims(x) ~= 2
                error('Wrong order.')
            end
            if isa(x,'double')
                y = t.truncSvd(x);
            elseif isa(x,'FullTensor')
                y = t.truncSvd(x.data);
            elseif isa(x,'TuckerLikeTensor')
                x = orth(x);
                y = t.truncSvd(x.core.data);
                y.space.spaces{1} = x.space.spaces{1}*y.space.spaces{1};
                y.space.spaces{2} = x.space.spaces{2}*y.space.spaces{2};
                y.space.sz = x.sz;
                y.sz = y.space.sz;
            elseif isa(x,'TreeBasedTensor')
                y = t.hsvd(x);
            else
                error('Method not implemented.')
            end
        end
        
        function y = hosvd(t,x)
            if isa(x,'double')
                x = FullTensor(x);
            end
            
            d = x.order;
            if d == 2
                y = svd(x);
            else
                localTolerance = t.tolerance / sqrt(d);
                if isa(x,'FullTensor')
                    sz = x.sz;
                    A = cell(d,1);
                    r = zeros(1,d);
                    for mu = 1:d
                        order = [mu,1:mu-1,mu+1:d];
                        y = permute(x.data,order);
                        y = reshape(double(y),sz(mu),prod(sz([1:mu-1,mu+1:d])));
                        A{mu} = t.truncSvd(y,localTolerance);
                        A{mu} = A{mu}.space.spaces{1};
                        r(mu) = size(A{mu},2);
                    end
                    a = timesMatrix(x,A{1}',1);
                    for mu = 2:d
                        a = timesMatrix(a,A{mu}',mu);
                    end
                    y = TuckerLikeTensor(a,TSpaceVectors(A));
                elseif isa(x,'TuckerLikeTensor') && isa(x.core,'FullTensor')
                    x = orth(x);
                    y = t.hosvd(x.core);
                    for mu = 1:d
                        y.space.spaces{mu} = x.space.spaces{mu}*y.space.spaces{mu};
                    end
                    y.space.sz = x.sz;
                    y.sz = x.sz;
                else
                    error('Wrong type.')
                end
            end
        end
        
        function y = hsvd(t,x,varargin)
            % y = hsvd(t,x,tree,isActiveNode)
            % Truncated SVD in tree-based format of a tensor x
            % x: TreeBasedTensor, TuckerLikeTensor, FullTensor
            % tree: DimensionTree
            % isActiveNode: logical 1-by-tree.nbNodes (1:tree.nbNodes by default)
            %
            % - if x is a TreeBasedTensor, tree = x.tree and isActiveNode =
            % x.isActiveNode
            % - if x is a TuckerLikeTensor with core class TreeBasedTensor, then
            % tree = x.core.tree and isActiveNode = true(1,tree.nbNodes)
            
            if isa(x,'FullTensor')
                tree = varargin{1};
                if nargin<4
                    isActiveNode = true(1,tree.nbNodes);
                else
                    isActiveNode = varargin{2};
                end
                
                maxRank=t.maxRank;
                if numel(maxRank)==1
                    maxRank = repmat(maxRank,1,tree.nbNodes);
                    maxRank(1)=1;
                end
                
                localTolerance = t.tolerance/sqrt(nnz(isActiveNode)-1);
                maxLvl = max(tree.level);
                C = cell(1,tree.nbNodes);
                szx = x.sz;
                nodesx = tree.dim2ind;
                ranks = ones(1,tree.nbNodes);
                for l = maxLvl:-1:1
                    nodl= tree.nodesWithLevel(l);
                    for nod = nodl
                        if isActiveNode(nod)
                            if tree.isLeaf(nod)
                                rep = find(nod==nodesx);
                            else
                                nodch = nonzeros(tree.children(:,nod));
                                [~,rep] = ismember(nodch,nodesx);
                            end
                            rep = rep(:)';
                            repc = setdiff(1:length(nodesx),rep);
                            
                            Z = permute(x,[rep,repc]);
                            Z = reshape(Z,[prod(szx(rep)),prod(szx(repc))]);
                            Ztr = t.truncSvd(Z.data,localTolerance);
                            C{nod} = Ztr.space.spaces{1};
                            ranks(nod) = size(C{nod},2);
                            C{nod} = FullTensor(C{nod},numel(rep)+1,[szx(rep),ranks(nod)]);
                            Z = Ztr.space.spaces{2}*diag(Ztr.core.data);
                            szx = [szx(repc),ranks(nod)];
                            x = FullTensor(Z,length(szx),szx);
                            nodesx = [nodesx(repc),nod];
                            %else
                            %    nodesx(nod==nodex) = tree.parent(nod);
                        end
                    end
                end
                
                rootch = nonzeros(tree.children(:,tree.root)');
                [~,rep] = ismember(rootch,nodesx);
                C{tree.root} = permute(x,rep);
                y = TreeBasedTensor(C,tree);
            elseif isa(x,'TreeBasedTensor')
                maxRank=t.maxRank;
                if numel(maxRank)==1
                    maxRank = repmat(maxRank,1,x.tree.nbNodes);
                    maxRank(1) = 1;
                end
                
                switch t.hsvdType
                    case 1
                        x = orth(x);
                        localTolerance = t.tolerance/sqrt(nnz(x.isActiveNode)-1);
                        G = x.gramians();
                        Q = cell(size(G));
                        Q = Q';
                        sz = zeros(size(G));
                        for i = 1:numel(G)
                            % truncation of the Gramian in trace norm for a control of frobenius norm of the tensor
                            t.maxRank = maxRank(i);
                            q = t.truncSvd(G{i},localTolerance^2,1);
                            sz(i) = q.core.sz(1);
                            Q{i} = q.space.spaces{1};
                        end
                        tree = x.tree;
                        lvls = tree.level;
                        maxLvl = max(lvls);
                        
                        % INTERIOR NODES WITHOUT ROOT
                        for l = 1:maxLvl-1
                            indLvl = false(1,tree.nbNodes);
                            indLvl(lvls==l) = true;
                            indLvl = indLvl & ~tree.isLeaf;
                            nodLvl = tree.nodesIndices(indLvl);
                            for i = 1:numel(nodLvl)
                                nod = nodLvl(i);
                                ord = x.tensors{nod}.order;
                                chNb = tree.childNumber(nod);
                                x.tensors{nod} = timesMatrix(x.tensors{nod},Q{nod}',ord);
                                pnod = tree.parent(nod);
                                x.tensors{pnod} = timesMatrix(x.tensors{pnod},Q{nod}',chNb);
                            end
                        end
                        % LEAVES
                        for nod = tree.dim2ind
                            if x.isActiveNode(nod)
                                ord = x.tensors{nod}.order;
                                x.tensors{nod} = timesMatrix(x.tensors{nod},Q{nod}',ord);
                                pnod = tree.parent(nod);
                                chNb = tree.childNumber(nod);
                                x.tensors{pnod} = timesMatrix(x.tensors{pnod},Q{nod}',chNb);
                            end
                        end
                        % UPDATE SZ
                        y = updateProperties(x);
                        y.isOrth = false;
                        
                    case 2
                        x = orth(x);
                        localTolerance = t.tolerance/sqrt(nnz(x.isActiveNode)-1);
                        tree = x.tree;
                        lvls = tree.level ;
                        maxLvl = max(lvls);
                        Gtree = gramians(x);
                        for l = maxLvl:-1:1
                            nodLvl = intersect(nodesWithLevel(tree,l),x.activeNodes());
                            %Gl = gramians(x,nodLvl);
                            %if ~isa(Gl,'cell')
                            %    Gl={Gl};
                            %end
                            
                            for i = 1:numel(nodLvl)
                                nod = nodLvl(i);
                                
                                %G = Gl{i};
                                G = Gtree{nod};
                                %G = x.gramians(nod);
                                % truncation of the Gramian in trace norm for a control of frobenius norm of the tensor
                                t.maxRank = maxRank(nod);
                                q = t.truncSvd(G,localTolerance^2,1);
                                Q = q.space.spaces{1};
                                ord = x.tensors{nod}.order;
                                chNb = tree.childNumber(nod);
                                x.tensors{nod} = timesMatrix(x.tensors{nod},Q',ord);
                                pnod = tree.parent(nod);
                                x.tensors{pnod} = timesMatrix(x.tensors{pnod},Q',chNb);
                            end
                        end
                        
                        y = updateProperties(x);
                        y.isOrth = true;
                        y.orthNode = tree.root;
                        
                    otherwise
                        error('Wrong value for hsvdType.')
                end
            elseif isa(x,'TuckerLikeTensor') && isa(x.core,'TreeBasedTensor')
                y = t.truncate(treeBasedTensor(x,varargin{:}));
                y = TuckerLikeTensor(y);
                %                 maxRank=t.maxRank;
                %                 if numel(maxRank)==1
                %                     maxRank = repmat(maxRank,1,x.core.tree.nbNodes);
                %                     maxRank(1) = 1;
                %                 end
                %
                %                 x = orth(x);
                %                 d = x.order;
                %                 localTolerance = t.tolerance/sqrt(x.core.tree.nbNodes-1);
                %                 G = x.core.gramians();
                %                 yc = x.core;
                %                 ys = x.space;
                %                 Q = cell(size(G));
                %                 Q = Q';
                %                 sz = zeros(size(G));
                %                 for i = 1:numel(G)
                %                     % truncation of the Gramian in trace norm for a control of frobenius norm of the tensor
                %                     t.maxRank = maxRank(i);
                %                     q = t.truncSvd(G{i},localTolerance^2,1);
                %                     sz(i) = q.core.sz(1);
                %                     Q{i} = q.space.spaces{1};
                %                 end
                %                 tree = yc.tree;
                %                 lvls = tree.level;
                %                 maxLvl = max(lvls);
                %                 % LEAVES
                %                 ys = spaceTimesMatrix(ys,Q(tree.dim2ind));
                %                 for mu = tree.dim2ind
                %                     pnod = tree.parent(mu);
                %                     chNb = tree.childNumber(mu);
                %                     yc.tensors{pnod} = timesMatrix(yc.tensors{pnod},Q{mu}',chNb);
                %                 end
                %                 % INTERIOR NODES WITHOUT ROOT
                %                 for l = 1:maxLvl-1
                %                     indLvl = false(1,tree.nbNodes);
                %                     indLvl(lvls==l) = true;
                %                     indLvl = indLvl & ~tree.isLeaf;
                %                     nodLvl = tree.nodesIndices(indLvl);
                %                     for i = 1:numel(nodLvl)
                %                         nod = nodLvl(i);
                %                         ord = yc.tensors{nod}.order;
                %                         chNb = tree.childNumber(nod);
                %                         yc.tensors{nod} = timesMatrix(yc.tensors{nod},Q{nod}',ord);
                %                         pnod = tree.parent(nod);
                %                         yc.tensors{pnod} = timesMatrix(yc.tensors{pnod},Q{nod}',chNb);
                %                     end
                %                 end
                %                 % UPDATE SZ
                %                 yc = updateProperties(yc);
                %                 yc.isOrth = false;
                %                 y = TuckerLikeTensor(yc,ys);
            else
                error('Method not implemented.')
            end
        end
        
        function y = ttsvd(t,x,localTolerance)
            if isa(x,'TTTensor')
                maxRank=t.maxRank;
                if numel(maxRank)==1
                    maxRank = repmat(maxRank,1,x.order-1);
                end
                x = orth(x);
                d = x.order;
                if nargin == 2
                    localTolerance = t.tolerance/sqrt(d-1);
                end
                Y = x.cores;
                for mu = 1:d-1
                    sz = Y{mu}.sz;
                    Ymu = reshape(Y{mu},[sz(1)*sz(2),sz(3)]);
                    t.maxRank=maxRank(mu);
                    Ymu = t.truncSvd(Ymu.data,localTolerance);
                    V = Ymu.space.spaces{1};
                    W = Ymu.space.spaces{2}*diag(Ymu.core.data);
                    sz(3) = size(V,2);
                    Y{mu} = FullTensor(reshape(V,sz),3,sz);
                    Y{mu+1} = timesMatrix(Y{mu+1},{W'},1);
                end
                y = TTTensor(Y);
            elseif isa(x,'TuckerLikeTensor') && isa(x.core,'TTTensor')
                y = orth(x);
                d = x.order;
                if nargin == 2
                    localTolerance = t.tolerance/sqrt(2*d-1);
                end
                y.core = t.ttsvd(y.core);
                for mu = 1:d
                    y = orth(y);
                    y.core = orth(y.core,mu);
                    [y.core.cores{mu},R] = orth(y.core.cores{mu},2);
                    y.space = spaceTimesMatrix(y.space,R',mu);
                    ymu = t.truncSvd(y.space.spaces{mu},...
                        localTolerance);
                    y.space.spaces{mu} = ymu.space.spaces{1};
                    R = ymu.space.spaces{2}*diag(ymu.core.data);
                    y.core.cores{mu} = timesMatrix(y.core.cores{mu},R',2);
                    y.core.sz(mu) = y.core.cores{mu}.sz(2);
                end
                y = updateProperties(y);
            elseif  isa(x,'FullTensor')
                maxRank=t.maxRank;
                if numel(maxRank)==1
                    maxRank = repmat(maxRank,1,x.order-1);
                end
                d = x.order;
                if nargin == 2
                    localTolerance = t.tolerance/sqrt(d);
                end
                szx = x.sz;
                Y = cell(1,d);
                Z = x.data;
                sx = x.sz;
                r=1;
                for mu = 1:d-1
                    Z = reshape(Z,r*sx(mu),prod(sx(mu+1:end)));
                    t.maxRank=maxRank(mu);
                    Ymu = t.truncSvd(Z,localTolerance);
                    V = Ymu.space.spaces{1};
                    W = Ymu.space.spaces{2}*diag(Ymu.core.data);
                    rmu = size(V,2);
                    Y{mu} = FullTensor(V,3,[r,sx(mu),rmu]);
                    Z = W.';
                    r = rmu;
                end
                Y{d} = FullTensor(Z,3,[r,szx(d),1]);
                y = TTTensor(Y);
            else
                error('Method not implemented.')
            end
        end
    end
    
    methods (Static)
        function sv = getSingularValues(x)
            warning('Will be removed in a future release, use singularValues(x) instead.')
            sv = singularValues(x);
        end
    end
end
