% Class DimensionTree

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

classdef DimensionTree
    
    properties
        adjacencyMatrix     % logical matrix. The nth row indicates the sons of nth node. The nth column indicates the parents of nth node
        nodesParentOfLeaves % indices of nodes which are parents of leaves
        internalNodes       % indices of internal (non leaf) nodes
        nbNodes             % number of nodes
        arity               % maximal number of children
        dims                % dims{n} is the set of dimensions associated with node n
        dim2ind             % dim2ind(k) is the index of the node (leaf) corresponding to dimension k
        parent              % parent(n) is the index of the parent of the nth node
        children            % children(:,n) is the set of indices of the children of nth node
        childNumber         % childNumber(n)=k means that nth node is the kth child of the parent of nth node
        sibling             % sibling(:,n) contains the indices of the children of the parent of nth node
        level               % level(n) is the level of the nth node
        root                % index of the root node
        isLeaf              % 1-by-nbNodes logical array with entry 1 if the corresponding node is a leaf and 0 otherwise
        plotOptions = struct('levelAlignment',false)
    end
    
    properties (Hidden)
        nodesIndices
    end
    
    
    
    methods
        
        function T = DimensionTree(dim2ind,adjacencyMatrix)
            % Constructor of the class DimensionTree
            %
            % T = DimensionTree(dim2ind,adjacencyMatrix)
            % Creates a dimension partition tree over D = {1,...,d} from an adjacency matrix
            %
            % adjacencyMatrix: n-by-n array (for a tree with n nodes)
            % dim2ind: array of length d given the indices of the nodes associated with the d dimensions
            
            T.dim2ind = dim2ind(:)';
            T.adjacencyMatrix = adjacencyMatrix;
            T = precomputeProperties(T);
            T = updateDimsFromLeaves(T);
        end
        
        function T = permute(T,sigma)
            T.dim2ind(sigma)=T.dim2ind;
            T = updateDimsFromLeaves(T);
        end
        
        function T = ipermute(T,sigma)
            T.dim2ind = T.dim2ind(sigma);
            T = updateDimsFromLeaves(T);
        end
        
        function ok=eq(T1,T2)
            % Compares two dimension trees
            % Returns a logical 1 if both trees are the same, and 0 otherwise
            
            ok = (length(T1.dim2ind) == length(T2.dim2ind)) & (T1.nbNodes == T2.nbNodes) & (max(T1.level) == max(T2.level));
            if ~ok
                return
            end
            for l=1:max(T1.level)
                n1 =  nodesWithLevel(T1,l);
                n2 =  nodesWithLevel(T2,l);
                if length(n1)~=length(n2)
                    ok=false;
                    return
                end
                dims1 = T1.dims(n1);
                dims2 = T2.dims(n2);
                for i=1:length(dims1)
                    for j=1:length(dims2)
                        if length(dims1{i})==length(dims2{j}) && all(sort(dims1{i})==sort(dims2{j}))
                            n1(i)=0;
                            break
                        end
                    end
                end
                if any(n1~=0)
                    ok=false;
                    return
                end
            end
        end
        
        function T = updateDimsFromLeaves(T)
            % Updates the dimensions of all nodes from the dimensions of the leaves given in T.dim2ind
            
            T.dims = cell(1,T.nbNodes);
            T.dims(T.dim2ind) = num2cell(1:numel(T.dim2ind));
            
            maxLvl = max(T.level);
            for l = maxLvl-1:-1:0
                nodLvl = nodesWithLevel(T,l);
                for nod = nodLvl
                    if ~T.isLeaf(nod)
                        c = nonzeros(T.children(:,nod));
                        dimi = [T.dims{c}];
                        T.dims{nod} = dimi;
                    end
                end
            end
        end
        
        function n = nodesWithLevel(T,l)
            % Returns the indices of nodes at a given level
            %
            % n = nodesWithLevel(T,l)
            % T: DimensionTree
            % l: integer from 0 (root node) to the depth of the tree
            
            indLvl = false(1,T.nbNodes);
            indLvl(T.level==l) = true;
            n = find(indLvl);
        end
        
        
        function [x,y] =  treelayout(T)
            % Lay out tree
            %
            % function [x,y] =  treelayout(T)
            % T : DimensionTree
            % If T.plotOptions.levelAlignment is
            % false, then classical matlab treelayout
            % true : align nodes  with same level
            %
            % See also treelayout

            [x,y] =  treelayout(T.parent);
            if T.plotOptions.levelAlignment
                H = max(y)-min(y);
                for alpha=1:T.nbNodes
                    l = T.level(alpha);
                    y(alpha)=max(y) - l*H/max(T.level);
                end
            end
        end
        
        function varargout=plot(T)
            % Plots the tree with nodes indices
            %
            % function plot(T)
            
            varargout = cell(1,nargout);
            [varargout{:}]=plotWithLabelsAtNodes(T,1:T.nbNodes);
        end
        
        function varargout = plotNodes(T,nodes,varargin)
            % Plots the nodes of the tree
            %
            % [x,y,H] = plotNodes(T,nodes,varargin)
            % plot nodes with plot properties specified by varargin
            % nodes: array of integers (1:T.nbNodes if nodes=[] or nargin==1)
            % x,y: coordinates of the nodes
            % H: handle
            
            if nargin==1 || isempty(nodes)
                nodes = 1:T.nbNodes;
            end
            
            [x,y] = treelayout(T);
            x=x';
            y=y';
            
            if nargin<=2
                varargin={'ko'};
            end
            
            H = plot(x(nodes),y(nodes),varargin{:});
            
            axis([0 1 0 1]);
            
            if nargout>0
                varargout{1}=x;
                varargout{2}=y;
                varargout{3}=H;
            end
        end
        
        function varargout = plotEdges(T,edges,varargin)
            % Plots the edges of the tree
            %
            % [H,xs,ys,xp,yp] = plotEdges(T,edges,varargin)
            % plot edges with plot properties specified by varargin
            % the number of an edge is the number of the node with maximum level connected to this edge.
            % edges: array of integers (setdiff(1:T.nbNodes,T.root) if edges=[] or nargin==1)
            % xs,ys: coordinates of child nodes
            % sp,yp: coordinates of parent nodes
            % H: handle
            
            if nargin==1 || isempty(edges)
                edges = setdiff(1:T.nbNodes,T.root);
            end
            
            if nargin<3
                varargin = {'k-'};
            end
            
            [x,y] = treelayout(T);
            xp = x(T.parent(edges));
            xs = x(edges);
            yp = y(T.parent(edges));
            ys = y(edges);
            H = plot([xs;xp],[ys;yp],varargin{:});
            if nargout>0
                varargout{1} = H;
                varargout{2} = xs;
                varargout{3} = ys;
                varargout{4} = xp;
                varargout{5} = yp;
            end
        end
        
        function varargout = plotLabelsAtNodes(T,L,nodes,varargin)
            % Plots labels at nodes
            %
            % H = plotLabelsAtNodes(T,Labels,listOfNodes)
            % plot labels for a list of nodes
            % Labels: cell of char of size 1-by-length(listOfNodes)
            % listOfNodes: array of integers (by default 1:T.nbNodes)
            % H: handle
            
            if isa(L,'logical')
                L = double(L);
            end
            
            if isa(L,'numeric')
                L = num2cell(L);
            end
            
            if nargin==2 || isempty(nodes)
                nodes = 1:T.nbNodes;
            end
            
            [x,y] = treelayout(T);
            x = x';
            y = y';
            shift = (max(y)-min(y))/max(T.level)/8;
            leafnodes = nodes(T.isLeaf(nodes));
            H = [];
            if ~isempty(leafnodes)
                Ltemp = L(T.isLeaf(nodes));
                H1 = text(x(leafnodes), y(leafnodes)-shift, Ltemp(:), 'VerticalAlignment','top','HorizontalAlignment','center',varargin{:});
                H =[H;H1];
            end
            intnodes = nodes(~T.isLeaf(nodes));
            if ~isempty(intnodes)
                Ltemp = L(~T.isLeaf(nodes));
                H2 = text(x(intnodes), y(intnodes)+shift, Ltemp(:), 'VerticalAlignment','bottom','HorizontalAlignment','center',varargin{:});
                H =[H;H2];
            end
            
            if nargout>0
                varargout{1}=H;
            end
        end
        
        function varargout = plotWithLabelsAtNodes(T,L,varargin)
            % Plots the tree with labels at nodes
            %
            % [N,E,H] = plotWithLabelsAtNodes(T,L,nodes)
            % plot the tree T with labels at selected nodes
            % nodes: list of nodes (1:T.nbNodes by default)
            % N, E, H: handles
            
            [~,~,N] = plotNodes(T);
            hold on
            E = plotEdges(T);
            H = plotLabelsAtNodes(T,L,varargin{:});
            hold off
            
            if nargout>0
                varargout{1}=N;
                varargout{2}=E;
                varargout{3}=H;
            end
        end
        
        function varargout = plotDims(T,I,varargin)
            % Plots the dimensions associated with the nodes of the tree
            %
            % [N,E,H] = plotDims(T)
            % plot the tree T and the dimensions associated with each leaf node of the tree
            % N, E, H: handles for nodes, edges and text
            %
            % [N,E,H] = plotDims(T,I)
            % plot the dimensions associated with the nodes listed in I
            % I: array if integer
            
            if nargin==1
                I = T.dim2ind;
            end
            
            varargout = cell(1,nargout);
            L = cellfun(@(x) [ '\{' num2str(x) '\}' ],T.dims(I),'uniformoutput',false);
            [varargout{:}] = plotWithLabelsAtNodes(T,L,I,varargin{:});
        end
        
        function varargout = plotDimsNodesIndices(T,varargin)
            % Plots the tree T with nodes indices and dimensions associated with the nodes
            %
            % [N,E,H] = plotDimsNodesIndices(T))
            % Plots the tree T with nodes indices and set of dimensions associated with each node
            % N, E, H: handles for nodes, edges and text
            
            varargout = cell(1,nargout);
            L1 = cellstr(num2str((1:T.nbNodes)'));
            L = cellfun(@(x,y) [y ' \{' num2str(x) '\}' ],T.dims(:),L1(:),'uniformoutput',false);
            [varargout{:}] = plotWithLabelsAtNodes(T,L,varargin{:});
        end
        
        function varargout = plotNodesIndices(T)
            % Plots the nodes indices
            %
            % [N,E,H] = plotNodesIndices(T)
            % Plots the tree T with nodes indices
            % N, E, H: handles for nodes, edges and text
            
            varargout = cell(1,nargout);
            L = cellstr(num2str((1:T.nbNodes)'));
            [varargout{:}] = plotWithLabelsAtNodes(T,L);
        end
        
        function anod = ascendants(T,nod)
            % Returns the ascendants of a given node
            %
            % anod = ascendants(T,k)
            % Returns the indices of the ascendants of the node with index k
            % T: DimensionTree
            % nod: integer
            % anod: array
            
            anod = [];
            pnod = T.parent(nod);
            while pnod~=0
                anod = [anod,pnod];
                pnod = T.parent(pnod);
            end
        end
        
        function dnod = descendants(T,nod)
            % Returns the descendants of a given node
            %
            % anod = descendants(T,k)
            % Returns the indices of the descendants of the node with index k
            % T: DimensionTree
            % nod: integer
            % anod: array
            
            dnod = [];
            chnod = nonzeros(T.children(:,nod));
            if ~isempty(chnod)
                for ch = chnod(:)'
                    dnod=[dnod,ch];
                    dnod = [dnod,descendants(T,ch)];
                end
            end
        end
        
        function p = path(T,a,b)
            % Returns the path between two nodes a and b
            % 
            % function p = path(T,a,b)
            % T : DimensionTree
            % a,b : integer
            % p : array of integers
            
            pa = ascendants(t,a);
            pb = ascendants(t,b);
            commonAs = fastIntersect(pa,pb);
            gamma = commonAs(T.level(commonAs) == max(T.level(commonAs)));
            pa = fastSetdiff(pa,commonAs);
            pb = fastSetdiff(pb,commonAs);
            p = [pa,gamma,flip(pb)];
        end
        
        
        function [subT,nod] = subDimensionTree(T,r)
            % Extracts a sub dimension tree
            %
            % [subT,nod] = subDimensionTree(T,r)
            % extracts a subtree subT from T
            % r is the index of the node which is the root of subT
            %
            % T: DimensionTree
            % r: integer
            % subT: DimensionTree
            % nod: extracted nodes from T
            %
            % the property dim2ind of subT gives the nodes indices in subT corresponding to dimensions in T.dims{r} (not sorted)
            
            dims = T.dims{r};
            nod = [r,descendants(T,r)];
            a = T.adjacencyMatrix(nod,nod);
            [~,dim2ind] = ismember(T.dim2ind(dims),nod);
            subT = DimensionTree(dim2ind,a);
        end
        
        function nod = nodeWithDims(T,dims)
            % NODEWITHDIMS - Return the index of the node containing the given set of dimensions.
            %
            % Returns the index of the node corresponding to dimensions dims or an empty array if no node corresponds to these dimensions.
            %
            % nod = NODEWITHDIMS(T,dims)
            % 
            % T: DimensionTree
            % dims: array
            % nod: double

            dims = sort(dims);
            nod = find(cellfun(@(x) isequal(sort(x),dims),T.dims));
        end
    end
    
    
    methods(Static)
        
        function tree = trivial(d)
            % Creates a dimension tree with one level
            %
            % tree = trivial(d)
            
            d2i = 2:d+1;
            
            adjMat = false(d+1);
            adjMat(1,2:end)=true;
            
            tree = DimensionTree(d2i,adjMat);
        end
        
        function tree = linear(ord)
            % Creates a linear dimension tree
            %
            % tree = linear(dims)
            % Creates a linear tree
            % dims: permutation of 1:n (numbering of leaves) or integer (number of leaves, ordered as 1:dims)
            
            if length(ord)==1
                d = ord;
                ord = 1:ord;
            else
                d=length(ord);
            end
            d2i = [2*d-2,2*(d-1:-1:1)+1];
            adjMat = false(2*d-1);
            adjMat(1,2:3)=true;
            for l = 1:d-2
                adjMat(2*l,2*l+2) = true;
                adjMat(2*l,2*l+3) = true;
            end
            d2i(ord)=d2i;
            tree= DimensionTree(d2i,adjMat);
        end
        
        function tree = balanced(ord)
            % Creates a balanced dimension tree
            %
            % tree = balanced(dims)
            % creates a balanced tree
            % dims: permutation of 1:n (numbering of leaves) or integer (number of leaves, ordered as 1:dims)
            
            if length(ord)==1
                d = ord;
                ord = 1:ord;
            else
                d=length(ord);
            end
            n = 2*d-1;
            d2i = d:(2*d-1);
            adjMat = false(n);
            adjMat(1,2:3) = true;
            for mu = 2:(d-1)
                adjMat(mu,(2*mu):(2*mu+1)) = true;
            end
            d2i(ord)=d2i;
            tree= DimensionTree(d2i,adjMat);
        end
        
        function tree=random(d,arity)
            % Creates a random dimension tree
            %
            % tree=random(d)
            % Creates a random dimension tree over {1,...,d} with arity 2
            %
            % tree=random(d,a)
            % Creates a random dimension tree over {1,...,d} with arity a if a is an integer.
            % If a is an interval [amin,amin], then the number of children of a node is randomly drawn from the uniform distribution over {amin,...,amax}.
            
            if nargin==1
                arity=2;
            end
            if length(arity)==1
                arity = [arity,arity];
            end
            dimsNodes = {1:d};
            nbNodes = 1;
            newNodes = 1;
            adjMat = false(d*2);
            dim2ind = zeros(1,d);
            while ~isempty(newNodes)
                parentNodes = newNodes;
                newNodes = [];
                for i=1:length(parentNodes)
                    pnod = parentNodes(i);
                    dims = dimsNodes{pnod};
                    if length(dims)==1
                        dim2ind(dims)=pnod;
                    else
                        a = min(randi(arity),length(dims));
                        for k=1:a
                            nbNodes = nbNodes+1;
                            newNodes = [newNodes,nbNodes];
                            adjMat(pnod,nbNodes)=true;
                            if k==a
                                dimsNodes{nbNodes} = dims;
                                dims=[];
                            else
                                n  = max(randi(length(dims)-a+k),1);
                                p = randperm(length(dims),n);
                                dimsNodes{nbNodes} = dims(p);
                                dims(p) = [];
                            end
                            if isempty(dims)
                                break
                            end
                        end
                    end
                end
            end
            adjMat = adjMat(1:nbNodes,1:nbNodes);
            tree= DimensionTree(dim2ind,adjMat);
        end
        
        function tree=randomBalanced(d,arity)
            % Creates a random balanced dimension tree
            %
            % tree=random(d)
            % Creates a random balanced dimension tree over {1,...,d} with arity 2
            %
            % tree=random(d,a)
            % Creates a random balanced dimension tree over {1,...,d}  with arity a if a is an integer.
            % If a is an interval [amin,amin], then the number of children of a node is randomly drawn from the uniform distribution over {amin,...,amax}.
            
            adjMat = false(d*2);
            nbNodes = d;
            nextNodes = 1:d;
            if nargin==1
                arity=2;
            end
            if length(arity)==1
                arity = [arity,arity];
            end
            while length(nextNodes)>1
                remainingNodes = nextNodes;
                nextNodes = [];
                while ~isempty(remainingNodes)
                    if length(remainingNodes)==1
                        nextNodes = [nextNodes,remainingNodes];
                        remainingNodes=[];
                    else
                        a = randi(arity);
                        a = min(a,length(remainingNodes));
                        p = randperm(length(remainingNodes),a);
                        nbNodes = nbNodes+1;
                        nextNodes = [nextNodes,nbNodes];
                        adjMat(nbNodes,remainingNodes(p))=true;
                        remainingNodes(p) = [];
                    end
                end
            end
            adjMat = adjMat(1:nbNodes,1:nbNodes);
            tree= DimensionTree(1:d,adjMat);
        end
        
    end
    
    methods (Hidden)
        function T = precomputeProperties(T)
            % Precomputes properties of the dimension tree from properties adjacencyMatrix and dim2ind
            
            % nbNodes
            T.nbNodes = size(T.adjacencyMatrix,1);
            
            % nodes Indices
            T.nodesIndices = 1:T.nbNodes;
            
            % arity
            T.arity = max(sum(T.adjacencyMatrix,2));
            
            % parent
            T.parent = ((1:T.nbNodes)*T.adjacencyMatrix);
            
            %root
            T.root = find(T.parent == 0);
            
            % isLeaf
            T.isLeaf = false(1,T.nbNodes);
            T.isLeaf(T.dim2ind) = true;
            
            % children
            T.children = zeros(T.arity,T.nbNodes);
            for i = 1:T.nbNodes
                c = find(T.adjacencyMatrix(i,:));
                T.children(1:numel(c),i) = c; % !!! sort the children
            end
            
            % childNumber
            T.childNumber = zeros(1,T.nbNodes);
            for i = 1:T.nbNodes
                if T.parent(i) ~= 0
                    c = T.children(:,T.parent(i));
                    j = find(c == i);
                    if ~isempty(j)
                        T.childNumber(i) = j;
                    end
                end
            end
            
            % sibling
            T.sibling = zeros(T.arity,T.nbNodes);
            for i = 1:T.nbNodes
                if T.parent(i) ~= 0
                    T.sibling(:,i) = T.children(:,T.parent(i));
                end
            end
            
            % level
            T.level = zeros(1,T.nbNodes);
            c = T.children(:,T.root);
            c(c == 0) = [];
            l = 1;
            while(~isempty(c))
                T.level(c) = l;
                c = T.children(:,c);
                c = c(:)';
                l = l + 1;
                c(c == 0) = [];
            end
            
            % internalNodes
            T.internalNodes = setdiff(1:T.nbNodes,T.dim2ind);
            
            % nodesParentOfLeaves
            T.nodesParentOfLeaves = unique(T.parent(T.dim2ind));
        end
    end
end