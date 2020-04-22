% Class DimensionTreeOptimizer

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

classdef DimensionTreeOptimizer
    
    properties
        cost    % function taking as argument a DimensionTree and returning the cost to minimize
        pA = @(t,k) 1   % pA(t,k) function taking as argument a dimension tree t and a node index k used to define a probability pA(T,k) for selecting the node k in t
        pB = @(t,k,j) 1 % pB(t,k,j) function taking as argument a dimension tree t and two nodes indices k and j  used to define a conditional probability pB(t,k,j) for selecting the node j knowing k
        acceptRate = 0.2 % acceptation rate (between 0 and 1)
        nbIter = 100 % number of iterations
        store = false % store the trees
    end
    
    methods
        
        function [tstar,results] = optimize(s,t0)
            % [t,results] = optimize(s,t0)
            % s : DimensionTreeOptimizer
            % t0 : initial DimensioTree
            
            C0 = s.cost(t0);
            tstar = t0;
            Cstar = C0;
            if s.store
                results.trees{1} = t0;
            end
            
            for iter = 1:s.nbIter
                
                probaA = s.computepA(t0);
                a = random(DiscreteRandomVariable(1:t0.nbNodes,probaA));
                probaB = s.computepB(t0,a);
                if ~isempty(probaB)
                    b = random(DiscreteRandomVariable(1:t0.nbNodes,probaB));
                    
                    t=t0;
                    t.adjacencyMatrix(t.parent(a),a) = 0;
                    t.adjacencyMatrix(t.parent(a),b) = 1;
                    t.adjacencyMatrix(t.parent(b),b) = 0;
                    t.adjacencyMatrix(t.parent(b),a) = 1;
                    t = precomputeProperties(t);
                    t = updateDimsFromLeaves(t);
                    
                    try
                        plot(t)
                    catch
                        keyboard
                    end
                    C = s.cost(t);
                    if C<C0
                        tstar = t;
                        Cstar = C;
                    end
                    
                    if (C<C0) || (rand()<=s.acceptRate)
                        t0=t;
                        C0=C;
                        
                        if s.store
                            results.trees{end+1} = t;
                        end
                    end
                end
            end
            
            results.C = Cstar;
            
        end
        
        
        function pA = computepA(s,t)
            pA = zeros(1,t.nbNodes);
            nodes = fastSetdiff(1:t.nbNodes,t.root);
            for k=nodes
                pA(k) = s.pA(t,k);
            end
            pA(t.root)=0;
            if sum(pA)==0
                pA = ones(1,t.nbNodes);
                pA(t.root)=0;
            end
            pA = pA/sum(pA);
        end
        
        
        function pB = computepB(s,t,k)
            
            pB = zeros(1,t.nbNodes);
            nodes = fastSetdiff(1:t.nbNodes,t.root);
            for j=nodes
                if ~(ismember(j,t.ascendants(k)) || ...
                        ismember(j,t.descendants(k)) || ...
                        t.parent(k) == t.parent(j))
                    pB(j) = s.pB(t,k,j);
                end
            end
            pB(t.root) = 0;
            if sum(pB)==0
                pB=[];
            else
                pB = pB/sum(pB);
            end
            
        end
        
        
    end
    
    
    methods (Static)
        
        function x = applyFunDims(t,fun)
            x = cellfun(fun,t.dims);
        end
            
        function c = tensorStorageCost(t,r,n)
            % function tensorStorageCost(t,r,n)
            % t: DimensionTree
            % r: ranks bounds
            % n : dimensions of the tensor
            if isa(r,'function_handle')
                r = cellfun(r,t.dims);
            end
            if nargin==2
                n = r(t.dim2ind);
            end
            c = ones(size(t.children,1)+1,t.nbNodes);
            c(1,:) = r;
            c(2:end,~t.isLeaf) = r(t.children(:,~t.isLeaf));
            c(2,t.dim2ind) = n;
            c = prod(c,1);
            c = sum(c);
        end
        
    end
end