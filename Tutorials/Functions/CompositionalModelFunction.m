% Class CompositionalModelFunction

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

classdef CompositionalModelFunction < Function
   
    properties
        % TREE - DimensionTree
        tree
        % FUN - Cell of functions
        fun
    end
    
    methods
        
        function f = CompositionalModelFunction(tree,fun,measure)
            % COMPOSITIONALMODELFUNCTION - Constructor for the class CompositionalModelFunction
            %
            % s = COMPOSITIONALMODELFUNCTION(tree,fun,measure)
            % tree: DimensionTree
            % fun: cell of functions
            % measure: Measure
            
            f.tree = tree;
            
            if ~isa(fun,'cell')
                f.fun = cell(1,tree.nbNodes);
                f.fun(tree.internalNodes) = {fun};
            else
                f.fun = fun;
            end
            
            f.dim = length(tree.dim2ind);
            f.measure = measure;
        end
        
        function y = eval(f,x)
            % EVAL - Function evaluation
            %
            % y = EVAL(f,x)
            % f: CompositionalModelFunction
            % x: n-by-f.dim double
            % y: n-by-1 double
            
            z = cell(1,f.tree.nbNodes);
            for nu = 1:f.dim
                z{f.tree.dim2ind(nu)} = x(:,nu);
            end
            for l=max(f.tree.level)-1:-1:0
                nodewithlevell = find(f.tree.level==l);
                for nod = setdiff(nodewithlevell,f.tree.dim2ind)
                    ch = nonzeros(f.tree.children(:,nod));
                    z{nod} = f.fun{nod}(z{ch});
                end
            end
            y = z{f.tree.root};
        end
    end
end