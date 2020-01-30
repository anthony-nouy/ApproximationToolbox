% Class GraphTensor

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

classdef GraphTensor

    properties
        G               % graph
        tensors         % cell containing objects of type FullTensor
        order           % order of the tensor
        sz              % size of the tensor
        cliques
        isOrth = false; % The flag is false if the representation of the tensor is orthogonal (i.e. one mu-matricization is orthogonal)
    end
    
    
    methods
        function x = GraphTensor(G,tensors,order,sz)
            % GraphTensor - Constructor for the class GraphTensor
            %
            % x = GraphTensor(G,tensors,order,sz)
            % G: graph
            % tensors: cell array of length the number of maximal cliques containing FullTensor
            % order: order of the tensor
            % sz: size of the tensor, array of size 1xorder
            % x: GraphTensor
            
            if ~isa(G,'graph')
                error('Must provide a graph.')
            end
            x.G = G;
            x.order = order;
            x.sz = sz(:).';
            x.tensors = tensors;
            x.cliques = maximalCliques(G.adjacency);
        end
        
        function n = storage(x)
            n = sum(cellfun(@storage, x.tensors));
        end
        
        function n = sparseStorage(x)
            n = sum(cellfun(@sparseStorage, x.tensors));
        end
        
        function y = evalAtIndices(x,I,dims)
            if isa(I,'MultiIndices')
                I = I.array;
            end
            if nargin==2
                dims=1:x.order;
            else
                [dims,isort] = sort(dims);
                I = I(:,isort) ;
                assert(numel(dims)==size(I,2),'Wrong arguments.')
                error('Method not implemented.')
            end
            
            y = ones(size(x,1),1);
            for k = 1:size(x.cliques,2)
                c = x.cliques(:,k);
                y = y.*evalAtIndices(x.tensors{k},I(:,c));
            end
        end
        
        function d = ndims(GM)
            d = length(GM.sz);
        end
        
        function y = full(x)
            d = x.order;
            y = FullTensor.ones(x.sz);
            for k = 1:size(x.cliques,2)
                t = x.tensors{k};
                c = find(x.cliques(:,k))';
                nc = setdiff(1:d,c);
                g = timesTensor(t,FullTensor.ones(x.sz(nc)),[],[]);
                g = ipermute(g,[c,nc]);
                y = y.*g;
            end
        end
    end
    
    methods (Static)
        function x = create(gen,G,sz)
            x = GraphTensor(G,{},length(sz),sz);
            for k = 1:size(x.cliques,2)
                c = x.cliques(:,k);
                x.tensors{k} = FullTensor.create(gen,sz(c));
            end
        end
        
        function x = rand(varargin)
            % x = rand(G,sz)
            % G: graph
            % sz: 1-by-d array containing the size of the tensor
            
            x = GraphTensor.create(@rand,varargin{:});
        end
    end
end