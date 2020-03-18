% TIKZDIMENSIONTREE - Generates the tikz code to represent a DimensionTree with labels at its nodes given by nodeLabels
%
% h = TIKZDIMENSIONTREE(T,nodeLabels,plotDims,plotDimsLeaves)
% T: DimensionTree
% nodeLabels: 1-by-T.nbNodes cell
% plotDims,plotDimsLeaves : boolean (false by default)
% h: char

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

function h = tikzDimensionTree(T,nodeLabels,plotDims,plotDimsLeaves)
% function h = tikzDimensionTree(T,nodeLabels,plotDims,plotDimsLeaves)
% creating a script for plotting a dimension tree in latex with tikz
% T : DimensionTree
% nodeLabels : labels at nodes (array or cell array with length T.nbNodes)
% plotDims : boolean, true for plotting the dimensions of the nodes (false by default)
% plotDimsLeaves : boolean, true for plotting the dimensions of the leaf nodes
%   (false by default, true if plotDims is true)

if nargin==1 || isempty(nodeLabels)
    nodeLabels = cell(1,T.nbNodes);
    nodeLabels(:)={''};
end
if nargin<3
    plotDimsLeaves = false;
    plotDims = false;
end
if nargin<4
    plotDimsLeaves=plotDims;
end
if isnumeric(nodeLabels)
    nodeLabels = num2cell(nodeLabels);
    nodeLabels = cellfun(@num2str,nodeLabels,'uniformoutput',false);
elseif ~isa(nodeLabels,'cell')
    error('wrong argument nodeLabels')
end

for k=1:T.nbNodes
    if plotDims || (plotDimsLeaves && T.isLeaf(k))
        temp = nodeLabels{k};
        dims = sort(T.dims{k});
        temp = [temp '$\{'];
        for l=1:length(dims)
            if l>1
                temp = [temp  ','];
            end
        temp = [temp  num2str(dims(l))];
        end
        nodeLabels{k} = [temp '\}$'];                
    end
end

h = '\begin{tikzpicture}[scale=1, level distance = 20mm]  ';
L = max(T.level);
sz = zeros(1,L);
sz0 = 150;
for i=1:L
    sz(i) = ceil(sz0/T.arity/1.);
    sz0 = sz(i);
end
for i=1:L
    h = [h,'\tikzstyle{level ',num2str(i),  '}=[sibling distance=', num2str(sz(i)),'mm]  '];
end
h = [h,'\tikzstyle{root}=[circle,draw,thick,fill=black,scale=.8]  '];
h = [h,'\tikzstyle{interior}=[circle,draw,solid,thick,fill=black,scale=.8]  '];
h = [h,'\tikzstyle{leaves}=[circle,draw,solid,thick,fill=black,scale=.8]  '];

h = add(h,T,T.root,nodeLabels);

h = [h,';\end{tikzpicture}'];
end

function h = add(h,T,nod,nodeLabels)
type = {'root','interior','leaves','active'};
chnod = nonzeros(T.children(:,nod));

if nod==T.root
    h = [h,'\node [',type{1},',label=above:{',nodeLabels{nod},'}]  {} '];
elseif isempty(chnod)
    h = [h,'node [' type{3} ',label=below:{',nodeLabels{nod},'}] {}'];
else
    h = [h,'node [' type{2} ',label=above:{',nodeLabels{nod},'}] {}'];
end

for i=1:length(chnod)
    h = [h,'child{'];
    h = add(h,T,chnod(i),nodeLabels);
    h = [h,'}'];
end
end