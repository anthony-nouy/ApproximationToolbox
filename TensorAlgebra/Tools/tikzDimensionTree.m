% TIKZDIMENSIONTREE - Generates the tikz code to represent a DimensionTree with labels at its nodes given by nodeLabels
%
% h = TIKZDIMENSIONTREE(T,nodeLabels)
% T: DimensionTree
% nodeLabels: 1-by-T.nbNodes cell
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

function h = tikzDimensionTree(T,nodeLabels)
if nargin==1
    nodeLabels = cell(1,T.nbNodes);
    nodeLabels(:)={''};
    for k=1:length(T.dim2ind)
        nodeLabels{T.dim2ind(k)} = ['$\{' num2str(k) '\}$'];
    end
elseif isnumeric(nodeLabels)
    nodeLabels = num2cell(nodeLabels);
    nodeLabels = cellfun(@num2str,nodeLabels,'uniformoutput',false);
elseif ~isa(nodeLabels,'cell')
    error('wrong argument nodeLabels')
end
h = '\begin{tikzpicture}[scale=0.4]  ';
L = max(T.level);
sz = zeros(1,L);sz(L)=20;
for i=L-1:-1:1
    sz(i) = sz(i+1)+30;
end
for i=1:L
    h = [h,'\tikzstyle{level ',num2str(i),  '}=[sibling distance=', num2str(sz(i)),'mm]  '];
end
h = [h,'\tikzstyle{root}=[circle,draw,thick,fill=red]  '];
h = [h,'\tikzstyle{interior}=[circle,draw,solid,thick,fill=red]  '];
h = [h,'\tikzstyle{leaves}=[circle,draw,solid,thick,fill=red]  '];
h = [h,'\tikzstyle{active}=[circle,draw,solid,thick,fill=red]  '];

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