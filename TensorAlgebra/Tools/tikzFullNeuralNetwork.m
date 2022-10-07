% tikzTreeBasedTensorNeuralNetwork - Generates the tikz code to represent a DimensionTree with labels at its nodes given by nodeLabels
%
% h = tikzTreeBasedTensorNeuralNetwork(x)
% x: TreeBasedTensor
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

function h = tikzFullNeuralNetwork(network)
% function h = tikzTreeBasedTensorNeuralNetwork(network)
if nargin==1
    inputLayer=1;
else
    inputLayer=double(inputLayer);
end


h = '\begin{tikzpicture}[xscale=15,yscale=10]  ';
h = [h,' \tikzstyle{neuron}=[circle,ball color = red,scale=0.2] '];
h = [h,' \tikzstyle{featureneuron}=[circle,ball color = yellow,scale=0.2] '];
h = [h,'  \tikzstyle{inputneuron}=[circle,ball color = blue,scale=0.2] '];
h = [h,'  \tikzstyle{outputneuron}=[circle,ball color = green,scale=0.2]  '];
h = [h,'\tikzstyle{connexion}=[draw,thick,fill=black,scale=0.1]  '];
h = [h,'\tikzset{every edge/.style={draw, gray,scale=0.1}}  '];

h = [h,'\draw '];
for k=1:network.nbNeurons
h = [h,'(' num2str(output.xn(k)) ',' num2str(output.yn(k)) ') '];
if inputLayer==1 && network.neuronsLayer(k)==2
h = [h,'node[featureneuron] (' num2str(k) ') {} '];
elseif network.neuronsLayer(k)==1
h = [h,'node[inputneuron] (' num2str(k) ') {} '];
elseif network.neuronsLayer(k)==max(network.neuronsLayer)
h = [h,'node[outputneuron] (' num2str(k) ') {} '];
else
h = [h,'node[neuron] (' num2str(k) ') {} '];
end

end
output.edges 
h = [h,';'];
edges = output.edges ;
for e=1:size(edges,1)
h = [h,'\path (' num2str(edges(e,1)) ') edge (' num2str(edges(e,2)) ');'];
end

h = [h,'\end{tikzpicture}'];
end
