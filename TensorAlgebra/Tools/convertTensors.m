% CONVERTTENSORS - Tensor conversion
%
% [x,y] = CONVERTTENSORS(x,y)

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

function [x,y] = convertTensors(x,y)
if ~isa(x,class(y))
    if isa(x,'FullTensor')
        y = full(y);
    elseif isa(y,'FullTensor')
        x = full(x);
    elseif isa(x,'DiagonalTensor') && isa(y,'TTTensor')
        x = TTTensor(x);
    elseif isa(y,'DiagonalTensor') && isa(x,'TTTensor')
        y = TTTensor(y);
    elseif isa(y,'TuckerLikeTensor') && isa(x,'TTTensor')
        [x,yc] = convertTensors(x,y.core);
        y = timesMatrix(yc,y.space.spaces);
        y=updateProperties(y);
    elseif isa(x,'TuckerLikeTensor') && isa(y,'TTTensor')
        [xc,y] = convertTensors(x.core,y);
        x = timesMatrix(xc,x.space.spaces);
        x=updateProperties(x);
    elseif isa(x,'DiagonalTensor') && isa(y,'TreeBasedTensor')
        x = treeBasedTensor(x,y.tree);
    elseif isa(y,'DiagonalTensor') && isa(x,'TreeBasedTensor')
        y = treeBasedTensor(y,x.tree);
    elseif isa(x,'TuckerLikeTensor') && isa(y,'CanonicalTensor')
        y = TuckerLikeTensor(y.core,y.space) ;
        [x.core,y.core] = convertTensors(x.core,y.core);
    elseif isa(x,'SparseTensor') && isa(y,'DiagonalTensor')
        y = sparse(y) ;
    else
        error('Method not implemented.')
    end
elseif isa(x,'TuckerLikeTensor')
    if ~isa(y,class(x)) % i.e. class(x) < class(y)
        [y,x] = convertTensors(y,x) ;
    else
        [x.core,y.core] = convertTensors(x.core,y.core);
    end
end
end