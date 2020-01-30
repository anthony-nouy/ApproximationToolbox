% Class Measure

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

classdef (Abstract) Measure
    
    methods (Abstract)
        ok = eq(p,q)
        % ok = eq(p,q)
        % Checks if the two Measure p and q are identical
        % f: Measure
        % g: Measure
        % ok: boolean
        
        m = mass(p)
        % m = mass(p)
        % Returns the mass of the Measure p
        
        s = support(p)
        % s = support(p)
        % Returns the support of the Measure p
        
        s = ndims(p)
        % s = ndims(p)
    end
end