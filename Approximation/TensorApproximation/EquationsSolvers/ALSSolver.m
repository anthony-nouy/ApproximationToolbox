% Class ALSSolver

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

classdef ALSSolver
    
    properties
        maxIterations
        stagnation
        display
        oneByOneFactor
        random
        innerLoops
        localSolver
    end
    
    methods
        function s = ALSSolver(varargin)
            p = ImprovedInputParser;
            addParamValue(p,'maxIterations',30,@isscalar);
            addParamValue(p,'stagnation',1e-6,@isscalar);
            addParamValue(p,'display',false,@islogical);
            addParamValue(p,'oneByOneFactor',false,@islogical);
            addParamValue(p,'random',false,@islogical);
            addParamValue(p,'innerLoops',1,@isscalar);
            addParamValue(p,'localSolver',[]);
            
            parse(p,varargin{:});
            s = passMatchedArgsToProperties(p,s);
        end
    end
end