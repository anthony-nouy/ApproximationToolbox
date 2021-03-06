% Class LossFunction

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

classdef LossFunction
    
    properties
        
    end
    
    methods
        function [e,eRef] = riskEstimation(s,v,z,varargin)
            [l,lRef] = eval(s,v,z,varargin{:});
            e = 1/length(l) * sum(l);
            
            if nargout == 2
                eRef = 1/length(lRef) * sum(lRef);
            end
        end
    end
    
    methods (Abstract)
        [l,lRef] = eval(s,v,z)
        
        e = relativeTestError(s,v,z)
        
        e = testError(s,v,z)
    end
end