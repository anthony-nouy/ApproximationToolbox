% Class DensityL2LossFunction

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

classdef DensityL2LossFunction < LossFunction
    
    properties
        errorType = 'risk'
    end
    
    methods
        function [l,lRef] = eval(s,v,x)
            lRef = norm(v)^2;
            l = lRef - 2*v(x);
        end
        
        function e = relativeTestError(s,v,x)
            error('Relative test error not available for DensityL2LossFunction.')
        end
        
        function e = testError(s,v,x)
            assert(strcmpi(s.errorType,'risk'),'The errorType property must be set to "risk".')
            e = riskEstimation(s,v,x);
        end
    end
end