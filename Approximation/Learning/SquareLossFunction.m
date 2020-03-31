% Class SquareLossFunction

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

classdef SquareLossFunction < LossFunction
    
    properties
        errorType = 'relative' % Choice between 'relative' or 'risk'
    end
    
    methods
        function [l,lRef] = eval(s,v,z)
            if isa(v,'Function')
                yPred = v.eval(z{1});
            else
                yPred = v;
            end
            y = z{2};
            
            l = sum((yPred-y).^2, 2);
            
            if nargout == 2
                lRef = sum(y.^2, 2);
            end
        end
        
        function e = relativeTestError(s,v,z)
            [e,eRef] = riskEstimation(s,v,z);
            e = sqrt(e / eRef);
        end
        
        function e = testError(s,v,z)
            switch s.errorType
                case 'risk'
                    e = sqrt(riskEstimation(s,v,z));
                case 'relative'
                    e = relativeTestError(s,v,z);
                otherwise
                    error('The errorType property must be set to "risk" or "relative".')
            end
        end
    end
end