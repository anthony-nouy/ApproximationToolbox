% Class Interval

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

classdef Interval < BorelSet
    
    properties
        s
    end
    
    methods
        
        function I = Interval(a,b)
            % Interval - Constructor of the class Interval
            %
            % I = Interval(a,b)
            % creates an interval [a,b]
            %
            % I = Interval(s)
            % creates an interval [s(1),s(2)]
            
            if nargin==1 && isa(a,'Interval')
                I=a;
            else
                I.dim = 1;
                if nargin==1
                    I.s=a;
                else
                    I.s = [a;b];
                end
            end
        end
        
        function s = isIn(I,x)
            s = (x>=I.s(1)) & (x<=I.s(2));
        end
        
        function x = max(I)
            x = I.s(2);
        end
        
        function x = min(I)
            x = I.s(1);
        end
        
        function n = ndims(I)
            n = I.dim;
        end
        
        function v = vol(I)
            v = prod(I.s(2)-I.s(1));
        end
        
        function ok = eq(I,J)
            if ~isa(I,'Interval') || ~isa(J,'Interval')
                ok=false;
                return
            end
            ok = (I.s(1)==J.s(1)) & (I.s(2)==J.s(2));
        end
    end
end