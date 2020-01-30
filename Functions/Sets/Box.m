% Class Box

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

classdef Box < BorelSet
    
    properties
        s % Array of size 2-by-dim containing the two extreme points
    end
    
    methods
        
        function B = Box(a,b)
            % Interval - Constructor of the class Box
            %
            % B = Box(a,b)
            % creates a set of the form [a(1),b(1)]x...x[a(d),b(d)]
            %
            % B = Box(s)
            % creates a set of the form [s(1,1),s(2,1)]x...x[s(1,d),s(2,d)]
            
            if nargin==1 && isa(a,'Box')
                B = a;
            elseif nargin==1 && isa(a,'Interval')
                B.dim = 1;
                B.s = a.s;
            elseif nargin==2
                B.dim = length(a);
                B.s = [a(:).';b(:).'];
            else
                if size(a,1)~=2
                    error('argument should be of size 2-by-dim')
                end
                B.dim = size(a,2);
                B.s = a;
            end
        end
        
        function n = ndims(B)
            n = B.dim;
        end
        
        function ok = isIn(B,x)
            dim = size(B.s,2);
            ok = true(size(x,1),1);
            for k=1:dim
                ok = ok & (x(:,k)>=B.s(1,k)) & (x(:,k)<=B.s(2,k));
            end
        end
        
        function s = eq(B,C)
            if ~isa(B,'Box') || ~isa(C,'Box') || ~(B.dim==C.dim)
                s=false;
                return
            end
            s = all(I.s(:)==J.s(:));
        end
        
        function x = max(B)
            x = B.s(2,:);
        end
        
        function x = min(B)
            x = B.s(1,:);
        end
        
        function v = vol(B)
            v = prod(B.s(2,:)-B.s(1,:));
        end
    end
end