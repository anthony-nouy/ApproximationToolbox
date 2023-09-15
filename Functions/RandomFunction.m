% Class RandomFunction

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

classdef RandomFunction
    
    properties
        sz
        f
        store = false
    end
    
    methods
        function Y = RandomFunction(f, sz)
            % Y = RandomFunction(f,sz)
            %
            % f: function such that f() generates a sample of a random array of size sz
            % sz: array
            
            Y.f = f;
            Y.sz = sz;
        end
        
        function sz = size(Y)
            sz = Y.sz;
        end
        
        function [y, lm] = random(Y,N)
            % y = random(Y,N)
            
            if nargin==1 || N==1
                [y, lm] = Y.f();
            else
                ys = cell(1,N);
                l = zeros(1,N);
                for k=1:N
                    ys{k} = Y.f();
                    l(k) = length(ys{k});
                end
                lm = max(l);
                y = zeros(N,lm);
                for k = 1:N
                    y(k,1:length(ys{k}))= ys{k};
                end
            end
        end
    end
end
