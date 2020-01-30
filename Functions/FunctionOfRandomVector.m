% Class FunctionOfRandomVector

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

classdef FunctionOfRandomVector
    
    properties
        X
        f
    end
    
    methods
        function Y = FunctionOfRandomVector(f,X)
            % Y = FunctionOfRandomVector(X,f)
            % Function of a RandomVector f(X)
            %
            % f: Function
            % X: RandomVector
            
            assert(isa(f,'Function'),'f must be a Function')
            assert(isa(X,'ProbabilityMeasure'),'X must be a ProbabilityMeasure')
            assert(ndims(X)==f.dim,'ndims(X) must be equal to f.dim');
            Y.X = X;
            Y.f = f;
        end
        
        function sz = size(Y)
            sz = size(Y.f);
        end
        
        function [y,x] = random(Y,varargin)
            % [y,x] = random(Y,N)
            % Execute f(x) with x=random(X,[N,1])
            
            x = random(Y.X,varargin{:});
            if Y.f.evaluationAtMultiplePoints
                y = Y.f.eval(x);
                y = reshape(y,[size(x,1),size(Y.f)]);
            else
                y = zeros([size(x,1),size(Y.f)]);
                for i=1:size(x,1)
                    y(i,:) = Y.f.eval(x(i,:));
                end
            end
        end
    end
end