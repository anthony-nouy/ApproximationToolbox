% Class UserDefinedFunctionalBasis

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

classdef UserDefinedFunctionalBasis < FunctionalBasis
    
    properties
        handleFun
        inputDimension = 1
    end
    
    methods
        function f = UserDefinedFunctionalBasis(hFun,measure, inputDim)
            % f = UserDefinedFunctionalBasis(hFun,measure)
            % hFun: cell array of size n-by-1 containing n functions
            % measure: Measure (can be a RandomVector or RandomVariable for
            % defining a random generator and an expectation)
            % inputDim: integer indicating the dimension of the domain of 
            % the functions in hFun
            %
            % The basis is a not L2-orthonormal a priori, hence the
            % isOrthonormal property remains at its default value: false
            
            switch nargin
                case 0
                    f.handleFun={};
                case 1
                    f.handleFun=hFun(:);
                case 2
                    f.handleFun = hFun(:);
                    f.measure = measure;
                    f.inputDimension = numel(measure);
                case 3
                    f.handleFun = hFun(:);
                    f.measure = measure;
                    f.inputDimension = inputDim;
            end
        end
        
        function y = eval(f,x)
            n = size(x,1);
            d = numel(f.handleFun);
            y = zeros(n,d);
            for mu = 1:d
                y(:,mu) = f.handleFun{mu}(x);
            end
        end
        
        function s = domain(f)
            s = support(f.rv);
        end
        
        function n = ndims(f)
            n = f.inputDimension;
        end
        
        function N = cardinal(f)
            N = length(f.handleFun);
        end
    end
end