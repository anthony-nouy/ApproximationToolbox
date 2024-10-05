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
        dimension
    end
    
    methods
        function f = UserDefinedFunctionalBasis(hFun,measure, inputDim,n)
            % f = UserDefinedFunctionalBasis(hFun,measure,inputDim , n)
            % hFun: cell array of size n-by-1 containing n functions
            % or a function
            % measure: Measure (can be a RandomVector or RandomVariable for
            % defining a random generator and an expectation)
            % inputDim: integer indicating the dimension of the domain of 
            % the functions in hFun
            % n : integer (required if hFun is a function)
            %
            % The basis is not L2-orthonormal a priori, hence the
            % isOrthonormal property remains at its default value: false
            
            if isa(hFun,'cell')
                f.handleFun=hFun(:);
                f.dimension = length(hFun);
            else
                f.handleFun=hFun;
                if nargin<4
                    error('missing dimension of the basis')
                end
                f.dimension = n;
            end

            if nargin>=2
                f.measure = measure;
            end

            if nargin>=2 && ~isempty(measure)
                f.inputDimension = numel(measure);
            elseif nargin>=3 && ~isempty(inputDim)
                f.inputDimension = inputDim;
            else
                error('missing input dimension')
            end


        end
        
        function y = eval(f,x)
            
            
            if isa(f.handleFun,'cell')
                n = size(x,1);
                d = f.cardinal();
                y = zeros(n,d);
                for mu = 1:d
                    y(:,mu) = f.handleFun{mu}(x);
                end
            else
                y = f.handleFun(x);
            end
        end
        
        function s = domain(f)
            s = support(f.measure);
        end
        
        function n = ndims(f)
            n = f.inputDimension;
        end
        
        function N = cardinal(f)
            N =  f.dimension;
        end
    end
end