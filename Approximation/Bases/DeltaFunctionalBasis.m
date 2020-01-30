% Class DeltaFunctionalBasis

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

classdef  DeltaFunctionalBasis < FunctionalBasis    
    
    properties
        values
    end
    
    methods
        function p = DeltaFunctionalBasis(values)
            % function p = DeltaFunctionalBasis(values)
            % delta functions associated with a set of n values in R^d
            % values : n-by-d array containing the set of values
            
            p.values = values;
            p.isOrthonormal = false;
            p.measure = DiscreteMeasure(values);
        end
        
        function n = ndims(p)
            n=size(p.values,2);
        end
        
        function n = domain(p)
            n = p.values;
        end
        
        function n = cardinal(p)
            n = size(p.values,1);
        end
        
        function px = eval(p,x)
            px = zeros(size(x,1),cardinal(p));
            if size(x,1)>cardinal(p)
                for i=1:cardinal(p)
                    px(ismember(x,p.values(i,:),'rows'),i)=1;
                end
            else
                for j=1:size(x,1)
                    px(j,ismember(p.values,x(j,:),'rows'))=1;
                end
            end
        end
        
        function H = one(p)
            H = ones(numel(p),1);
        end
        
        
        function m = mean(p,rv)
            % function m = mean(f)
            % gives the expectation of basis functions, when associated
            % with a uniform measure
            %
            % m = mean(f,rv)
            % rv : DiscreteRandomVariable or DiscreteRandomVector
            if nargin==1
                rv = DiscreteRandomVariable(p.values);
            end
            m = rv.probabilities(:);
        end
        
        function [px,x] = random(p,n,rv)
            % function [px,x] = random(p,n,rv)
            
            if nargin<3
                rv = DiscreteRandomVariable(p.values);
            end
            if nargin<2
                n=1;
            end
            if isa(n,'cell')
                n=[n{:}];
            end
            x = random(rv,prod(n));
            px = p.eval(x);
            if numel(n)>1
                px = reshape(px,[n(:)',numel(p)]);
            end
        end
    end
end