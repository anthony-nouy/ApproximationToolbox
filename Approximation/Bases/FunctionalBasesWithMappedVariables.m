% Class FunctionalBasesWithMappedVariables

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

classdef FunctionalBasesWithMappedVariables < FunctionalBases
    
    properties
        mapping
        pushedForwardMarginalMeasures
        standardPushedForwardMarginalMeasures
    end
    
    methods
        function H = FunctionalBasesWithMappedVariables(bases,mapping,measure,pushedForwardMarginalMeasures,standardPushedForwardMarginalMeasures)
            % Class FunctionalBasesWithMappedVariables
            %
            % H = FunctionalBasesWithMappedVariables(bases,mapping,measure,pushedForwardMarginalMeasures,standardPushedForwardMarginalMeasures)
            % bases: cell of length d containing objects of type FunctionalBasis
            % mapping: Mapping
            % measure: Measure
            % pushedForwardMarginalMeasures: Measure
            % standardPushedForwardMarginalMeasures: Measure
            %
            % To create a FunctionalBasesWithMappedVariables by replication
            % of FunctionalBasis, see FunctionalBasesWithMappedVariables.duplicate
            
            if isa(bases,'FunctionalBases')
                bases = bases.bases;
            end
            
            H@FunctionalBases(bases);
            
            if ~isa(mapping,'Mapping')
                error('Must provide a Mapping object')
            end
            H.mapping = mapping;
            
            if nargin == 2
                H.measure = [];
            else
                H.measure = measure;
            end
            
            if nargin == 5
                H.pushedForwardMarginalMeasures = pushedForwardMarginalMeasures;
                H.standardPushedForwardMarginalMeasures = standardPushedForwardMarginalMeasures;
            end
        end
        
        function n = ndims(f)
            % n = ndims(f)
            % Returns the dimension of each basis of f
            % f: FunctionalBasesWithMappedVariables
            % n: d-by-1 double
            
            n = ones(f.mapping.m,1);
        end
        
        function f = removeMapping(f,k)
            % f = removeMapping(f,k)
            % Removes the mappings of f associated to the index k
            % f: FunctionalBasesWithMappedVariables
            % k: 1-by-n or n-by-1 double
            
            f.mapping = removeMapping(f.mapping,k);
        end
        
        function f = keepMapping(f,k)
            % f = keepMapping(f,k)
            % Keeps only the mappings of f associated to the index k
            % f: FunctionalBasesWithMappedVariables
            % k: 1-by-n or n-by-1 double
            
            f.mapping = keepMapping(f.mapping,k);
        end
        
        function f = modifyComponentMap(f,dims,coeff)
            f.mapping = modifyComponentMap(f.mapping,dims,coeff);
        end
        
        function m = addComponentMap(m,coeff)
            m.mapping = addComponentMap(m.mapping,coeff);
        end
        
        function f = permute(f,num)
            % f = permute(f,p)
            % Returns f with the basis permutation num
            % f: FunctionalBasesWithMappedVariables
            % p: array containing a permutation of 1:numel(f)
            
            f.bases = f.bases(num);
            f.mapping = permute(f.mapping,num);
        end
        
        function ok = eq(f,g)
            % ok = eq(f,g)
            % Checks if the two FunctionalBasesWithMappedVariables f and g
            % are identical
            % f: FunctionalBasesWithMappedVariables
            % g: FunctionalBasesWithMappedVariables
            % ok: boolean
            
            if ~isa(g,'FunctionalBasesWithMappedVariables')
                ok = 0;
            else
                ok = all(cellfun(@(x,y) x == y, f.bases, g.bases)) && ...
                    f.mapping == g.mapping;
            end
        end
        
        function [H,x] = eval(f,x,dims)
            % [H,x] = eval(f,x,dims)
            % Computes evaluations of the basis functions of f at points x.
            % If the properties pushedForwardMarginalMeasures and
            % standardPushedForwardMarginalMeasures are provided, an
            % isoprobabilistic transformation is performed on the mapped
            % variables
            % f: FunctionalBasesWithMappedVariables
            % x: n-by-number(f) double
            % dims: d-by-1 or 1-by-d double (optional)
            % H: d-by-1 cell containing doubles of size n-by-p, where p is
            % the number of basis functions in each dimension
            
            if nargin==2
                dims = 1:length(f);
            end
            
            x = f.mapping.eval(x);
            
            if ~isempty(f.pushedForwardMarginalMeasures) && ...
                    ~isempty(f.standardPushedForwardMarginalMeasures)
                x = transfer(f.pushedForwardMarginalMeasures,f.standardPushedForwardMarginalMeasures,x);
            end
            
            [H,x] = eval@FunctionalBases(f,x,dims);
        end
        
        function [H,x] = random(f,n,dims)
            if nargin == 1
                n = 1;
            end
            if nargin <= 2
                dims = 1:f.mapping.m;
            end
            if ismethod(f.measure,'random')
                x = random(f.measure,n);
                H = f.eval(x);
                H = H(dims);
            else
                error('The property measure must have a random method to generate random variables.')
            end
        end
        
        function M = gramMatrix(f,dims,rv)
            % Not compatible with the mapping
            
            error('Not compatible with the mapping.');
        end
        
        function H = mean(f,dims,varargin)
            % H = mean(f,dims,rv)
            % Estimates the mean of f in the dimensions in dims according
            % to the RandomVector rv if provided, or to the standard
            % RandomVector associated with each polynomial if not.
            % If dims is not provided or empty, the mean is computed for
            % all the dimensions.
            % f: FunctionalBases
            % dims: d-by-1 or 1-by-d double (optional)
            % rv: RandomVector or RandomVariable (optional)
            % H: d-by-1 cell containing doubles of size the number of basis
            % functions in each dimension
            
            if nargin == 1
                dims = 1:f.mapping.m;
            end
            if ismethod(f.measure,'random')
                x = random(f.measure,1e3);
                H = f.eval(x);
                H = cellfun(@(x) mean(x,1)',H(dims),'UniformOutput',false);
            else
                error('The property measure must have a random method to compute the mean.')
            end
        end
        
        function [points,I] = magicPoints(f,x,J)
            % Not compatible with the mapping
            
            error('Not compatible with the mapping.');
        end
    end
    
    methods (Static)
        function H = duplicate(basis,d,mapping)
            % H = duplicate(basis,d,mapping)
            % Creates a FunctionalBases with bases created with a
            % duplication of basis d times
            % basis: FunctionalBasis
            % d: 1-by-1 double
            % mapping: Mapping
            % H: FunctionalBasesWithMappedVariables
            
            basis = repmat({basis},1,d);
            H = FunctionalBasesWithMappedVariables(basis,mapping);
        end
    end
end