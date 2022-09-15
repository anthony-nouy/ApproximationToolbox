% Class ProductMeasure

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

classdef ProductMeasure < Measure
    
    properties
        measures
    end
    
    methods
        function p = ProductMeasure(measures)
            % p = ProductMeasure(measures)
            % Constructor for the class ProductMeasure
            % measures: cell of Measure objects or RandomVector
            % p: ProductMeasure
            
            if isa(measures,'RandomVector')
                if ~isa(measures.copula,'IndependentCopula')
                    warning('The given Copula is replaced by an IndependentCopula.')
                end
                measures = measures.randomVariables;
            elseif ~isa(measures,'cell')
                error('measures must be a cell of Measure')
            end
            p.measures = measures;
        end
        
        function ok = isa(obj,ClassName)
            % ok = isa(obj,ClassName)
            % Determine if input is object of specified class.
            %
            % See also isa
            
            if strcmp(ClassName,'ProbabilityMeasure')
                ok = all(cellfun(@(x) isa(x,'ProbabilityMeasure'),obj.measures));
            else
                ok = builtin('isa',obj,ClassName);
            end
        end
        
        function ok = eq(p,q)
            % ok = eq(p,q)
            % Checks if the two Measure p and q are identical
            % f: Measure
            % g: Measure
            % ok: boolean
            
            ok = all(cellfun(@(x,y) x == y,p.measures,q.measures));
        end
        
        function rv = randomVector(p)
            % rv = randomVector(p)
            % Returns, if p is a ProbabilityMeasure, the associated RandomVector
            % p: ProductMeasure
            % rv: RandomVector
            
            if ~all(cellfun(@(x)isa(x,'ProbabilityMeasure'),p.measures))
                error('The measures should be ProbabilityMeasure.')
            end
            rv = RandomVector(p.measures);
        end
        
        function x = random(p,n)
            % x = random(p,n)
            % Computes n random samples of the ProductMeasure p, if p is a ProbabilityMeasure
            % p: ProductMeasure
            % x: n-by-1 double
            
            %if ~all(cellfun(@(p) isa(p,'ProbabilityMeasure'),p.measures))
            %    error('The object is not a ProbabilityMeasure.')
            %end
            if nargin > 2
                warning('random should have only two input arguments.')
            end
            if nargin==1
                n=1;
            end
            if numel(n)>1
                error('n must be an integer.')
            end
            
            dims = cellfun(@ndims,p.measures);
            x = zeros(n,sum(dims));
            for i = 1:length(p.measures)
                rep = sum(dims(1:i-1)) + (1:dims(i));
                x(:,rep) = random(p.measures{i},n);
            end
        end
        
        function x = randomSequential(p,n)
            % x = random(p,n)
            % Computes n random samples of the ProductMeasure p, if p is a ProbabilityMeasure
            % p: ProductMeasure
            % x: n-by-1 double
            
            if ~all(cellfun(@(p) isa(p,'ProbabilityMeasure'),p.measures))
                error('The object is not a ProbabilityMeasure.')
            end
            if nargin > 2
                warning('random should have only two input arguments.')
            end
            if nargin==1
                n=1;
            end
            if numel(n)>1
                error('n must be an integer.')
            end
            
            dims = cellfun(@ndims,p.measures);
            x = zeros(n,sum(dims));
            for i = 1:length(p.measures)
                rep = sum(dims(1:i-1)) + (1:dims(i));
                x(:,rep) = randomSequential(p.measures{i},n);
            end
        end
        
        function pmarg = marginal(p,ind)
            pmarg = ProductMeasure(p.measures(ind));
        end
        
        function px = pdf(p,x)
            if ~all(cellfun(@(p) isa(p,'ProbabilityMeasure'),p.measures))
                error('The object is not a ProbabilityMeasure.')
            end
            if ~isa(x,'cell')
                n=cellfun(@ndims,p.measures);
                x = mat2cell(x,size(x,1),n);
                px = cellfun(@(p,x) pdf(p,x),p.measures(:),x(:),'uniformoutput',false);
                px = prod([px{:}],2);
            end
        end
        
        function s = support(p)
            % s = support(p)
            % s: cell array containing the supports of the measures
            
            s = cell(1,numel(p.measures));
            for k = 1:numel(p.measures)
                s{k} = support(p.measures{k});
            end
        end
        
        function s = truncatedSupport(p)
            % s = truncatedSupport(p)
            % s: cell array containing the supports of the measures
            
            s = cell(1,numel(p.measures));
            for k = 1:numel(p.measures)
                s{k} = truncatedSupport(p.measures{k});
            end
        end
        
        function m = mass(p)
            % m = mass(p)
            
            m = prod(cellfun(@mass,p.measures));
        end
        
        function m = ndims(p)
            m = sum(cellfun(@ndims,p.measures));
        end



        function m = moment(p,I)
            % function m = moment(p,I)
            % Returns the moments m_i(p) = integral x^i dp(x) of the measure p 
            % with i listed in I
            % p: ProductMeasure
            % I: k-by-d array of integers, with d the dimension 
            % m : k-by-1 vector with m(i) = m_{I(i,:)}(X) 

            m = ones(size(I,1));
            n=cellfun(@ndims,p.measures);
            I = mat2cell(I,size(I,1),n);
            ms = cellfun(@(p,x) moment(p,x),p.measures(:),I(:),'uniformoutput',false);
            for k=1:length(p.measures)
                m = m.*ms{k}(:);
            end

        end

    end
    
    methods (Static)
        function p = duplicate(nu,d)
            measures = cell(1,d);
            measures(:) = {nu};
            p = ProductMeasure(measures);
        end
    end
end