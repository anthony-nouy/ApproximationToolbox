% Class SensitivityAnalysis

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

classdef SensitivityAnalysis
    
    methods (Static)
        function S = closedSobolIndices(f,alpha,d)
            % S = closedSobolIndices(f,alpha,d)
            % Computes the Closed Sobol indices of f associated with the group of variables alpha in {1,...,d}
            %
            % f: Function
            % alpha: 1-by-s array of integers or 1-by-d logical
            % - if alpha is an array of integers, indices with respect to variables alpha
            % - if alpha is 1-by-d logical, indices with respect to variables find(alpha)
            %
            % If size(alpha,1)=n, returns the n indices associated with the rows of alpha
            
            if ~isa(alpha,'logical') && nargin==2
                error('Must provide the dimension d.')
            elseif ~isa(alpha,'logical')
                alpha = SensitivityAnalysis.indices2logical(alpha,d);
            end
            
            v = variance(f);
            S = varianceConditionalExpectation(f,alpha)./v;
        end
        
        function S = totalSobolIndices(f,alpha,d)
            % S = totalSobolIndices(f,alpha,d)
            % Computes the Total Sobol indices of f associated with the group of variables alpha in {1,...,d}
            %
            % f: Function
            % alpha: 1-by-s array of integers or 1-by-d logical
            % - if alpha is an array of integers, indices with respect to variables alpha
            % - if alpha is logical, indices with respect to variables find(alpha)
            %
            % If size(alpha,1)=n, returns the n indices associated with the rows of alpha
            
            if ~isa(alpha,'logical') && nargin==2
                error('Must provide the dimension d.')
            elseif ~isa(alpha,'logical')
                alpha = SensitivityAnalysis.indices2logical(alpha,d);
            end
            
            S = 1 - SensitivityAnalysis.closedSobolIndices(f,~alpha);
        end
        
        function S = sobolIndices(f,alpha,d)
            % S = sobolIndices(f,alpha,d)
            % Computes the Sobol indices of f associated with the group of variables alpha in {1,...,d}
            %
            % f: Function
            % alpha: 1-by-s array of integers or 1-by-d logical
            % - if alpha is an array of integers, indices with respect to variables alpha
            % - if alpha is logical, indices with respect to variables find(alpha)
            %
            % If size(alpha,1)=n, returns the n indices associated with the rows of alpha
            
            if ~isa(alpha,'logical') && nargin==2
                error('Must provide the dimension d.')
            elseif ~isa(alpha,'logical')
                alpha = SensitivityAnalysis.indices2logical(alpha,d);
            end
            
            if all(sum(alpha,2)==1)
                S = SensitivityAnalysis.closedSobolIndices(f,alpha);
            else
                d = size(alpha,2);
                I = powerSet(d);
                Sc = SensitivityAnalysis.closedSobolIndices(f,I);
                S = zeros(size(alpha,1),1);
                for k=1:size(alpha,1)
                    u = alpha(k,:);
                    repv = MultiIndices(I)<=MultiIndices(u);
                    a = (-1).^sum(repmat(u,nnz(repv),1) - I(repv,:),2);
                    S(k) = Sc(repv)'*a;
                end
            end
        end
        
        function S = sensitivityIndices(f,alpha,d)
            % S = sensitivityIndices(f,alpha,d)
            % Computes the sensitivity indices of f associated with the group of variables alpha in {1,...,d}
            %
            % f: Function
            % alpha: 1-by-s array of integers or 1-by-d logical
            % - if alpha is an array of integers, indices with respect to variables alpha
            % - if alpha is 1-by-d logical, indices with respect to variables find(alpha)
            %
            % If size(alpha,1)=n, returns the n indices associated with the rows of alpha
            
            if ~isa(alpha,'logical') && nargin==2
                error('Must provide the dimension d.')
            elseif ~isa(alpha,'logical')
                alpha = SensitivityAnalysis.indices2logical(alpha,d);
            end
            
            v = variance(f);
            S = varianceConditionalExpectation(f,alpha)./max(v);
        end
        
        function Sh = shapleyIndices(f,dims,d,cost)
            % S = shapleyIndices(f,dims,d)
            % Computes the Shapley indices of f associated with variables dims in {1,...,d}
            %
            % f: Function
            % dims: 1-by-s array (1:f.tensor.order by default)
            % d: integer
            % cost: function which associates to a logical array u of size 1-by-d the value cost(u) (optional)
            % By default, cost(u) is the closed Sobol index S^C_u
            
            if ~isa(dims,'logical') && nargin==2
                error('Must provide the dimension d.')
            elseif ~isa(dims,'logical')
                dims = SensitivityAnalysis.indices2logical(dims(:),d);
            end
            
            U = powerSet(d);
            if nargin<=3
                Sc = SensitivityAnalysis.closedSobolIndices(f,U);
                cost = @(v) Sc(ismember(U,v,'rows'));
            end
            Sh = zeros(size(dims,1),1);
            for i=1:size(dims,1)
                V = powerSet(~dims(i,:));
                for k = 1:size(V,1)
                    v = V(k,:);
                    vplusi = v;vplusi(i)=true;
                    Sh(i) = Sh(i) + (cost(vplusi) - cost(v))/(d)/nchoosek(d-1,sum(v));
                end
            end
        end
    end
    
    methods (Static, Hidden)
        function alpha = indices2logical(ind,d)
            alpha = false(size(ind,1),d);
            for k=1:size(alpha,1)
                alpha(k,ind(k,:)) = true;
            end
        end
    end
end
