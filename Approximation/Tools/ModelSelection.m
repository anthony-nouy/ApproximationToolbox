% Class ModelSelection

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

classdef ModelSelection
    properties
        % PENSHAPE - Function specifying the penalization shape
        penshape
        % DATA - Structure containing the data used for the model selection: data.C contains the complexities of the models, and data.Rn the associated empirical risk values
        data
        % GAPFACTOR - Multiplicative factor used in the slope heuristic
        gapFactor = 2
    end
    
    methods
        function m = mlambda(s,lambda)
            % MLAMBDA - Compute the argument of the minimum of the penalized risk for given values of the penalization factor lambda
            %
            % mlambda = MLAMBDA(s,lambda)
            % s: ModelSelection
            % lambda: n-by-1 or 1-by-n double
            % m: n-by-1 or 1-by-n double
            
            C = s.data.C;
            Rn = s.data.Rn;
            m = zeros(size(lambda));
            
            for i = 1:length(lambda)
                [~,m(i)] = min(Rn + lambda(i) * s.penshape(C));
            end
        end
        
        function [path,mPath] = lambdaPath(s)
            % LAMBDAPATH - Return the path of possible values of lambda and associated arguments of the minimum of the penalized risk
            %
            % [path,mPath] = LAMBDAPATH(s)
            % s: ModelSelection
            % path: 1-by-p double
            % mpath: 1-by-p double
            
            C = s.data.C;
            Rn = s.data.Rn;
            
            path = [];
            mPath = [];
            lambda0 = 0;
            lambdaCurrent = lambda0;
            [~,I] = min(Rn);
            
            while ~all(isnan(lambdaCurrent))
                lambdaCurrent = (Rn - Rn(I)) ./ (s.penshape(C(I)) - s.penshape(C));
                lambdaCurrent(lambdaCurrent<=lambda0) = nan;
                lambdaCurrent(lambdaCurrent==inf)=nan;
                [lambda0,I] = min(lambdaCurrent);
                if ~isnan(lambda0)
                    path = [path, lambda0];
                    mPath = [mPath, I];
                end
            end
            if isempty(path)
                path = 0;
                mPath = 1;
            else
                [~,Imin]  = min(Rn);
                path = [path(1)/2,path,path(end)*2];
                mPath = [Imin,mPath,mPath(end)];
            end
        end
        
        function [lambdaHat,mHat,lambdaPath,mPath] = slopeHeuristic(s,lambdaPath,mPath)
            % SLOPEHEURISTIC - Apply the slope heuristic to the path of possible values of lambda to compute its optimal value lambdaHat and associated argument of the minimum of the penalized risk mHat
            %
            % [lambdahat,mhat,lambdaPath,mPath] = SLOPEHEURISTIC(s,lambdaPath,mPath)
            % s: ModelSelection
            % path: 1-by-p double
            % mpath: 1-by-p double
            % lambdaHat: 1-by-1 double
            % mHat: 1-by-1 double
            
            % If all the complexities are equal, choose the first model
            if all(diff(s.data.C) == 0)
                lambdaHat = 0;
                mHat = 1;
                lambdaPath = lambdaHat;
                mPath = mHat;
                return
            end
            
            if nargin==1
                [lambdaPath,mPath] = s.lambdaPath;
            end
            C = s.data.C;
            
            lmid = (lambdaPath(1:end-1)+lambdaPath(2:end))/2;
            gaps = C(s.mlambda(lmid(1:end-1)))-C(s.mlambda(lmid(2:end)));
            [~,I] = max(gaps);
            lambdaHat = s.gapFactor * lambdaPath(I+1);
            mHat = mlambda(s,lambdaHat);
        end
    end
    
    methods (Static)
        function C = complexity(x,varargin)
            % COMPLEXITY - Return the complexity associated to the input argument's type
            %
            % C = COMPLEXITY(x,varargin)
            % x: cell or FunctionalBasisArray or FunctionalTensor or TreeBasedTensor
            % varargin: additional parameters (see methods complexityFunctionalBasisArray and complexityTreeBasedTensor)
            % C: double
            %
            % See also COMPLEXITYFUNCTIONALBASISARRAY, COMPLEXITYTREEBASEDTENSOR
            
            if isa(x,'cell')
                C = cellfun(@(x) ModelSelection.complexity(x,varargin{:}),x);
            elseif isa(x,'FunctionalBasisArray')
                C = numel(x.data);
            elseif isa(x,'FunctionalTensor')
                C = ModelSelection.complexity(x.tensor,varargin{:});
            elseif isa(x,'TreeBasedTensor')
                C = ModelSelection.complexityTreeBasedTensor(x,varargin{:});
            else
                error('Wrong argument.')
            end
        end
        
        function C = complexityFunctionalBasisArray(x,fun)
            % COMPLEXITYFUNCTIONALBASISARRAY - Return the complexity associated with a FunctionalBasisArray
            %
            % function C = COMPLEXITYFUNCTIONALBASISARRAY(x,fun)
            % fun : function applied to the array to extract the storage complexity
            %       (@storage by default)
            % C: double
            
            if nargin<2
                fun = @storage;
            end
            
            C = fun(x);
        end
        
        function C = complexityTreeBasedTensor(x,varargin)
            % COMPLEXITYTREEBASEDTENSOR - Return the complexity associated with the TreeBasedTensor
            %
            % function C = COMPLEXITYTREEBASEDTENSOR(x,fun,type)
            % fun: function applied to the tensor to extract the storage complexity
            % (e.g. @storage (by default), @sparseStorage, @sparseLeavesStorage)
            % type: 'standard' (by default), 'stiefel' or 'grassman'
            % C: double
            
            if nargin<2 || isempty(varargin{1})
                fun = @storage;
            else
                fun = varargin{1};
            end
            if nargin<3
                type = 'standard';
            else
                type = varargin{2};
            end
            
            if isa(x,'cell')
                C = ModelSelection.complexity(x,fun,type);
                return
            elseif isa(x,'FunctionalTensor')
                C = ModelSelection.complexity(x.tensor,fun,type);
                return
            elseif ~isa(x,'TreeBasedTensor')
                error('The first argument should be a TreeBasedTensor.')
            end
            
            switch lower(type)
                case 'standard'
                    C = fun(x);
                case 'grassman'
                    C =  fun(x) - sum(x.ranks.^2);
                case 'stiefel'
                    C =  fun(x) - sum(x.ranks.*(x.ranks+1)/2);
                otherwise
                    error('Wrong argument.')
            end
        end
    end
end