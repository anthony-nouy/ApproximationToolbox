% Class FunctionalTensorPCAInterpolation

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

classdef FunctionalTensorPCAInterpolation < FunctionalTensorPCA
    
    properties
        tol
    end
    
    methods
        
        function FTPCA = FunctionalTensorPCAInterpolation()
            % s = FunctionalTensorPCAInterpolation()
        end
        
        
        
        function [projection, outputproj] = getalphaProjection(FTPCA, alpha,Valpha,gridalpha,varargin)
            if (numel(alpha) ==1 && FTPCA.basisAdaptation == true)
                alg = AdaptiveSparseTensorAlgorithm();
                alg.fullOutput = true;
                alg.maxIndex = [];
                alg.tol = FTPCA.tol(1); % to be eventually changed
                alg.displayIterations = false;
                alg.adaptiveSampling = true;
                projection = @(f)alg.interpolate(f,Valpha,gridalpha);

                if isa(Valpha , 'cell') || isa(Valpha , 'FunctionalBases')
                    Valpha = FullTensorProductFunctionalBasis(Valpha);
                end
            else
                if isa(Valpha , 'FunctionalBases')  || isa(Valpha , 'cell') 
                    Valpha = FullTensorProductFunctionalBasis(Valpha);
                end
                if ~isa(gridalpha, 'double')
                    gridalpha = array(gridalpha);
                end
                projection = @(y)Valpha.interpolate(y,gridalpha);
                outputproj.numberOfEvaluations = size(gridalpha,1);
            end
             outputproj.basis = Valpha;
        end
        
        function [gridalpha,nbeval] = alphaGrid(~,alpha,X,basis,gridalpha, varargin)
            
            if isa(gridalpha, 'FullTensorGrid')
                gridalpha = array(gridalpha);
            end
            if isempty(gridalpha)
                gridalpha = max(cardinal(basis)*10,1000);
                Xalpha = RandomVector(X.randomVariables(alpha));
                gridalpha = random(Xalpha,gridalpha);
            end
            
            if size(gridalpha,1) > cardinal(basis)
                gridalpha = magicPoints(basis,gridalpha);
            elseif size(gridalpha,1)<cardinal(basis)
                error('The number of grid points must be higher than the dimension of the basis.')
            end
            nbeval = size(gridalpha,1);
        end
    end
end

