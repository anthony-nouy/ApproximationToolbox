% Class Learning

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

classdef Learning
    
    properties
        % LOSSFUNCTION - LossFunction object specifying the loss function
        lossFunction
        % DISPLAY - Logical to enable or disable the displays
        display = true
        % MODELSELECTION - Logical to enable or disable model selection
        modelSelection = true
        % MODELSELECTIONOPTIONS - Structure specifying options for model selection
        modelSelectionOptions = struct('stopIfErrorIncrease',false)
        % ERRORESTIMATION - Logical to enable or disable error estimation
        errorEstimation = false
        % ERRORESTIMATIONTYPE - Char specifying the type of error estimation
        errorEstimationType = 'leaveout'
        % ERRORESTIMATIONOPTIONS - Structure specifying options for error estimation
        errorEstimationOptions = struct('correction',true)
        % TRAININGDATA - Cell (in supervised learning) or double (in unsupervised learning) containing the training data
        trainingData
        % TESTERROR - Logical indicating if a test error is evaluated
        testError = false
        % TESTDATA - Cell (in supervised learning) or double (in unsupervised learning) containing the data to compute the test error
        testData
    end
    
    methods
        function s = Learning(loss)
            % LEARNING - Constructor for the class Learning
            %
            % s = LEARNING(loss)
            % loss: LossFunction
            % s: LEARNING
            
            if ~isa(loss,'LossFunction')
                error('Must provide a LossFunction object.')
            end
            s.lossFunction = loss;
        end
        
        function l = linearModel(s)
            % LINEARMODEL - Construction of the associated LinearModel object
            %
            % l = LINEARMODEL(s)
            % s: Learning
            % l: LinearModel
            
            switch class(s.lossFunction)
                case 'SquareLossFunction'
                    l = LinearModelLearningSquareLoss;
                case 'DensityL2LossFunction'
                    l = LinearModelLearningDensityL2;
            end
        end
        
        function r = riskEstimation(s,v,z,varargin)
            % RISKESTIMATION - Estimation of the risk associated with the function v using a dataset z
            % 
            % r = RISKESTIMATION(s,v,z,varargin)
            % s: Learning
            % v: Function or n-by-1 double of evaluations of the function on the dataset used to estimate the risk
            % z: cell or double containing the dataset used to estimate the risk
            % varargin: additional inputs
            % r: 1-by-1 double
            
            r = s.lossFunction.riskEstimation(v,z,varargin{:});
        end
    end
end