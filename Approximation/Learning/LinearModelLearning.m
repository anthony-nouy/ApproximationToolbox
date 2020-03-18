% Class LinearModelLearning

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

classdef LinearModelLearning < Learning
    
    properties
        basis
        basisEval
        basisEvalTest
        regularization = false
        regularizationType = 'l1'
        regularizationOptions = struct('lambda',0)
        basisAdaptation = false
        basisAdaptationPath
        options
    end
    
    methods
        function s = LinearModelLearning(loss)
            s@Learning(loss);
        end
        
        function s = initialize(s)
            % If the test error cannot be computed, it is disabled
            if s.testError && isempty(s.basis) && isempty(s.basisEvalTest)
                warning('The test error cannot be computed.')
                s.testError = false;
            end
            
            % Bases evaluation
            if ismethod(s.basis, 'eval')
                if ~isempty(s.trainingData) && isempty(s.basisEval)
                    if iscell(s.trainingData) && ~isempty(s.trainingData{1})
                        s.basisEval = s.basis.eval(s.trainingData{1});
                    elseif ~iscell(s.trainingData) && ~isempty(s.trainingData)
                        s.basisEval = s.basis.eval(s.trainingData);
                    else
                        error('Must provide input training data.')
                    end
                end
                
                if s.testError && ~isempty(s.testData) && isempty(s.basisEvalTest)
                    if iscell(s.testData) && ~isempty(s.testData{1})
                        s.basisEvalTest = s.basis.eval(s.testData{1});
                    elseif ~iscell(s.testData) && ~isempty(s.testData)
                        s.basisEvalTest = s.basis.eval(s.testData);
                    else
                        error('Must provide input test data.')
                    end
                end
            end
            s.basisEval = full(s.basisEval);
            if ~isempty(s.basisEvalTest)
                s.basisEvalTest = full(s.basisEvalTest);
            end
        end
    end
end