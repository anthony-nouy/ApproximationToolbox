% Class LinearSolver: abstract Class for defining a linear solver

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

classdef (Abstract) LinearSolver
    
    properties
        A % Operator
        b % Right-hand side
        P % Preconditioner
        x0 % Initial guess
        maxIterations
        tolerance
        stagnation
        display
        errorCriterion
    end
    
    methods
        function s = LinearSolver(A,b,varargin)
            p = ImprovedInputParser;
            addRequired(p,'A');
            addRequired(p,'b');
            addParamValue(p,'P',[]);
            addParamValue(p,'x0',[]);
            addParamValue(p,'maxIterations',30,@isscalar);
            addParamValue(p,'tolerance',1e-6,@isscalar);
            addParamValue(p,'stagnation',1e-6,@isscalar);
            addParamValue(p,'display',true,@islogical);
            addParamValue(p,'errorCriterion',[]);
            parse(p,A,b,varargin{:});
            s = passMatchedArgsToProperties(p,s);
        end
    end
    
    methods (Abstract)
        [x,output] = solve(s)
    end
end