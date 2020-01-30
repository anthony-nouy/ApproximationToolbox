% Class TruncatedRichardsonIterations

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

classdef TruncatedRichardsonIterations < LinearSolver
    
    properties
        stepSize
        truncator
    end
    
    methods
        function s = TruncatedRichardsonIterations(A,b,truncator,varargin)
            % function s = TruncatedRichardsonIterations(A,b,truncator)
            % Constructor of a solver using Truncated Richardson Iterations for the solution of a tensor structured
            % linear system of equations Ax=b
            % A: AlgebraicTensor (Operator in tensor format)
            % b: Algebraic Tensor
            % truncator: Truncator
            % x: AlbegraicTensor
            % output: structure
            %
            % Optional arguments
            % TruncatedRichardsonIterations(A,b,truncator,'parameter',value)
            % P: AlgebraicTensor (default = []) - preconditioner
            % x0: AlgebraicTensor (default = []) - initial guess
            % maxIterations: integer (default = 30)
            % tolerance: double (default = 1e-6)
            % stagnation: double (default = 1e-6)
            % display: boolean (default = true)
            % errorCriterion: function_handle (default = [])
            % stepSize: double (default = 1)
            
            p = ImprovedInputParser;
            addRequired(p,'truncator');
            addParamValue(p,'stepSize',1,@isscalar);
            parse(p,truncator,varargin{:});
            s@LinearSolver(A,b,p.unmatchedArgs{:});
            s = passMatchedArgsToProperties(p,s);
        end
        
        function [x,output] = solve(s)
            t = s.truncator;
            isPrecond = ~isempty(s.P);
            if isempty(s.x0)
                x = TuckerLikeTensor.zeros(s.b.sz);
            else
                x = s.x0;
            end
            flag = 1;
            if isempty(s.errorCriterion)
                nb = norm(s.b);
                errCrit = @(x) norm(s.b-s.A*x)/nb;
            else
                errCrit = s.errorCriterion;
            end
            R = s.b-s.A*x;
            errvec = zeros(s.maxIterations,1);
            for k = 1:s.maxIterations
                x0 = x;
                if isPrecond
                    dir = s.P*R;
                else
                    dir = R;
                end
                if isa(s.stepSize,'double')
                    rho = s.stepSize;
                elseif isa(s.stepSize,'optimal')
                    rho = dot(R,PR) / dot(PR,A*PR);
                end
                x = t.truncate(x+rho*dir);
                R = s.b-s.A*x;
                
                stag = norm(x-x0)/norm(x);
                errvec(k) = errCrit(x);
                if s.display
                    fprintf(['Iteration %3d - Stagnation %.2d - Error ' ...
                        '%.2d\n'],k,stag,errvec(k));
                end
                if stag < s.stagnation
                    flag = 2;
                    break;
                end
                if errvec(k) < s.tolerance
                    flag = 0;
                    break
                end
            end
            output.flag = flag;
            output.iter = k;
            output.error = errvec(k);
            output.errvec = errvec(1:k);
        end
    end
end