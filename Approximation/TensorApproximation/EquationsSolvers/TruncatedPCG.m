% Class TruncatedPCG

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

classdef TruncatedPCG < LinearSolver
    
    properties
        truncator
    end
    
    methods
        function s = TruncatedPCG(A,b,truncator,varargin)
            % s = TruncatedPCG(A,b,truncator)
            % Truncated Preconditioned Conjugate Gradient for the solution of a tensor structured
            % linear system of equations Ax=b
            % A: AlgebraicTensor (Operator in tensor format)
            % b: Algebraic Tensor
            % truncator: Truncator
            % x: AlbegraicTensor
            % output: structure
            %
            % Optional arguments
            % usage: truncatedPCG(A,b,truncator,'parameter',value)
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
            errvec = zeros(s.maxIterations,1);
            A = s.A;
            b = s.b;
            if isPrecond
                P = s.P;
            end
            r = b-A*x;
            if isPrecond
                z = P*r;
            else
                z = r;
            end
            p = z;
            Ap = A*p;
            pAp = dot(p,Ap);
            for k = 1:s.maxIterations
                x0 = x;
                omega = dot(r,p)/pAp;
                x = t.truncate(x + omega*p);
                stag = norm(x-x0)/norm(x);
                r = t.truncate(b-A*x);
                errvec(k) = errCrit(x);
                if s.display
                    fprintf(['Iteration %3d - Stagnation %.2d - Error ' ...
                        '%.2d\n'],k,stag,errvec(k));
                end
                if  stag < s.stagnation
                    flag = 2;
                    break
                end
                if errvec(k) < s.tolerance
                    flag = 0;
                    break;
                end
                if isPrecond
                    z = P*r;
                else
                    z = r;
                end
                beta = -dot(z,Ap)/pAp;
                p = t.truncate(z + beta*p);
                Ap = A*p;
                pAp = dot(p,A*p);
            end
            output.iter = k;
            output.flag = flag;
            output.errvec = errvec(1:k);
            output.error = errvec(k);
        end
    end
end