% Class GreedyInverseApproximator

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

classdef GreedyInverseApproximator
    
    properties
        A
        I
        P0
        minimizeResidual
        localSolver
        update
        maxIterations
        tolerance
        stagnation
        display
        checkResidual
    end
    
    methods
        function s = GreedyInverseApproximator(A,varargin)
            p = ImprovedInputParser;
            addRequired(p,'A');
            addParamValue(p,'I',TuckerLikeTensor.eye(A.sz(1,:)));
            addParamValue(p,'P0',[]);
            addParamValue(p,'minimizeResidual',false,@islogical);
            addParamValue(p,'localSolver',@(A,B) ...
                RankOneInverseApproximatorByALS(A,B, ...
                'display',false));
            addParamValue(p,'update',[]);
            addParamValue(p,'maxIterations',5,@isscalar);
            addParamValue(p,'tolerance',1e-6,@isscalar);
            addParamValue(p,'stagnation',1e-6,@isscalar);
            addParamValue(p,'display',true,@islogical);
            addParamValue(p,'checkResidual',1,@isscalar);
            parse(p,A,varargin{:});
            s = passMatchedArgsToProperties(p,s);
        end
        
        function [P,output] = solve(s)
            flag = 1;
            I = s.I;
            if ~s.minimizeResidual
                A = s.A;
                B = I;
            else
                A = s.A*(s.A');
                B = I*(s.A');
            end
            nI = norm(I);
            if isempty(s.P0)
                R = B;
            else
                R = B-s.P0*A;
            end
            resvec = zeros(s.maxIterations,1);
            localSolver = s.localSolver(A,R);
            for k = 1:s.maxIterations
                localSolver.B = R;
                W = localSolver.solve();
                if (k == 1) && isempty(s.P0)
                    P0 = TuckerLikeTensor.zeros(s.I.sz(1,:),s.I.sz(2,:));
                    P = W;
                else
                    P0 = P;
                    P = P + W;
                end
                if ~isempty(s.update)
                    P = s.update(A,B,P);
                end
                stagnation = norm(P-P0)/norm(P);
                if mod(k,s.checkResidual) == 0
                    resvec(k) = norm(I-P*s.A)/nI;
                end
                if s.display
                    if mod(k,s.checkResidual) ~= 0
                        fprintf('Iteration %3d - Stagnation %.2d\n', ...
                            k,stagnation);
                    else
                        fprintf(['Iteration %3d - Stagnation %.2d ' ...
                            '- Residual %.2d\n'],k,stagnation,resvec(k));
                    end
                end
                if stagnation < s.stagnation
                    flag = 2;
                    break;
                end
                if (mod(k,s.checkResidual) == 0) && (resvec(k) < s.tolerance)
                    flag = 0;
                    break
                end
                R = B-P*A;
            end
            output.iter = k;
            output.flag = flag;
            output.stagnation = stagnation;
            output.relres = resvec(k);
            output.resvec = resvec(1:k);
        end
    end
end
