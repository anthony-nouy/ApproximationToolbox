% Class AnisotropicGreedySpaceLinearSolver

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

classdef AnisotropicGreedySpaceLinearSolver < LinearSolver

    properties
        minimizeResidual
        checkResidual
        localSolver
        theta
    end
    
    methods
        function s = AnisotropicGreedySpaceLinearSolver(A,b,varargin)
            p = ImprovedInputParser;
            addParamValue(p,'minimizeResidual',false,@islogical);
            addParamValue(p,'checkResidual',1,@isscalar);
            addParamValue(p,'localSolver',RankOneALSLinearSolver(A,b,'display', ...
                false));
            addParamValue(p,'theta',0.8,@isscalar);
            parse(p,varargin{:});
            s@LinearSolver(A,b,p.unmatchedArgs{:});
            s = passMatchedArgsToProperties(p,s);
        end
        
        function [x,output] = solve(s)
            flag = 1;
            if ~s.minimizeResidual
                A = s.A;
                b = s.b;
            else
                A = s.A'*s.A;
                b = s.A'*s.b;
            end
            if isempty(s.errorCriterion)
                nb = norm(s.b);
                errCrit = @(x) norm(s.b-s.A*x)/nb;
            else
                errCrit = s.errorCriterion;
            end
            if isempty(s.x0)
                R = b;
            else
                x = s.x0;
                R = b-A*x;
            end
            errvec = zeros(s.maxIterations,1);
            
            su = TuckerLikeTensorALSLinearSolver(A,b,'useDMRG',true);
            
            for k = 1:s.maxIterations
                s.localSolver.b = R;
                c = s.localSolver.solve();
                if (k == 1) && isempty(s.x0)
                    x0 = TuckerLikeTensor.zeros(s.b.sz);
                    x = c;
                    x.core = TTTensor(x.core);
                    lambda = 1:x.order;
                else
                    x0 = x;
                    w = x+c;
                    sp = c.space.spaces;
                    % if k == 2
                    %     lambda = 1:x.order;
                    % else
                    sv = Truncator.getSingularValues(w);
                    sv = sv{2};
                    sv = cellfun(@(x) min(x),sv);
                    sref = s.theta*max(sv);
                    lambda = find(sv >= sref)'
                    %end
                    
                    for mu = lambda
                        x.space.spaces{mu} = [x.space.spaces{mu} sp{mu}];
                        sz = x.core.cores{mu}.sz;
                        sz(2) = 1;
                        xc = FullTensor.zeros(sz);
                        x.core.cores{mu} = cat(x.core.cores{mu},xc,2);
                    end
                    x.space = updateProperties(x.space);
                    x.core = updateProperties(x.core);
                    x = updateProperties(x);
                    su.tolerance = 1e-6;%errvec(k-1)/10;
                    su.stagnation = 1e-6;%errvec(k-1)/10;
                    x = su.updateCore(x);
                    
                end
                output.lambda{k}=lambda;
                output.dims{k}=x.space.dim;
                output.ranks{k}=x.core.ranks;
                output.storage(k) = storage(x);
                
                stagnation = norm(orth(x-x0))/norm(x);
                if mod(k,s.checkResidual) == 0
                    errvec(k) = errCrit(x);
                end
                if s.display
                    if mod(k,s.checkResidual) ~= 0
                        fprintf('Iteration %3d - Stagnation %.2d\n', ...
                            k,stagnation);
                    else
                        fprintf(['Iteration %3d - Stagnation %.2d ' ...
                            '- Error %.2d\n'],k,stagnation,errvec(k));
                    end
                end
                if stagnation < s.stagnation
                    flag = 2;
                    break;
                end
                if (mod(k,s.checkResidual) == 0) && (errvec(k) < s.tolerance)
                    flag = 0;
                    break
                end
                R = b-A*x;
            end
            % Compute the residual for the last iteration if needed
            if mod(k,s.checkResidual) ~= 0
                errvec(k) = norm(s.A*x-s.b)/nb;
            end
            output.flag = flag;
            output.iter = k;
            output.error = errvec(k);
            output.errvec = errvec(1:k);
        end
    end
end