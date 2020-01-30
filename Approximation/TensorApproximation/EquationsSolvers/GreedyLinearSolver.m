% Class GreedyLinearSolver

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

classdef GreedyLinearSolver < LinearSolver
    
    properties
        localSolver
        checkResidual
        minimizeResidual
        update
        initialUpdate
    end
    
    methods
        function s = GreedyLinearSolver(A,b,varargin)
            p = ImprovedInputParser;
            addParamValue(p,'minimizeResidual',false,@islogical);
            addParamValue(p,'checkResidual',1,@isscalar);
            addParamValue(p,'localSolver',RankOneALSLinearSolver(A,b,'display', ...
                false));
            addParamValue(p,'update',@(x) x);
            addParamValue(p,'initialUpdate',false,@islogical);
            parse(p,varargin{:});
            s@LinearSolver(A,b,p.unmatchedArgs{:});
            s = passMatchedArgsToProperties(p,s);
        end
        
        function  [x,output] = solve(s)
            flag = 1;
            clock0 = clock;
            if ~s.minimizeResidual
                A = s.A;
                b = s.b;
            else
                A = s.A'*s.A;
                b = s.A'*s.b;
                s.localSolver.A = A;
                s.localSolver.b = b;
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
                % Initial update
                if s.initialUpdate
                    x = s.update(x);
                    initialError = errCrit(x);
                    converged = false;
                    if initialError < s.tolerance
                        flag = 0 ;
                        converged = true ;
                    end
                    % no stagnation criterion for initial update
                    if converged
                        if s.display
                            fprintf(['Greedy: Converged on initialization ' ...
                                '- Error %.2d\n'],initialError);
                        end
                        output.times = etime(clock,clock0);
                        output.flag = flag;
                        output.iter = 0;
                        output.error = initialError ;
                        output.errvec = initialError ;
                        output.time = output.times;
                        return
                    end
                end % end of initial update
                R = b-A*x;
            end
            errvec = zeros(s.maxIterations,1);
            for k = 1:s.maxIterations
                s.localSolver.b = R;
                c = s.localSolver.solve();
                if (k == 1) && isempty(s.x0)
                    x0 = TuckerLikeTensor.zeros(s.b.sz);
                    x = c;
                else
                    x0 = x;
                    x = x + c;
                end
                
                if k>1 || (s.initialUpdate && ~isempty(s.x0))
                    x = s.update(x);
                end
                
                stagnation = norm(x-x0)/norm(x);
                if mod(k,s.checkResidual) == 0
                    errvec(k) = errCrit(x);
                end
                if s.display
                    if mod(k,s.checkResidual) ~= 0
                        fprintf('Greedy: Iteration %3d - Stagnation %.2d\n', ...
                            k,stagnation);
                    else
                        fprintf(['Greedy: Iteration %3d - Stagnation %.2d ' ...
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
                output.times(k)=etime(clock,clock0);
            end
            % Compute the residual for the last iteration if needed
            if mod(k,s.checkResidual) ~= 0
                errvec(k) = errCrit(x);
            end
            if k == 1 % ensure output structure is always the same
                output.times = etime(clock,clock0) ;
            end
            output.flag = flag;
            output.iter = k;
            output.error = errvec(k);
            output.errvec = errvec(1:k);
            output.time=etime(clock,clock0);
        end
    end
    
    methods (Static)
        
        function output = outputStructure(sz,maxIter)
            % output = outputStructure(sz,maxIter)
            % Allocate pre-formatted output structure as returned by solve.
            % Useful inside loops, for concatenation purposes.
            % sz : array of desired structure array sizes.
            % maxIter : [Optional] greedy algorithm maximum iterations, for
            % array allocations. Defaults to 1.
            %
            % Example 1
            % output = GreedyLinearSolver.outputStructure([10 2]) ;
            %
            % Example 2
            % output = GreedyLinearSolver.outputStructure(3,10) ;
            
            if nargin < 2
                maxIter = 1 ;
            end
            % List field names (order is important)
            fieldNames = {'times','flag','iter','error','errvec','time'} ;
            % Allocate values according to what is expected (scalar or
            % array)
            arrayFields = ismember(fieldNames,{'times','errvec'}) ;
            scalarFields = ~arrayFields ;
            fieldValues = cell(size(fieldNames)) ;
            fieldValues(arrayFields) = {{zeros(maxIter,1)}} ;
            fieldValues(scalarFields) = {{[]}} ;
            % Set whole structure size (one arbitrary field value is enough)
            fieldValues(find(scalarFields,1)) = {repmat({[]},sz)} ;
            % Allocate structure with name-value pairs
            fields = [fieldNames ; fieldValues] ;
            output = struct(fields{:}) ;
        end
    end
end