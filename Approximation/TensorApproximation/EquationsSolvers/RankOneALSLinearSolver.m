% Class RankOneALSLinearSolver

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

classdef RankOneALSLinearSolver < LinearSolver
    
    properties
        localSolver
        adjoint
        adjointDimensions
    end
    
    methods
        function s = RankOneALSLinearSolver(A,b,varargin)
            p = ImprovedInputParser;
            addParamValue(p,'localSolver',@(A,b,mu) A\b);
            addParamValue(p,'adjoint',false,@islogical);
            addParamValue(p,'adjointDimensions',1:A.order,@isnumeric);
            addParamValue(p,'stagnation',5e-2,@isscalar);
            addParamValue(p,'maxIterations',10,@isscalar);
            parse(p,varargin{:});
            s@LinearSolver(A,b,p.unmatchedArgs{:})
            s = passMatchedArgsToProperties(p,s);
        end
        
        function [x,output] = solve(s)
            % Some preparations to go faster
            order = s.b.order;
            Acore = s.A.core;
            Aspace = s.A.space;
            bcore = s.b.core;
            bspace = s.b.space;
            
            % initialization of some variables
            if isempty(s.x0)
                xs = TSpaceVectors.randn(s.b.sz);
                xc = DiagonalTensor(1,s.b.order);
            else
                xs = s.x0.space;
                xc = s.x0.core;
            end
            xca0 = 1;
            xst = xs;
            xtAx = dotWithMetrics(xst,xs,Aspace);
            xtb = dot(bspace,xst);
            flag = 1;
            % alternating Least Square iteration
            for k=1:s.maxIterations
                % optimization per order
                for mu=1:order
                    % reduce A
                    [Ac,xc2] = convertTensors(Acore,xc);
                    coef = timesTensorTimesMatrixExceptDim(Ac,xc2,xtAx,mu);
                    M = evalInSpace(Aspace,mu,coef);
                    % reduce b
                    [bc,xc2] = convertTensors(bcore,xc);
                    coef = timesTensorTimesMatrixExceptDim(bc,xc2,xtb,mu);
                    rhs = evalInSpace(bspace,mu,coef);
                    % solve
                    sol = s.localSolver(M,rhs,mu);
                    xca = norm(sol);
                    xs.spaces{mu} = sol/xca;
                    xc.data = xca;
                    if s.adjoint==1 && ismember(mu,s.adjointDimensions)
                        xst.spaces{mu} = s.localSolver(M',xs.spaces{mu},mu);
                    else
                        xst.spaces{mu} = xs.spaces{mu};
                    end
                    % updates
                    xtAx(mu) = dotWithMetrics(xst,xs,Aspace,mu);
                    xtb(mu)  = dot(bspace,xst,mu);
                end
                % stagnation
                stagnation = abs(xca0-xc.data)/(xca0+xc.data);
                xca0 = xc.data;
                % display
                if s.display
                    fprintf('Iteration %3d - Stagnation %.2d\n',k,stagnation);
                end
                % stopping criterium
                if (k>1) && stagnation < s.stagnation
                    flag = 2;
                    break
                end
            end
            output.flag = flag;
            output.iter = k;
            x = TuckerLikeTensor(xc,xs);
        end
    end
end