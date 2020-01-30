% Class RankOneInverseApproximatorByALS

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

classdef RankOneInverseApproximatorByALS
    
    properties
        A
        B
        P0
        W0
        maxIterations
        stagnation
        display
        symmetric
        sparse
        sparseOptions
    end
    
    methods
        function s = RankOneInverseApproximatorByALS(A,B,varargin)
            p = ImprovedInputParser;
            addRequired(p,'A');
            addRequired(p,'B');
            addParamValue(p,'P0',[]);
            addParamValue(p,'W0',[]);
            addParamValue(p,'maxIterations',5,@isscalar);
            addParamValue(p,'stagnation',1e-6,@isscalar);
            addParamValue(p,'display',true,@islogical);
            addParamValue(p,'symmetric',false,@islogical);
            addParamValue(p,'sparse',false,@islogical);
            addParamValue(p,'sparseOptions',{});
            parse(p,A,B,varargin{:});
            s = passMatchedArgsToProperties(p,s);
            % Extend options if numel(opt) = 1
            d = s.A.order;
            if numel(s.sparse) == 1
                s.sparse = s.sparse.*true(1,d);
            end
            if numel(s.symmetric) == 1
                s.symmetric = s.symmetric.*true(1,d);
            end
            % Check compatibility of the options
            if numel(s.sparse) ~= d
                error('Wrong size of the sparse vector');
            end
            if numel(s.symmetric) ~= d
                error('Wrong size of the symmetric vector');
            end
            if any(s.sparse & s.symmetric)
                error('Approximation can not be sparse AND symmetric');
            end
        end
        
        function [W,output] = solve(s)
            if isempty(s.W0)
                W = TuckerLikeTensor.randn(s.B.sz(1,:),s.B.sz(2,:));
            else
                W = s.W0;
            end
            W.space;
            Wt = W';
            WW = Wt.space*W.space;
            WAW = dot(s.A.space,WW);
            BW = dot(s.B.space,W.space);
            flag=0;
            a = W.core.data;
            for k = 1:s.maxIterations
                a0 = a;
                for lambda = 1:s.A.order
                    [Bc,Wc] = convertTensors(s.B.core,W.core);
                    coefZ = timesTensorTimesMatrixExceptDim(Bc,Wc,BW,lambda);
                    Z = evalInSpace(s.B.space,lambda,coefZ);
                    [Ac,Wc] = convertTensors(s.A.core,W.core);
                    coefQ = timesTensorTimesMatrixExceptDim(Ac,Wc,WAW,lambda);
                    Q = evalInSpace(s.A.space,lambda,coefQ);
                    if s.symmetric(lambda);
                        if size(Q,1) < 1500
                            v = lyapsym(Q,(Z+Z'));
                        else
                            warning('Symmetric constraint : too large for lyapsym.m')
                            v = full(Z)/full(Q);
                            v = (v+v')/2;
                        end
                    else
                        if s.sparse(lambda);
                            v = gspai(Q,Z,'auto',100,1e-5);
                        else
                            v = Z/Q;
                        end
                    end
                    a = norm(v,'fro');
                    v = v/a;
                    W.space.spaces{lambda}{1} = v;
                    WW.spaces{lambda} = v'*v;
                    for i = 1:s.A.space.dim(lambda)
                        WAW{lambda}(i) = sum(sum(s.A.space.spaces{lambda}{i}.*WW.spaces{lambda}));
                    end
                    for i = 1:s.B.space.dim(lambda)
                        BW{lambda}(i) = sum(sum(s.B.space.spaces{lambda}{i}.*v));
                    end
                end
                W.core.data = a;
                stag = abs(a0-a)/abs(a0+a);
                % display
                if s.display
                    fprintf('Iteration %3d - Stagnation %.2d\n', k, stag);
                end
                if (k>1) && stag < s.stagnation
                    flag = 1;
                    break
                end
            end
            % finalisation
            output.flag = flag;
            output.stagnation = stag;
            output.iter = k;
        end
    end
end