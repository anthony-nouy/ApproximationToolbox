% Class TTTensorALSLinearSolver

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

classdef TTTensorALSLinearSolver < LinearSolver
    
    properties
        minimizeResidual
        algorithm
        truncator
    end
    
    methods
        function s = TTTensorALSLinearSolver(A,b,varargin)
            p = ImprovedInputParser;
            addParamValue(p,'minimizeResidual',false,@islogical);
            addParamValue(p,'algorithm','als',@ischar);
            addParamValue(p,'truncator',...
                Truncator('maxRank',100,'tolerance',1e-12));
            parse(p,varargin{:});
            s@LinearSolver(A,b,p.unmatchedArgs{:});
            s = passMatchedArgsToProperties(p,s);
        end
        
        function [x,output] = solve(s,varargin)
            switch lower(s.algorithm)
                case 'als'
                    [x,output] = solveALS(s,varargin{:});
                case 'dmrg'
                    [x,output] = solveDMRG(s);
                case 'limitedmemorydmrg'
                    [x,output] = solveLimitedMemoryDMRG(s);
                otherwise
                    error('Wrong type of algorithm.');
            end
        end
        
        function [x,output] = solveALS(s,ranks)
            flag = 1;
            if ~s.minimizeResidual
                A = s.A;
                b = s.b;
            else
                A = s.A'*s.A;
                b = s.A'*s.b;
            end
            if isempty(s.x0)
                if nargin == 1
                    ranks = randi(10,A.order-1,1);
                end
                x = TTTensor.randn(b.sz,ranks);
            else
                x = s.x0;
            end
            d = s.A.order;
            i = 1;
            stag = 1;
            relres = 1;
            while (i <= s.maxIterations) && (stag > s.stagnation) ...
                    && (relres > s.tolerance)
                x0 = x;
                x = orth(x);
                for lambda = 1:d
                    x = TTTensorALSLinearSolver.updateCore(A,b,x,lambda);
                    if lambda ~= d
                        [x.cores{lambda},r] = orth(x.cores{lambda},3);
                        x.cores{lambda+1} = timesMatrix(x.cores{lambda+1},r,1);
                    end
                end
                relres = norm(orth(s.A*x-s.b))/norm(s.b);
                stag = norm(orth(x-x0))/norm(x0);
                if s.display
                    fprintf('Iteration %3d - Stagnation %.2d - Error %.2d\n',i,stag,relres);
                end
                i = i + 1;
            end
            if  stag < s.stagnation
                flag = 2;
            end
            output.relres = norm(orth(s.A*x-s.b))/norm(s.b);
            if output.relres < s.tolerance
                flag = 0;
            end
            output.iter = i;
            output.flag = flag;
        end
        
        function [x,output] = solveDMRG(s)
            flag = 1;
            if ~s.minimizeResidual
                A = s.A;
                b = s.b;
            else
                A = s.A'*s.A;
                b = s.A'*s.b;
            end
            if isempty(s.x0)
                ranks = randi(10,A.order-1,1);
                x = TTTensor.randn(b.sz,ranks);
            else
                x = s.x0;
            end
            d = s.A.order;
            i = 1;
            stag = 1;
            relres = 1;
            while (i <= s.maxIterations) && (stag > s.stagnation) ...
                    && (relres > s.tolerance)
                x0 = x;
                x = orth(x);
                for lambda = 1:d-1
                    x = TTTensorALSLinearSolver.update2Cores(A,b,x,lambda,s.truncator,'left');
                end
                relres = norm(orth(s.A*x-s.b))/norm(s.b);
                stag = norm(orth(x-x0))/norm(x0);
                if s.display
                    fprintf('DMRG: Iteration %3d - Stagnation %.2d - Error %.2d\n',i,stag,relres);
                end
                i = i + 1;
            end
            if  stag < s.stagnation
                flag = 2;
            end
            output.relres = norm(orth(s.A*x-s.b))/norm(s.b);
            if output.relres < s.tolerance
                flag = 0;
            end
            output.iter = i;
            output.flag = flag;
        end
        
        function [x,output] = solveLimitedMemoryDMRG(s)
            flag = 1;
            if ~s.minimizeResidual
                A = s.A;
                b = s.b;
            else
                A = s.A'*s.A;
                b = s.A'*s.b;
            end
            if isempty(s.x0)
                ranks = randi(10,A.order-1,1);
                x = TTTensor.randn(b.sz,ranks);
            else
                x = s.x0;
            end
            d = A.order;
            i = 1;
            stag = 1;
            relres = 1;
            while (i <= s.maxIterations) && (stag > s.stagnation) ...
                    && (relres > s.tolerance)
                x0 = x;
                x = orth(x);
                for lambda = 1:d-1
                    x = TTTensorALSLinearSolver.update2CoresLimitedMemory(A, ...
                        b,x,lambda,'left');
                end
                relres = norm(orth(s.A*x-s.b))/norm(s.b);
                stag = norm(orth(x-x0))/norm(orth(x0));
                if s.display
                    fprintf('DMRG: Iteration %3d - Stagnation %.2d - Error %.2d\n',i,stag,relres);
                end
                i = i + 1;
            end
            if  stag < s.stagnation
                flag = 2;
            end
            output.relres = norm(orth(s.A*x-s.b))/norm(s.b);
            if output.relres < s.tolerance
                flag = 0;
            end
            output.iter = i;
            output.flag = flag;
        end
    end
    
    methods (Static)
        function L = reduceOperatorOnLeft(A,x,lambda)
            L = FullTensor.ones(ones(1,3));
            for mu = 1:lambda-1
                % L_{i'j'k'} = L_{ijk} X_{imi'}A_{jmnj'}X_{jnj'}
                L = timesTensor(L,x.cores{mu},1,1);
                L = timesTensor(L,A.cores{mu},[1 3],[1 2]);
                L = timesTensor(L,x.cores{mu},[1 3],[1 2]);
            end
        end
        
        function R = reduceOperatorOnRight(A,x,lambda)
            R = FullTensor.ones(ones(1,3));
            for mu = x.order:-1:lambda+1
                % R_{i'j'k'} = R_{ijk} X_{i'mi}A_{j'mnj}X_{k'nk}
                R = timesTensor(R,x.cores{mu},1,3);
                R = timesTensor(R,A.cores{mu},[1 4],[4 2]);
                R = timesTensor(R,x.cores{mu},[1 4],[3 2]);
            end
        end
        
        function Ared = reduceOperatorALS(A,x,lambda)
            N = prod(x.cores{lambda}.sz);
            LA = TTTensorALSLinearSolver.reduceOperatorOnLeft(A,x,lambda);
            RA = TTTensorALSLinearSolver.reduceOperatorOnRight(A,x,lambda);
            Ared = timesTensor(LA,A.cores{lambda},2,1);
            Ared = timesTensor(Ared,RA,5,2);
            Ared = permute(Ared,[1 3 5 2 4 6]);
            Ared = reshape(Ared,[N,N]);
            Ared = Ared.data;
        end
        
        function Ared = reduceOperatorDMRG(A,x,lambda)
            N = prod([x.cores{lambda}.sz(1:2) x.cores{lambda+1}.sz(2:3)]);
            LA = TTTensorALSLinearSolver.reduceOperatorOnLeft(A,x,lambda);
            RA = TTTensorALSLinearSolver.reduceOperatorOnRight(A,x,lambda+1);
            Ared = timesTensor(LA,A.cores{lambda},2,1);
            Ared = timesTensor(Ared,A.cores{lambda+1},5,1);
            Ared = timesTensor(Ared,RA,7,2);
            Ared = permute(Ared,[1:2:7 2:2:8]);
            Ared = reshape(Ared,[N,N]);
        end
        
        function Alr = reduceOperatorLimitedMemoryDMRG(A,x,lambda)
            LA = TTTensorALSLinearSolver.reduceOperatorOnLeft(A,x,lambda);
            RA = TTTensorALSLinearSolver.reduceOperatorOnRight(A,x,lambda+1);
            % Ared_{ijkl i'j'k'l'} = L_{imi'}A^{\lambda}_{mjj'p}
            % A^{\lambda+1}_{pkk'n} R_{lnl'}
            szAred = zeros(1,3);
            szAred(3) = size(A.cores{lambda},4);
            szAred(1) = size(LA,1)*size(A.cores{lambda},2);
            szAred(2) = size(LA,3)*size(A.cores{lambda},3);
            Ared = timesTensor(LA,A.cores{lambda},2,1);
            Ared = permute(Ared,[1 3 2 4 5]);
            Ared = reshape(Ared,szAred);
            szBred = szAred;
            szBred(1) = size(RA,1)*size(A.cores{lambda+1},2);
            szBred(2) = size(RA,3)*size(A.cores{lambda+1},3);
            Bred = timesTensor(A.cores{lambda+1},RA,4,2);
            Bred = permute(Bred,[2 4 3 5 1]);
            Bred = reshape(Bred,szBred);
            sp = cell(2,1);
            sp{1} = num2cell(Ared.data,[1 2]);
            sp{1} = sp{1}(:);
            sp{2} = num2cell(Bred.data,[1 2]);
            sp{2} = sp{2}(:);
            Alr = TuckerLikeTensor(DiagonalTensor.ones(szAred(3),2),...
                TSpaceOperators(sp));
        end
        
        function L = reduceRHSOnLeft(b,x,lambda)
            L = FullTensor.ones(ones(1,2));
            if lambda > 1
                for mu = 1:lambda-1
                    L = timesTensor(L,x.cores{mu},1,1);
                    L = timesTensor(L,b.cores{mu},[1 2],[1 2]);
                end
            end
        end
        
        function R = reduceRHSOnRight(b,x,lambda)
            R = FullTensor.ones(ones(1,2));
            d = x.order;
            if lambda < d
                for mu = d:-1:lambda+1
                    R = timesTensor(x.cores{mu},R,3,1);
                    R = timesTensor(R,b.cores{mu},[2 3],[2 3]);
                end
            end
        end
        
        function bred = reduceRHSALS(b,x,lambda)
            Lb = TTTensorALSLinearSolver.reduceRHSOnLeft(b,x,lambda);
            Rb = TTTensorALSLinearSolver.reduceRHSOnRight(b,x,lambda);
            bred = timesTensor(Lb,b.cores{lambda},2,1);
            bred = timesTensor(bred,Rb,3,2);
            bred = bred.data(:);
        end
        
        function bred = reduceRHSDMRG(b,x,lambda)
            Lb = TTTensorALSLinearSolver.reduceRHSOnLeft(b,x,lambda);
            Rb = TTTensorALSLinearSolver.reduceRHSOnRight(b,x,lambda+1);
            bred = timesTensor(Lb,b.cores{lambda},2,1);
            bred = timesTensor(bred,b.cores{lambda+1},3,1);
            bred = timesTensor(bred,Rb,4,2);
        end
        
        function blr = reduceRHSLimitedMemoryDMRG(b,x,lambda)
            Lb = TTTensorALSLinearSolver.reduceRHSOnLeft(b,x,lambda);
            Rb = TTTensorALSLinearSolver.reduceRHSOnRight(b,x,lambda+1);
            bred = timesTensor(Lb,b.cores{lambda},2,1);
            bred = reshape(bred,[bred.sz(1)*bred.sz(2) bred.sz(3)]);
            cred = timesTensor(b.cores{lambda+1},Rb,3,2);
            cred = reshape(cred,[cred.sz(1) cred.sz(2)*cred.sz(3)]);
            sp = {bred.data; cred.data'};
            blr = TuckerLikeTensor(DiagonalTensor.ones(cred.sz(1),2),...
                TSpaceVectors(sp));
        end
        
        function x = updateCore(A,b,x,lambda)
            Ared = TTTensorALSLinearSolver.reduceOperatorALS(A,x,lambda);
            bred = TTTensorALSLinearSolver.reduceRHSALS(b,x,lambda);
            xlambda = Ared\bred;
            x.cores{lambda}.data(:) = xlambda(:);
        end
        
        function x = update2Cores(A,b,x,lambda,truncator,orthSide)
            if nargin == 5
                orthSide = 'left';
            end
            Ared = TTTensorALSLinearSolver.reduceOperatorDMRG(A,x,lambda);
            bred = TTTensorALSLinearSolver.reduceRHSDMRG(b,x,lambda);
            
            szw = bred.sz;
            w = Ared.data\bred.data(:);
            w = reshape(w,[szw(1)*szw(2) szw(3)*szw(4)]);
            w = truncator.truncate(w);
            U = w.space.spaces{1};
            V = w.space.spaces{2};
            S = diag(w.core.data);
            
            switch lower(orthSide)
                case 'left'
                    V = V*S;
                    x.isOrth = lambda+1;
                case 'right'
                    U = U*S;
                    x.isOrth = lambda-1;
                otherwise
                    error('Wrong input values for orthSide')
            end
            
            szU = x.cores{lambda}.sz;
            szV = x.cores{lambda+1}.sz;
            r = size(U,2);
            szU(3) = r;
            szV(1) = r;
            x.cores{lambda} = FullTensor(reshape(U,szU),3,szU);
            x.cores{lambda+1} = FullTensor(reshape(V',szV),3,szV);
            x.ranks(lambda) = r;
        end
        
        function x = update2CoresLimitedMemory(A,b,x,lambda,orthSide)
            if nargin == 5
                orthSide = 'left';
            end
            Alr = TTTensorALSLinearSolver.reduceOperatorLimitedMemoryDMRG(A,x,lambda);
            blr = TTTensorALSLinearSolver.reduceRHSLimitedMemoryDMRG(b,x,lambda);
            nblr = norm(orth(blr));
            solv = GreedyLinearSolver(Alr,blr,...
                'maxIterations',50,...
                'tolerance',1e-8,...
                'stagnation',1e-8,...
                'errorCriterion',@(x) norm(orth(blr-Alr*x))/nblr,...
                'display',false);
            xlr = solv.solve();
            
            
            % U = FullTensor(xlr.space.spaces{1}*diag(xlr.core.data));
            % szU = [x.cores{lambda}.sz(1:2) r];
            % U = reshape(U,szU);
            
            % V = FullTensor(xlr.space.spaces{2}');
            % szV = [r x.cores{lambda+1}.sz(2:3)];
            % V = reshape(V,szV);
            
            % switch lower(orthSide)
            %   case 'left'
            %     [U,R] = orth(U,3);
            %     V = timesMatrix(V,R,1);
            %   case 'right'
            %     [V,R] = orth(V,1);
            %     U = timesMatrix(U,R,3);
            %   otherwise
            %     error('Wrong input values for orthSide')
            % end
            
            xlr = orth(xlr);
            R = xlr.core.data;
            
            U = FullTensor(xlr.space.spaces{1});
            szU = [x.cores{lambda}.sz(1:2) size(R,1)];
            U = reshape(U,szU);
            
            V = FullTensor(xlr.space.spaces{2}');
            szV = [size(R,2) x.cores{lambda+1}.sz(2:3)];
            V = reshape(V,szV);
            
            switch lower(orthSide)
                case 'left'
                    V = timesMatrix(V,R,1);
                    x.isOrth = lambda+1;
                case 'right'
                    U = timesMatrix(U,R,3);
                    x.isOrth = lambda-1;
                otherwise
                    error('Wrong input values for orthSide')
            end
            
            x.cores{lambda} = U;
            x.cores{lambda+1} = V;
            x.ranks(lambda) = U.sz(3);
        end
    end
end