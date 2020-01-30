% Class TuckerLikeTensorALSLinearSolver

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

classdef TuckerLikeTensorALSLinearSolver < LinearSolver
    
    properties
        maxIterationsCore
        maxIterationsTSpace
        stagnationCore
        stagnationTSpace
        useDMRG
    end
    
    methods
        function s = TuckerLikeTensorALSLinearSolver(A,b,varargin)
            p = ImprovedInputParser();
            addParamValue(p,'maxIterationsCore',10,@isscalar);
            addParamValue(p,'maxIterationsTSpace',10,@isscalar);
            addParamValue(p,'stagnationCore',1e-6,@isscalar);
            addParamValue(p,'stagnationTSpace',1e-6,@isscalar);
            addParamValue(p,'useDMRG',false,@islogical);
            parse(p,varargin{:});
            s@LinearSolver(A,b,p.unmatchedArgs{:});
            s = passMatchedArgsToProperties(p,s);
            
            assert(isa(A,class(b)),'A and b must be of same type');
            if isa(A,'TuckerLikeTensor')
                assert(isa(A.core,class(b.core)),...
                    'A and b must have same type of cores');
            end
        end
        
        function [x, output] = solve(s,varargin)
            if nargin == 2
                x = varargin{1};
            else
                assert(~isempty(s.x0), 'An initial guess must be provided');
                x = s.x0;
            end
            flag = 1;
            clock0 = clock;
            
            if isempty(s.errorCriterion)
                nb = norm(s.b);
                errCrit = @(x) norm(s.b-s.A*x)/nb;
            else
                errCrit = s.errorCriterion;
            end
            
            times = zeros(s.maxIterations, 1);
            stagnation = times;
            errvec = times;
            
            for i = 1:s.maxIterations
                x0 = x;
                x = s.updateCore(x);
                x = s.updateTSpaceByALS(x);
                times(i) = etime(clock, clock0);
                errvec(i) = errCrit(x);
                stagnation(i) = norm(x - x0) / norm(x);
                if s.display
                    fprintf('ALS: Iteration %3d - Stagnation %.2d - Error %.2d\n', ...
                        i, stagnation(i), errvec(i));
                end
                if stagnation(i) < s.stagnation
                    flag = 2;
                    break
                end
                if errvec(i) < s.tolerance
                    flag = 1;
                    break
                end
            end
            output.flag = flag;
            output.times = times(1:i);
            output.stagnation = stagnation(1:i);
            output.errvec = errvec(1:i);
            output.error = errvec(i);
        end
        
        function x = updateCore(s,x)
            switch class(s.A.core)
                case 'DiagonalTensor'
                    x = s.updateDiagonalTensorCore(x);
                case 'FullTensor'
                    x = s.updateFullTensorCore(x);
                case 'TreeBasedTensor'
                    x = s.updateTreeBasedCore(x);
                case 'TTTensor'
                    if s.useDMRG
                        x = s.updateTTTensorCoreDMRG(x);
                    else
                        x = s.updateTTTensorCoreALS(x);
                    end
                otherwise
                    error('not implemented')
            end
        end
        
        function x = updateDiagonalTensorCore(s,x)
            d = x.order;
            Ar = s.A.core.sz(1);
            br = s.b.core.sz(1);
            xr = x.core.sz(1);
            MA = dotWithMetrics(x.space,x.space,s.A.space);
            MA = [MA{:}];
            MA = reshape(MA,Ar,xr,d,xr);
            MA = prod(MA,3);
            MA = reshape(MA,Ar,xr*xr);
            MA = s.A.core.data'*MA;
            MA = reshape(MA,xr,xr);
            MB = dot(s.b.space,x.space);
            MB = [MB{:}];
            MB = reshape(MB,br,xr,d);
            MB = prod(MB,3);
            MB = reshape(MB,br,xr);
            MB = s.b.core.data'*MB;
            MB = MB';
            x.core.data = MA\MB;
        end
        
        function x = updateFullTensorCore(s,x)
            order = x.order;
            x = orth(x);
            sz = x.space.dim;
            MA = dotWithMetrics(x.space,x.space,s.A.space);
            MA = cellfun(@(m) reshape(m,size(m,1),size(m,2)*size(m,3))' ...
                ,MA,'UniformOutput',0);
            MA = timesMatrix(s.A.core,MA,1:order);
            SZ = [sz';sz'];
            MA = reshape(MA,SZ(:)');
            perm = [1:2:2*order,2:2:2*order];
            MA = permute(MA,perm);
            MA = reshape(MA,[prod(sz),prod(sz)]);
            MB = dot(s.b.space,x.space);
            MB = cellfun(@(x) x',MB,'UniformOutput',0);
            MB = timesMatrix(s.b.core,MB,1:order);
            MB = reshape(MB,[prod(sz),1]);
            sol = MA.data\MB.data;
            x.core.data = reshape(sol,sz);
        end
        
        function x = updateInternalNodeTreeBasedCore(s,x,nod)
            x.core = orthAtNode(x.core,nod);
            Axc = kron(s.A.core,x.core);
            Axxs = dot(s.A.space*x.space,x.space);
            bxs = dot(s.b.space,x.space);
            MA = reduceDotWithRankOneMetricAtNode(Axc,x.core,Axxs,nod);
            Mb = reduceDotWithRankOneMetricAtNode(s.b.core,x.core,bxs,nod);
            N = numel(x.core.tensors{nod});
            Aa = s.A.core.tensors{nod};
            d = Aa.order;
            szA = s.A.core.tensors{nod}.sz;
            szx = x.core.tensors{nod}.sz;
            szAx = Axc.tensors{nod}.sz;
            szM = zeros(size(MA));
            for i = 1:numel(MA)
                szM(i) = size(MA{i},2);
            end
            xAx = zeros(N);
            %% Left hand side
            if  d == 3 % optimized version
                iA=((1:szA(1)) - 1) * szx(1);
                jA=((1:szA(2)) - 1) * szx(2);
                kA=((1:szA(3)) - 1) * szx(3);
                kku = [1 cumprod(szx(1:end-1))];
                kkAx = [1 cumprod(szAx(1:end-1))];
                for n=1:N
                    %% Ndkron + first reshape for the TIMESMATRIX
                    % [iu,ju,ku] = ind2sxb(szx,n);
                    iu = zeros(3,1);
                    ndx = n;
                    for i = 3:-1:1
                        vi = rem(ndx-1, kku(i)) + 1;
                        vj = (ndx - vi)/kku(i) + 1;
                        iu(i) = vj;
                        ndx = vi;
                    end
                    i=iu(1)+iA;
                    j=iu(2)+jA;
                    k=iu(3)+kA;
                    % [I,J,K] = meshgrid(i,j,k);
                    I = cell(3,1);
                    nx = numel(i);
                    ny = numel(j);
                    nz = numel(k);
                    xx = reshape(i(:),[1 nx 1]); % Make sure x is a full row vector.
                    yy = reshape(j(:),[ny 1 1]); % Make sure y is a full column vector.
                    zz = reshape(k(:),[1 1 nz]); % Make sure z is a full page vector.
                    I{1} = xx(ones(ny,1),:,ones(nz,1));
                    I{2} = yy(:,ones(1,nx),ones(nz,1));
                    I{3} = zz(ones(ny,1),ones(nx,1),:);
                    % reshape(ndkron(Aa.a,an));
                    % subInd = sub2ind(szAx,I,J,K);
                    subInd = 1;
                    for i = 1:3
                        v = I{i};
                        subInd = subInd + (v-1)*kkAx(i);
                    end
                    %Axa = sparse(szAx(1),szAx(2)*szAx(3));
                    Axa = spalloc(szAx(1),szAx(2)*szAx(3),numel(subInd));
                    Axa(subInd) = Aa.data;
                    %% timesMatrix
                    B = reshape(MA{1}'*Axa,szM(1)*szAx(2),szAx(3));
                    B = reshape(full(B*MA{3}),szM(1),szAx(2),szM(3));
                    % B = timesMatrix(B,MA{2}',2);
                    B = reshape(...
                        permute(B,[1 3 2]),szM(1)*szM(3),szAx(2))...
                        *MA{2};
                    B = ipermute(reshape(B,szM(1),szM(3),szM(2)),[1 3 2]);
                    xAx(n,:) = B(:);
                end
            else % simple version
                az = FullTensor.zeros(szx);
                MA = cellfun(@(x) x', MA,'UniformOutput',false);
                for n=1:N
                    an = az;
                    an.data(n) = 1;
                    Axa = kron(Aa, an);
                    aa = timesMatrix(Axa,MA);
                    xAx(:,n)=aa.data(:);
                end
            end
            %% Right hand side
            Mb = cellfun(@(x) x', Mb,'UniformOutput',false);
            xb = timesMatrix(s.b.core.tensors{nod},Mb);
            xb = xb.data(:);
            x.core.tensors{nod}.data = reshape(xAx\xb,x.core.tensors{nod}.sz);
        end
        
        function x = updateTreeBasedCore(s,x)
            if ~isa(x.core, 'TreeBasedTensor')
                x.core = treeBasedTensor(x.core, s.A.core.tree);
            end
            for i = 1:s.maxIterationsCore
                x0 = x;
                for nod = s.A.core.tree.internalNodes
                    x = s.updateInternalNodeTreeBasedCore(x,nod);
                end
                stag = norm(x-x0)/norm(x0);
                if s.display
                    fprintf('updateTreeBasedCore: iteration %2d - stagnation %.2d\n',i,stag);
                end
                if stag < s.stagnationCore
                    break
                end
            end
        end
        
        function x = updateTTTensorCoreALS(s,x)
            x.core = TTTensor(x.core);
            x = orth(x);
            [Ac,bc,x] = reduceProblemToTTCore(s,x);
            options = {'maxIterations',s.maxIterationsCore, ...
                'stagnation',s.stagnationCore, ...
                'x0',x.core, ...
                'display',s.display};
            localSolver = TTTensorALSLinearSolver(Ac,bc,options{:});
            x.core = localSolver.solve();
        end
        
        function x = updateTTTensorCoreDMRG(s,x)
            x.core = TTTensor(x.core);
            x = orth(x);
            [Ac,bc,x] = reduceProblemToTTCore(s,x);
            mu = find(prod(Ac.sz)~=1,1);
            Ac = orth(Ac,mu);
            bc = orth(bc,mu);
            [Acs,dimsAc] = squeeze(Ac);
            bcs = squeeze(bc,dimsAc);
            if isa(Acs,'double')
                x.core = TTTensor.ones(ones(Ac.order,1));
                x.core.cores{1}.data = bcs/Acs;
            else
                options = {'maxIterations',s.maxIterationsCore, ...
                    'stagnation',s.stagnationCore, ...
                    'x0',squeeze(x.core,dimsAc), ...
                    'display',s.display, ...
                    'tolerance',s.tolerance, ...
                    'algorithm','limitedmemorydmrg'};
                localSolver = TTTensorALSLinearSolver(Acs,bcs,options{:});
                xcs = orth(localSolver.solve());
                x.core = insertDims(xcs,dimsAc);
            end
        end
        
        function x = updateTSpaceByALS(s,x)
            xAx = dotWithMetrics(x.space,x.space,s.A.space);
            xb  = dot(s.b.space,x.space);
            for k = 1:s.maxIterationsTSpace
                x0 = x;
                for mu = 1:s.A.order
                    x = updateTSpaceDim(s,x,mu,xAx,xb);
                    % updates
                    xAx(mu) = dotWithMetrics(x.space,x.space,s.A.space,mu);
                    xb(mu) = dot(s.b.space,x.space,mu);
                end
                stag = norm(x-x0)/norm(x0);
                if s.display
                    fprintf('updateTSpace: iteration %2d - stagnation %d\n',k,stag);
                end
                if (k>1) && stag < s.stagnationTSpace
                    break
                end
            end
        end
        
        function x = updateTSpaceDim(s,x,mu,varargin)
            if nargin == 3
                xAx = dotWithMetrics(x.space,x.space,s.A.space);
                xb = dot(s.b.space,x.space);
            elseif nargin == 5
                xAx = varargin{1};
                xb = varargin{2};
            else
                error('Wrong number of arguments');
            end
            % reduce operator A
            [Acore,xcore] = convertTensors(s.A.core,x.core);
            szxAx = size(xAx{mu});
            xcorexcore = kron(xcore,xcore);
            M = cellfun(@(m) reshape(m,size(m,1),size(m,2)*size(m,3)),xAx,'UniformOutput',0);
            coef = timesTensorTimesMatrixExceptDim(Acore,xcorexcore,M,mu);
            coef = reshape(coef,szxAx);
            szc = size(coef);
            if numel(szc) == 2 % do not forget singleton dimension
                szc = [szc 1];
            end
            As = s.A.space;
            K = kron(reshape(coef(1,:,:),szc(2),szc(3)),As.spaces{mu}{1});
            for l=2:As.dim(mu)
                K = K + kron(reshape(coef(l,:,:),szc(2),szc(3)),As.spaces{mu}{l});
            end
            % reduce rhs b
            [bcore,xcore] = convertTensors(s.b.core,x.core);
            coef = timesTensorTimesMatrixExceptDim(bcore,xcore,xb,mu);
            rhs = s.b.space.spaces{mu}*coef;
            rhs = rhs(:);
            % solve
            sol = K\rhs;
            sol = reshape(sol,x.space.sz(mu), ...
                x.space.dim(mu));
            % normalization & update
            n = sqrt(sum(sol.^2,1))';
            x.core = timesDiagMatrix(x.core,n,mu);
            x.space.spaces{mu} = full(sol*spdiags(1./n,0, ...
                numel(n), ...
                numel(n)));
        end
    end
    
    methods (Access = private)
        function [Ac,bc,x] = reduceProblemToTTCore(s,x)
            % U = cellfun(@(x) {x},x.space.spaces,'uniformoutput',false);
            % PU = CanonicalTensor(TSpaceOperators(U),1);
            PU = CanonicalTensor(TSpaceOperators(x.space.spaces),1);
            Ac = PU'*s.A*PU;
            Ac = OperatorTTTensor(Ac);
            bc = PU'*s.b;
            bc = TTTensor(bc);
        end
    end
end