% Class LaplacianLikeOperatorFactory

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

classdef LaplacianLikeOperatorFactory
    
    properties
        K
        M
        order
    end
    
    methods
        function f = LaplacianLikeOperatorFactory(K,M,order)
            f.K = K;
            f.M = M;
            f.order = order;
        end
        
        function A = makeCanonicalTensor(f)
            d = f.order;
            As = cell(d,1);
            for mu = 1:d
                As{mu} = cell(d,1);
                for k = 1:d
                    if k == mu
                        As{mu}{k} = f.K;
                    else
                        As{mu}{k} = f.M;
                    end
                end
            end
            As = TSpaceOperators(As);
            Ac = DiagonalTensor(ones(d,1),d);
            A = TuckerLikeTensor(Ac,As);
        end
        
        function A = makeTuckerTensor(f)
            As = f.makeTSpaceForTuckerLikeTensor();
            d = f.order;
            szCore = num2cell(2*ones(1,d));
            Ac = zeros(szCore{:});
            for mu = 1:d
                ind = szCore;
                ind{mu} = 1;
                Ac(ind{:}) = 1;
            end
            Ac = FullTensor(Ac);
            A = TuckerLikeTensor(Ac,As);
        end
        
        function A = makeTTTuckerLikeTensor(f)
            As = f.makeTSpaceForTuckerLikeTensor();
            d = f.order;
            Ac = cell(d,1);
            Ac{1} = FullTensor.zeros([1 2 2]);
            Ac{1}.data(1,1,1) = 1;
            Ac{1}.data(1,2,2) = 1;
            Ac{d} = FullTensor.zeros([2 2 1]);
            Ac{d}.data(1,2,1) = 1;
            Ac{d}.data(2,1,1) = 1;
            for mu = 2:d-1
                Ac{mu} = FullTensor.zeros([2 2 2]);
                Ac{mu}.data(1,2,1) = 1;
                Ac{mu}.data(2,1,1) = 1;
                Ac{mu}.data(2,2,2) = 1;
            end
            Ac = TTTensor(Ac);
            A = TuckerLikeTensor(Ac,As);
        end
        
        function A = makeTreeBasedTensor(f)
            t = DimensionTree.balanced(f.order);
            C = cell(t.nbNodes,1);
            C{t.root} = FullTensor.zeros([2 2]);
            C{t.root}.data(1,2) = 1;
            C{t.root}.data(2,1) = 1;
            nodList = 1:t.nbNodes;
            maxLvl = max(t.level);
            for lvl = 1:maxLvl
                nodLvl = nodList(t.level == lvl);
                for nod = nodLvl
                    if ~t.isLeaf(nod)
                        C{nod} = FullTensor.zeros([2 2 2]);
                        C{nod}.data(1,2,1) = 1;
                        C{nod}.data(2,1,1) = 1;
                        C{nod}.data(2,2,2) = 1;
                    end
                end
            end
            Ac = TreeBasedTensor(C,t);
            As = f.makeTSpaceForTuckerLikeTensor();
            A = TuckerLikeTensor(Ac,As);
        end
        
        function A = makeOperatorTTTensor(f)
            d = f.order;
            n = size(f.K,1);
            As = cell(d,1);
            As{1} = FullTensor.zeros([1 n n 2]);
            As{1}.data(1,:,:,1) = f.K;
            As{1}.data(1,:,:,2) = f.M;
            As{d} = FullTensor.zeros([2 n n 1]);
            As{d}.data(1,:,:,1) = f.M;
            As{d}.data(2,:,:,1) = f.K;
            for mu = 2:d-1
                As{mu} = FullTensor.zeros([2 n n 2]);
                As{mu}.data(1,:,:,1) = f.M;
                As{mu}.data(2,:,:,1) = f.K;
                As{mu}.data(2,:,:,2) = f.M;
            end
            A = OperatorTTTensor(As);
        end
        
        function As = makeTSpaceForTuckerLikeTensor(f)
            As = cell(f.order,1);
            for mu = 1:f.order
                As{mu} = {f.K;f.M};
            end
            As = TSpaceOperators(As);
        end
    end
end