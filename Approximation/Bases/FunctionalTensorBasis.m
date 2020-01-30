% Class FunctionalTensorBasis

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

classdef FunctionalTensorBasis < FunctionalBasis
    
    properties
        f
    end
    
    methods
        function F = FunctionalTensorBasis(f)
            % F = FunctionalTensorBasis(f)
            % f: FunctionalTensor with a root rank equal to the number of basis functions
            
            assert(isa(f,'FunctionalTensor'),'f should be a FunctionalTensor')
            root = f.tensor.tree.root;
            nbchroot = nnz(f.tensor.tree.children(:,root));
            Croot = f.tensor.tensors{root};
            if Croot.order==nbchroot
                Croot = FullTensor(Croot,nbchroot,[Croot.sz,1]);
            end
            F.f = f;
            F.measure = f.bases.measure;
        end
        
        function ch = christoffel(F)
            % ch = christoffel(F)
            % input F a FunctinalTensorBasis
            % returns a FunctionalTensor ch
            
            ch = times(F.f,F.f);
            C = ch.tensor.tensors{ch.tensor.tree.root};
            C = timesVector(C,ones(C.sz(end),1),C.order);
            ch.tensor.tensors{ch.tensor.tree.root}=C;
            ch = FunctionalTensor(ch,ch.bases);
        end
        
        function y = eval(F,x)
            y = eval(F.f,x);
            y = reshape(y,[size(x,1),cardinal(F)]);
        end
        
        function n = ndims(F)
            n = F.f.tensor.order;
        end
        
        function nu = optimalSamplingMeasure(f)
            w = 1/cardinal(f)*christoffel(f);
            nu = ProbabilityMeasureWithRadonDerivative(f.measure,w);
        end
        
        function n = cardinal(F)
            if isa(F.f.tensor,'TreeBasedTensor')
                tree = F.f.tensor.tree;
                n = F.f.tensor.ranks(tree.root);
            else
                error('Method not implemented.')
            end
        end
    end
end