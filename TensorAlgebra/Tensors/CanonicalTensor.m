% Class CanonicalTensor: algebraic tensors in canonical tensor format
%
% See also TSpaceVectors, TSpaceOperators, DiagonalTensor, TuckerLikeTensor

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

classdef CanonicalTensor < TuckerLikeTensor
    
    methods
        function x = CanonicalTensor(space,alpha)
            % CanonicalTensor - Constructor for the class CanonicalTensor
            %
            % x = CanonicalTensor(space,alpha)
            % Creates a tensor of order d
            % x = \sum_{i=1}^r alpha_i v^1_i\otimes ... \otimes v^d_i
            %
            % space: - TSpace containing the v_i^k (TSpaceVectors or TSpaceOperators)
            %        - if space is a cell, interpreted as TSpaceVectors
            % alpha: array of length r
            %
            % x.core  = DiagonalTensor(alpha,space.order);
            % x.space = space;
            %
            % See also TSpaceVectors, TSpaceOperators, DiagonalTensor, TuckerLikeTensor
            
            if isa(space,'cell')
                space = TSpaceVectors(space);
            end
            core = DiagonalTensor(alpha,space.order);
            x@TuckerLikeTensor(core,space);
        end
        
        function n = storage(x)
            n = storage(x.core) + sum(cellfun(@numel,x.space.spaces));
        end
        
        function n = sparseStorage(x)
            n = nnz(x.core.data) + sum(cellfun(@nnz,x.space.spaces));
        end
        
        function r = representationRank(x)
            r = size(x.core,1);
        end
        
        function A = timesMatrixExceptDimsEvalDiag(x,mu,H,flag)
            % Contractions with a set of matrices and evaluation of a partial diagonal
            %
            % A = timesMatrixExceptDimsEvalDiag(x,mu,H,flag)
            
            N = size(H{1},1);
            f = timesMatrix(x,H);
            r = length(x.core.data);
            
            if nargin == 4
                switch flag
                    case 'oneByOneFactor'
                        fmu = ones(N,r);
                        nomu = 1:x.order;
                        i = mu - r * (ceil(mu / r) - 1); % Problem here
                        nomu(mu)=[];
                        for nu=nomu
                            fmu = fmu.*f.space.spaces{nu};
                        end
                        A = H{mu}.*repmat(fmu(:,i),1,x.sz(mu));
                    case 'core'
                        A = ones(N,r);
                        for nu = 1:x.order
                            A = A.*f.space.spaces{nu};
                        end
                end
            else
                sz = x.sz;
                fmu = ones(N,r);
                nomu = 1:x.order;
                
                nomu(mu)=[];
                for nu=nomu
                    fmu = fmu.*f.space.spaces{nu};
                end
                
                A = repmat(permute(fmu,[1,3,2]),[1,sz(mu),1]);
                A = A.*repmat(full(H{mu}),[1,1,r]);
                A = reshape(A,[N,sz(mu)*r]);
            end
        end
        
        function x = timesMatrix(x,M,varargin)
            x.space = matrixTimesSpace(x.space,M,varargin{:});
        end
        
        function [g,ind] = parameterGradientEvalDiag(f,mu,H)
            % Returns the diagonal of the gradient of the tensor with respect to a given parameter
            %
            % [g,ind] = parameterGradientEvalDiag(x,mu)
            % x: CanonicalTensor
            % mu: index of the parameter (integer from 1 to x.order+1)
            %
            % The (x.order+1)-th parameter is the core
            
            r = length(f.core.data);
            if mu == f.order+1
                N = size(f.space.spaces{1},1);
                g = ones(N,r);
                for nu=1:f.order
                    g = g.*f.space.spaces{nu};
                end
                g = FullTensor(g);
                ind = [];
            else
                nomu = 1:f.order; nomu(mu)=[];
                N = size(f.space.spaces{nomu(1)},1);
                fmu = ones(N,r);
                for nu=nomu
                    fmu = fmu.*f.space.spaces{nu};
                end
                if nargin == 3
                    g = kronEvalDiag(FullTensor(fmu),FullTensor(H{mu}),1,1);
                else
                    g = kronEvalDiag(FullTensor(fmu),FullTensor(eye(f.sz(mu))),[],[],true);
                end
                ind = [mu ; 3];
            end
        end
    end
    
    
    methods (Static)
        function z=create(generator,rank,sz1,sz2)
            % Builds a CanonicalTensor using a given generator
            %
            % x=create(generator,rank,sz1,sz2)
            % Builds a CanonicalTensor of type Vector (if nargin=3) or Operator (if nargin=4) using the function generator (randn, ones, ...).
            
            d = numel(sz1);
            dims = repmat(rank,1,d);
            if nargin == 3
                s = TSpaceVectors.create(generator,sz1,dims);
            elseif nargin == 4
                s = TSpaceOperators.create(generator,sz1,sz2,dims);
            end
            z = CanonicalTensor(s,ones(rank,1));
        end
        
        function z = randn(rank,varargin)
            % Creates a tensor in Canonical format with i.i.d. parameters drawn according to the standard gaussian distribution
            % z = randn(rank,sz1)
            % z = randn(rank,sz1,sz2)
            z = CanonicalTensor.create(@randn,rank,varargin{:});
        end
        function z = rand(rank,varargin)
            % Creates a tensor in Canonical format with i.i.d. parameters drawn according to the uniform distribution on (0,1)
            % z = rand(rank,sz1)
            % z = rand(rank,sz1,sz2)
            
            z = CanonicalTensor.create(@rand,rank,varargin{:});
        end
        function z = zeros(rank,varargin)
            % Creates a tensor in Canonical format with parameters equal to zero
            % z = zeros(rank,sz1)
            % z = zeros(rank,sz1,sz2)
            
            z = CanonicalTensor.create(@zeros,rank,varargin{:});
        end
        function z = ones(rank,varargin)
            % Creates a tensor in Canonical format with parameters equal to one
            % z = ones(rank,sz1)
            % z = ones(rank,sz1,sz2)
            
            z = CanonicalTensor.create(@ones,rank,varargin{:});
        end
    end
end