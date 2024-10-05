% Class AlgebraicTensor: abstract class of algebraic tensors
%
% AlgebraicTensor class provides a common interface for all tensors developed in this toolbox.
%
% See also FULLTENSOR, DIAGONALTENSOR, TUCKERLIKETENSOR, CANONICALTENSOR, TTTENSOR, TREEBASEDTENSOR

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

classdef (Abstract) AlgebraicTensor
    
    properties (Abstract)
        % ORDER - Order of the tensor
        order
        % SZ - Row vector containing the size of the AlgebraicTensor along each dimension
        sz
        % ISORTH - True if the representation of the tensor is orthogonal
        isOrth
    end
    
    methods
        function s = size(x,d)
            % SIZE - Size of a tensor
            %
            % s = SIZE(x)
            % Returns the size along all dimensions of the tensor x. It is equivalent to x.sz
            %
            % s = SIZE(x,d)
            % Returns the size of the tensor x along dimension d. It is equivalent to x.sz(d)
            %
            % x: AlgebraicTensor
            % d,s: integer
            %
            %  See also ALGEBRAICTENSOR, NDIMS, NUMEL
            
            if nargin == 1
                s = x.sz;
            else
                s = x.sz(d);
            end
        end
        
        function n = ndims(x)
            % NDIMS - Order of a tensor
            %
            % s = NDIMS(x)
            % Equivalent to x.order
            %
            % x: AlgebraicTensor
            % n: integer
            %
            % See also ALGEBRAICTENSOR, SIZE, NUMEL
            
            n = x.order;
        end
        
        function n = numel(x)
            % NUMEL - Number of coefficients of the tensor
            %
            % n = NUMEL(x) returns the number of entries of the tensor. It is equivalent to prod(size(x))
            %
            % x: AlgebraicTensor
            % n: integer
            %
            % See also ALGEBRAICTENSOR, NDIMS, NUMEL
            
            n = prod(x.sz);
        end
        
        function y = double(x)
            % DOUBLE - Conversion of an AlgebraicTensor to a double
            %
            % y = DOUBLE(x) returns the tensor y as a double. It is equivalent to y = full(x); y = y.data;
            %
            % x: AlgebraicTensor
            % y: x.sz(1)-by-...-by-x.sz(d) double
            %
            % See also ALGEBRAICTENSOR, FULLTENSOR, DOUBLE, FULL
            
            y = full(x);
            y = y.data;
        end
        
        function s = evalAtIndices(x,I,dims)
            % EVALATINDICES - Evaluation of tensor entries
            %
            % s = EVALATINDICES(x,I)
            % s(k) = x(I(k,1),I(k,2),...,I(k,d)), 1 \le k \le N
            % x: AlgebraicTensor
            % I: N-by-x.order integer
            % s: N-by-1 double
            %
            % s = EVALATINDICES(x,I,dims)
            % Partial evaluation
            % x: AlgebraicTensor
            % I: N-by-M integer, with M=numel(dims)
            % s: N-by-n_1-by-...-by-n_{d'} double, with d'=x.order-M
            % Up to a permutation (placing dimensions dims on the left)
            % s(k,i_1,...,i_d') = x(I(k,1),I(k,2),...,I(k,M),i_1,...,i_d'), 1 \le k \le N
            
            N = size(I,1);
            I = mat2cell(I,size(I,1),ones(1,size(I,2)));
            if nargin>2
                J = repmat({':'},1,x.order);
                J(dims)=I;
                I=J;
            end
            
            s = subTensor(x,I{:});
            if nargin>2 && numel(dims)>1 && N==1
                s = squeeze(s,dims(2:end)); 
            else
                if nargin>2 && numel(dims)>1
                    s = evalDiag(s,dims);
                elseif nargin==2
                    s = evalDiag(s);
                end
            end
        end
        
        function s = evalOperatorAtIndices(x,I1,I2,dims)
            % EVALOPERATORATINDICES - Evaluation of tensor (operator) entries
            %
            % s = EVALOPERATORATINDICES(x,I1,I2)
            % s(k) = x(I1(k,1),I2(k,1),I1(k,2),I2(k,2),...,I2(k,d)), 1 \le k \le N
            % x: AlgebraicTensor
            % I1, I2: N-by-x.order integer
            % s: N-by-1 double
            %
            % s = EVALOPERATORATINDICES(x,I1,I2,dims)
            % Partial evaluation
            % x: AlgebraicTensor
            % I1, I2: an array of size N-by-M, with M=numel(dims)
            % s: N-by-(n_1xm_1)-by-...-by-(n_{d'}xm_{d'}) double, with d'=x.order-M
            % Up to a permutation (placing dimensions dims on the left)
            % s(k,i_1,j_1,...,i_d',j_d') = x(I1(k,1),I2(k,1),...,I1(k,M),I2(k,M),i_1,j_1,...,i_d',j_d'), 1\le k \le N
            
            I1 = mat2cell(I1,size(I1,1),ones(1,size(I1,2)));
            I2 = mat2cell(I2,size(I2,1),ones(1,size(I2,2)));
            if nargin>3
                J = repmat({':'},1,x.order);
                J(dims)=I1;
                I1=J;
                J = repmat({':'},1,x.order);
                J(dims)=I2;
                I2=J;
            end
            I = [I1 ; I2];
            
            s = subTensor(x,I{:});
            if nargin>3 && numel(dims)>1
                s = evalDiag(s,dims);
            elseif nargin==3
                s = evalDiag(s);
            end
        end
        
        function y = timesMatrixEvalDiag(c,H,varargin)
            % TIMESMATRIXEVALDIAG - Evaluation of the diagonal of a tensor obtained by contraction of c with matrices
            %
            % y = TIMESMATRIXEVALDIAG(c,H)
            % Provides the diagonal of the tensor obtained by contracting tensor c with matrices H{k} along dimensions k, for k=1...c.order
            % c: AlgebraicTensor
            % H: 1-by-c.order or c.order-by-1 cell
            % y: n-by-1 double
            %
            % function y = TIMESMATRIXEVALDIAG(c,H,dims)
            % Provides the diagonal of the tensor obtained by contracting tensor c with matrices H{k} along dimensions dims(k), for k=1...length(dims)
            
            y = timesMatrix(c,H,varargin{:});
            y = evalDiag(y,varargin{:});
        end
        
        function x = squeeze(x,varargin)
            % SQUEEZE - Removal of dimensions of a tensor
            %
            % s = SQUEEZE(x)
            % Removes dimensions with size 1
            %
            % s = squeeze(x,dims)
            % Removes dimensions dims
        end

        function y = sum(x,i)
            % Computes the sum of a tensor along given dimensions
            %
            % y = sum(x,i)
            % x: AlgebraicTensor of order d
            % i: array of integers or 'all'
            % y: AlgebraicTensor of order d
            % Use squeeze to remove dimensions            

            if isa(i,'char') && strcmpi(i,'all')
                i = 1:x.order;
            end
            a = cell(1,length(i));
            for k=1:length(a)
                a{k} = ones(x.sz(i(k)),1);
            end
            y = timesVector(x,a,i);

        end
    end
    
    methods (Abstract)
        % PLUS - Addition of tensors
        %
        % z = x + y
        % Sums tensors x and y
        %
        % See also ALGEBRAICTENSOR, MINUS, UMINUS, MTIMES
        x = plus(x,y)
        
        % MINUS - Substraction of tensors
        %
        % z = x - y
        % Substracts tensor y to tensor x
        %
        % See also ALGEBRAICTENSOR, PLUS, UMINUS, MTIMES
        x = minus(x,y)
        
        % UMINUS - Change of the sign of a AlgebraicTensor
        %
        % z = -x
        % Changes the sign of x
        %
        % See also ALGEBRAICTENSOR, PLUS, MINUS, MTIMES
        x = uminus(x)
        
        % MTIMES - Multiplication of a tensor by a scalar
        %
        % z = s * x or x * s
        % Mutliplies each entry of the tensor x by s
        %
        % See also ALGEBRAICTENSOR, PLUS, MINUS, UMINUS
        x = mtimes(x,s)
        
        % TIMES - Element-by-element multiplication of two tensors
        %
        % z = x.*y
        % Returns a tensor z having as entries the product of corresponding entries of x and y
        %
        % See also ALGEBRAICTENSOR, PLUS, MINUS, UMINUS
        x = times(x,s)
        
        % FULL - Full representation of a tensor
        %
        % z = FULL(x)
        % Computes each element of the multidimensional array x and stores it in the FullTensor z
        %
        % See also ALGEBRAICTENSOR, FULLTENSOR
        xf = full(x)
        
        % STORAGE - Storage complexity
        %
        % n = storage(x)
        % Returns the storage complexity of tensor x (number of real or complex numbers)
        n = storage(x)
        
        % SPARSESTORAGE - Sparse storage complexity
        %
        % n = sparseStorage(x)
        % Returns the storage complexity of a tensor x without counting the zeros (number of nonzeros real or complex numbers)
        n = sparseStorage(x)
        
        % TIMESVECTOR - Contraction of a tensor with vectors
        %
        % z = TIMESVECTOR(x,V)
        % Computes the contraction of x with each vector contained in the cell array V. V{k} is contracted with the k-th dimension of x
        %
        % z = TIMESVECTOR(x,V,dim)
        % Contracts x with a vector of double V along dimension dim
        %
        % z = TIMESVECTOR(x,V,dims)
        % Computes the contraction of x with each vector contained in the cell array V along dimensions specified by dims. The operation is such that V{k} is contracted with the dims(k)-th dimension of x
        %
        % Note that if the output is expected to be a simple scalar, then the method returns a double, otherwise it returns an AlgebraicTensor
        %
        % See also ALGEBRAICTENSOR, TIMESMATRIX, TIMESDIAGMATRIX
        x = timesVector(x,V,varargin)
        
        % TIMESMATRIX - Contraction of a tensor with matrices
        %
        % z = TIMESMATRIX(x,M)
        % Computes the contraction of x with each matrix contained in the cell array M. The second dimension of M{k} is contracted with the k-th dimension of
        %
        % z = TIMESMATRIX(x,M,dim)
        % Contracts the dim-th dimension of x with the second dimension of M
        %
        % z = TIMESMATRIX(x,M,dims)
        % Computes the contraction of x with each matrix contained in the cell array M along dimensions specified by dims. The operation is such that the second dimension of M{k} is contracted with the dims(k)-th dimension of x
        %
        % See also ALGEBRAICTENSOR, TIMESVECTOR, TIMESDIAGMATRIX
        x = timesMatrix(x,M,varargin)
        
        % TIMESDIAGMATRIX - Contraction of a tensor with diagonal matrices
        %
        % z = TIMESDIAGMATRIX(x,M)
        % Computes the contraction of x with each matrix contained in the cell array M. The second dimension of M{k} is contracted with the k-th dimension of x
        %
        % z = TIMESDIAGMATRIX(x,M,dim)
        % Contracts the dim-th dimension of x with the second dimension of M
        %
        % z = TIMESDIAGMATRIX(x,M,dims)
        % Computes the contraction of x with  each matrix contained in the cell array M along dimensions specified by dims. The operation is such that the second dimension of M{k} is contracted with the dims(k)-th dimension of x
        %
        % See also ALGEBRAICTENSOR, TIMESVECTOR, TIMESMATRIX
        x = timesDiagMatrix(x,M,varargin)
        
        % DOT - Inner product of two tensors
        %
        % n = DOT(x,y)
        % Returns the canonical inner product of x and y
        %
        % n = DOT(x,y,M)
        % Returns the canonical inner product of M*x and y, where M is a symmetric positive definite operator
        %
        % See also ALGEBRAICTENSOR, NORM
        n = dot(x,y)
        
        % NORM - Norm of the tensor
        %
        % n = NORM(x)
        % Returns the canonical norm of a tensor. It is equivalent to sqrt(dot(x,x)) up to numerical accuracy
        %
        % n = NORM(x,M)
        % Returns the canonical norm of a tensor associated with a positive definite operator M
        % It is equivalent to sqrt(dot(M*x,x)) up to numerical accuracy
        %
        % See also ALGEBRAICTENSOR, DOT
        n = norm(x)
        
        % ORTH - Orthogonalization of the tensor
        %
        % z = orth(x) returns an orthogonalized representation of the tensor x (to be precised for each format) if possible
        %
        % See also ALGEBRAICTENSOR
        x = orth(x)
        
        % CAT - Concatenation of cores
        %
        % z = CAT(x,y)
        % Concatenates x and y in z. The core z is such that: $z_{i_1,\hdots,i_d} = x_{i_1,\hots,i_d}$ if $i_k \le x.sz(k)$ for all k, $z_{i_1,\hdots,i_d} = y_{i_1-x.sz(1),\hots,i_d-x.sz(d)}$ if $i_k > x.sz(k)$ for all k, $z_{i_1,\hdots,i_d} = 0$ otherwise
        % This function is useful for the addition of TuckerLikeTensors
        x = cat(x,y)
        
        % KRON - Kronecker product of tensors
        % z = kron(x,y)
        % Extends the function kron to cores in arbritary dimensions
        x = kron(x,y)
        
        % DOTWITHRANKONEMETRIC - Weighted inner product of two tensors
        %
        % s = DOTWITHRANKONEMETRIC(x,y,M)
        % Computes the weighted canonical inner product of x and y, where the inner product related to dimension k is weighted by M{k}. It is equivalent to dot(x,timesMatrix(y,M)), but can be much faster
        %
        % See also ALGEBRAICTENSOR, TIMESMATRIX, DOT, TIMESTENSORTIMESMATRIXEXCEPTDIM
        s = dotWithRankOneMetric(x,y,M)
        
        % TIMESTENSORTIMESMATRIXEXCEPTDIM - Particular type of contraction
        %
        % s = TIMESTENSORTIMESMATRIXEXCEPTDIM(x,y,M,dim)
        % Computes a special contraction of two tensors x, y, a cell of matrices M and a particular dimension dim. Note that dim must be a scalar, while M must be a cell array with x.order elements
        %
        %  The operation is equivalent to (sometimes faster than):
        % otherDims = setdiff(1:x.order,dim);
        % s = timesMatrix(y,M(otherDims),otherDims);
        % s = timesTensor(x,s,otherDims,otherDims);
        % s = s.data;
        %
        % See also ALGEBRAICTENSOR, TIMESMATRIX, DOT, DOTWITHRANKONEMETRIC
        s = timesTensorTimesMatrixExceptDim(x,y,M,dim)
        
        % EVALDIAG - Extraction of the diagonal of a tensor
        %
        % s = evalDiag(x)
        % x must be such that size(x,mu)=n for all mu
        % s is a vector of size n such that s(k) = x(k,k,..,k)
        %
        % s = evalDiag(x,dims)
        % x must be such that size(x,mu)=n for all mu in dims=[a1,...,ak]
        % Noting the complementary dimensions [b1,...,bm], with l=x.order-k, s is a tensor of size n_{b1}x...x n x ...x n_{bl} where n is at position a1. The entries of s are such that s(i_1,...,i_(a1-1),k,i_(a1),...,i_m) = x(i_1,...,i_(a1-1),k,i_(a1),...,k,...,k,,...,i_m)
        %
        % Warning: for TTTensors or TreeBasedTensor, dims must contain adjacent dimensions and s(i_1,...,i_(a1-1),k,i_(a1),...i_m) = x(i_1,...,i_(a1-1),k,...,k,i_(a1),...,im)
        s = evalDiag(x,dims);
        
        % SUBTENSOR - Extraction of a subtensor
        %
        % s = subtensor(x,I1,...,Id)
        % s is a tensor of size numel(I1)-by-....-by-numel(Id)
        % s(k1,...,kd)= x(I1(k1),...,Id(kd))
        s = subTensor(x,varargin)
        
        % SUPERTENSOR - Creation of a tensor with given subtensor and zeroentries elsewhere
        %
        % s = supertensor(x,sz,I1,...,Id)
        % s is a tensor of size sz(1)-by-...-by-sz(d) such that
        % x(I1(k1),...,Id(kd)) = s(k1,...,kd) and x(i) = 0 for entries i
        % not in I1 x ... x Id
        % s = superTensor(x,sz,varargin)
    end
end
