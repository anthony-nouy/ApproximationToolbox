% Tutorial for FullTensor objects

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

clearvars, clc, close all

%% Creation of random FullTensor with precribed sizes

% Order-4 full tensor with i.i.d. entries drawn according to the uniform distribution on (0,1)
t1 = FullTensor.rand([2 3 4 5]);
disp('t1 = FullTensor.rand([2 3 4 5]) = '); disp(t1)

% Order-6 full tensor with i.i.d. entries drawn according to the standard gaussian distribution
t2 = FullTensor.randn([2 3 4 4 3 2]);
fprintf('\nt2 = FullTensor.randn([2 3 4 4 3 2]) = \n'); disp(t2)

fprintf('Number of entries of t1 = %i\n',storage(t1))
fprintf('Number of non-zero entries of t1 = %i\n\n',sparseStorage(t1))


%% Operations on FullTensor

t1 = FullTensor.rand([2 3 4 5]);
t2 = FullTensor.randn([2 3 4 5]);

% Frobenius norm
fprintf('\nFrobenius norm of a tensor = %f \n',norm(t1))

% Multiplication and division by a scalar
fprintf('Multiplication by a scalar\n')
disp(t1 * 2)
fprintf('Division by a scalar\n')
disp(t1 / 2)

% Sum and difference of two tensors 
fprintf('Sum of two tensors\n')
disp(t1 + t2)
fprintf('Difference between two tensors\n')
disp(t1 - t2)

% Hadamard (component-wise) product of two tensors
fprintf('Hadamard (component-wise) product of two tensors tensors\n')
disp(t1 .* t2)

%% More operations on FullTensor

t1 = FullTensor.rand([2 3 4 5]);
t2 = FullTensor.randn([2 3 4 4 3 2]);

% Contraction of the tensors t1 and t2, in the dimensions 1 and 3, and 6 and 4, respectively
% The resulting tensor t3 is of order t3.order = 4+6-4 = 6, and of dimensions t3.sz = [3 5 2 3 4 3]
t3 = timesTensor(t1,t2,[1 3],[6 4]);
fprintf('\nt3 = timesTensor(t1,t2,[1 3],[6 4]) = \n'); disp(t3)

% Contraction of the tensors t1 and t2, in the dimensions 1 and 3, and 6 and 4, respectively,
% and evaluation of the diagonal in the dimensions 2 and 5, respectively
% The resulting tensor t4 is of order t4.order = 4+6-4-1 = 5, and of dimensions t4.sz = [3 5 2 3 4]
t4 = timesTensorEvalDiag(t1,t2,[1 3],[6 4],2,5);
fprintf('\nt4 = timesTensorEvalDiag(t1,t2,[1 3],[6 4],2,5) = \n'); disp(t4)

% Outer product of the tensors t1 and t2 with evaluation of the diagonal in the dimensions 3 and 4, respectively
% The resulting tensot t5 is such that
% t5(i1,i2,k,i4,j1,j2,j3,j5,j6) = t1(i1,i2,k,i4)t2(j1,j2,j3,k,j5,j6)
t5 = outerProductEvalDiag(t1,t2,3,4);
fprintf('\nt5 = outerProductEvalDiag(t1,t2,3,4) = \n'); disp(t5)

% Outer product of the tensors t1 and t2 with evaluation of several diagonals, in the dimensions 2 and 3, and 5 and 4 respectively
% The resulting tensor t6 is such that
% t6(i1,k,l,i4,j1,j2,j3,j6) = t1(i1,k,l,i4)t2(j1,j2,j3,l,k,j6)
t6 = outerProductEvalDiag(t1,t2,[2,3],[5,4],true);
fprintf('\nt6 = outerProductEvalDiag(t1,t2,[2,3],[5,4],true) = \n'); disp(t6)

% Contraction by vectors along specified dimensions
dims = [2,3];
V = {rand(size(t1,dims(1)),1),rand(size(t1,dims(2)),1)};
t7 = timesVector(t1,V,dims)
fprintf('squeezing the tensor')
squeeze(t7)

% Contraction by a matrix along a specified dimension
dims = [2,3];
A = {rand(6,size(t1,dims(1))),rand(8,size(t1,dims(2)))};
t8 = timesMatrix(t1,A,dims)


%% Orthogonalization of a FullTensor

% The tensor t7 is of same size and order at t4, but is such that its 
% {3}-matricization, denoted by Mt7, verifies Mt7*Mt7.' = eye(t4.sz(3))
t7 = orth(t4,3);
fprintf('\nt7 = orth(t4,3) = \n'); disp(t7)

% We check the orthogonality by computing the product of the {3}-matricization of t7 by itself
Mt7 = matricize(t7,3);
P = timesTensor(Mt7,Mt7,2,2);
fprintf('\nChecking the orthogonality condition on t7:\n'); disp(P.data);

%% Singular values of a FullTensor
SV = singularValues(t3);

d = length(SV);
n2 = floor(sqrt(d));
n1 = ceil(d/n2);
for i = 1:d
    subplot(n1,n2,i)
    bar(SV{i})
    title(sprintf('%i-singular values',i))
end
