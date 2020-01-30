% Class SubFunctionalBasis

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

classdef SubFunctionalBasis < FunctionalBasis
    
    properties
        underlyingBasis
        basis
    end
    
    methods
        function f = SubFunctionalBasis(underlyingBasis,basis)
            % f = SubFunctionalBasis(underlyingBasis,basis)
            % underlyingBasis : FunctionalBasis
            % basis : array of size n\times m, where n is the number of elements in FunctionalBasis
            % basis defines a set of m basis functions in the space generated
            % by underlyingBasis
            
            switch nargin
                case 0
                    f.underlyingBasis = [];
                    f.basis = [];
                case 1
                    f.underlyingBasis = underlyingBasis;
                    f.basis = zeros(numel(f.underlyingBasis),0);
                case 2
                    f.underlyingBasis = underlyingBasis;
                    f.basis = basis;
                otherwise
                    error('Wrong number of input arguments.');
            end
            
            if nargin > 0
                f.measure = underlyingBasis.measure;
                
                if underlyingBasis.isOrthonormal && norm(basis'*basis - speye(size(basis,2)),'fro')/sqrt(size(basis,2)) < 1e-15
                    f.isOrthonormal = true;
                end
            end
        end
        
        function s = storage(f)
            s = numel(f.basis);
        end
        
        function N = cardinal(f)
            N = size(f.basis,2);
        end
        
        function n = ndims(f)
            n = ndims(f.underlyingBasis);
        end
        
        function D = domain(f)
            D = domain(f.underlyingBasis);
        end
        
        function f = orthonormalize(f)
            if ~f.underlyingBasis.isOrthonormal
                error('Method not implemented.')
            end
            a = f.basis;
            a(:,1)=a(:,1)/norm(a(:,1));
            for i=2:size(a,2)
                a(:,i) = a(:,i) - a(:,1:i-1)*(a(:,1:i-1)'*a(:,i));
                a(:,i)=a(:,i)/norm(a(:,i));
            end
            f.basis = a;
        end
        
        function fx = eval(f,x,indices)
            % fx = eval(f,x,indices)
            % Computes evaluations of f at points x
            % f: SubFunctionalBasis
            % x: N-by-d array of doubles
            
            if nargin<3
                indices = 1:cardinal(f);
            end
            fx = full(eval(f.underlyingBasis,x)*f.basis(:,indices));
            
        end
        
        function fx = evalDerivative(f,n,x)
            % fx = evalDerivative(f,n,x)
            % Computes the n-derivative of f at points x in R^d, with n a
            % multi-index of size d
            % f: SubFunctionalBasis
            % n: 1-by-d array of integers
            % x: N-by-d array of doubles
            
            fx = full(evalDerivative(f.underlyingBasis,n,x)*f.basis);
        end
        
        function df = derivative(f,n)
            % df = derivative(f,n)
            % Computes the n-derivative of f
            % f: SubFunctionalBasis
            % n: 1-by-d array of integers
            % df: SubFunctionalBasis
            
            df = f;
            df.underlyingBasis = derivative(f.underlyingBasis,n);
        end
        
        function m = mean(f)
            m = full((mean(f.underlyingBasis)'*f.basis)');
        end
    end
end