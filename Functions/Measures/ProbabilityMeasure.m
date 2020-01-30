% Class ProbabilityMeasure

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

classdef ProbabilityMeasure < Measure
    
    methods
        function m = mass(p)
            m = 1;
        end
        
        function x = randomRejection(X,n,Y,c,m)
            % x = randomRejection(X,n,Y,c)
            % Generates n samples of X with a rejection method, with rejection probability measure Y
            % X and Y should admit densities with respect to the same reference measure, denoted p and q respectively, and such that p(x) <= c g(x) for some constant c
            % X: ProbabilityMeasure
            % n: integer number of samples
            % c: real number
            % m: integer (number of simultaneous trials, n by default)
            % if c is not provided or empty, c is taken as (an upper bound of) an
            % approximation sup_x
            % f(x)/g(x), where the supremum is taken over the support of $Y$
            
            if nargin <= 3 || isempty(c)
                x = random(Y,100000);
                c = 1.1 * max(X.pdf(x)/Y.pdf(x));
            end
            if nargin <= 4
                m=n;
            end
            
            x = zeros(0,ndims(X));
            while length(x)<n
                y = random(Y,m);
                u = rand(m,1);
                t = u.*(c*Y.pdf(y))<X.pdf(y);
                x = [x;y(t,:)];
            end
            x = x(1:n,:);
        end
    end
end