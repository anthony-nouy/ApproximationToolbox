% Class DiscreteMeasure

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

classdef DiscreteMeasure < Measure
    
    properties
        values
        weights
    end
    
    methods
        function X = DiscreteMeasure(values,weights)
            % X = DiscreteMeasure(values,weights)
            % DiscreteMeasure in R^d \sum_{i=1}^n w_i \delta_{x_i}
            % values: n-by-d array containing the set of values (x_1,...,x_N) taken by the random variables
            % weights: array containing the weights w_1,...,w_n
            % By default, w_i = 1
            %
            % X = DiscreteMeasure(Y)
            % if Y is a DiscreteRandomVariable
            % returns values = Y.values and weights = Y.probabilities
            
            if nargin==1 && isa(values,'DiscreteRandomVariable')
                X.values = values.values;
                X.weights = values.probabilities;
                return
            end

            N = size(values, 1);
            if nargin == 1
                weights = ones(1,N);
            elseif length(weights)~=N
                error('Arguments must have the same size.')
            end
            X.weights = weights(:);
        end

        function o = isDiscrete(~)
            o = true;
        end
        

        function n = ndims(X)
            n = size(X.values,2);
        end
        
        function m = mass(X)
            m = sum(X.weights);
        end
        
        function ok = eq(X,Y)
            if ~strcmp(class(X),class(Y)) || ~all(size(X.values)==size(Y.values)) || ~all(X.values(:)==Y.values(:)) || ~all(X.weights(:)==Y.weights(:))
                ok = false;
            else
                ok = true;
            end
        end
        
        function s = support(X)
            % s = support(X)
            % Returns two extreme points of the box in R^d containing the set of possible values
            % X: DiscreteRandomVariable in R^d
            % s: 2-by-d array
            
            s=[min(X.values,[],1),max(X.values,[],1)];
        end
        
        
        function varargout = plot(X,type,varargin)
            % varargout = plot(X,type)
            % Plots a graphical representation of the discrete measure.
            % X: RandomVariable
            % type: char (not used).
            dim = size(X.values,2);

            switch dim
                case 1
                    x = X.values(:)';
                    y = X.weights(:)';
                    delta = max(x)-min(x);
                    ax = [min(x)-delta/10,max(x)+delta/10];

                    plot([x;x],[zeros(1,length(x));y],varargin{:});

                    xlim(ax)
                    ylim([0,max(y)*1.1])
                otherwise
                    error('plot not implemented for dimension >1')

            end

            if nargout >= 1
                varargout{1} = x;
            end
            if nargout >= 2
                varargout{2} = y;
            end
        end

        function G = integrationRule(X)
            points = X.values;
            G = IntegrationRule(points(:),X.weights);
            
        end
        

        function p = orthonormalPolynomials(X,varargin)
            % p = orthonormalPolynomials(X)
            % Returns the orthonormal polynomials according to the DiscreteRandomVariable X
            % X: DiscreteMeasure
            % p: DiscretePolynomials
 
            p = DiscretePolynomials(X);

        end


        function [r,I] = random(X,n,varargin)
            % r = random(X,n)
            % Generates n random numbers according to the probability distribution obtained by rescaling the DiscreteMeasure
            % X: DiscreteMeasure
            % n: integer
            % r: n-by-d array 
            % I: n-by-1 array (indices of the generated points in X.values)
            
            if nargin==1
                n=1;
            end
            if any(X.weights<0)
                error('weights should be non negative')
            end
            Y = DiscreteRandomVariable((1:length(X.weights))',X.weights/sum(X.weights));
            I = icdf(Y,rand(n,1));
            r = X.values(I,:);
        end
    end
end