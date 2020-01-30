% Class alphaFunction

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

classdef alphaFunction < Function
    
    properties
        f % Function
        alpha % Tuple
        notalpha % Tuple
        alphameasure
    end
    
    methods
        function falpha = alphaFunction(f,alpha,notalpha)
            d = f.dim;
            falpha.f = f;
            falpha.alpha = alpha;
            falpha.notalpha = notalpha;
            if ~isempty(f.measure)
                falpha.measure = marginal(f.measure,notalpha);
                falpha.alphameasure = marginal(f.measure,setdiff(1:d,notalpha));
            end
            falpha.dim = length(notalpha);
            falpha.evaluationAtMultiplePoints = f.evaluationAtMultiplePoints;
        end
        
        function yalpha = eval(falpha, xnotalpha)
            f = falpha.f;
            d = f.dim;
            alpha = falpha.alpha;
            notalpha = falpha.notalpha;
            [~,I]=ismember(1:d,[alpha,notalpha]);
            s.type = '()';
            s.subs = {':',I};
            yalpha = @(xalpha) eval(f,subsref(array(FullTensorGrid({xalpha,xnotalpha})),s));
            yalpha = UserDefinedFunction(yalpha, length(alpha),[size(xnotalpha,1),1]);
            yalpha.evaluationAtMultiplePoints = falpha.evaluationAtMultiplePoints;
            yalpha.measure = falpha.alphameasure ;
        end
        
        function [yalpha, xnotalpha] = random(falpha, n, Xnotalpha)
            if nargin ==1
                n=1;
            end
            
            if n > 1
                error('The number of samples must be one.')
            end
            
            if nargin < 3
                Xnotalpha = falpha.measure;
            end
            xnotalpha = random(Xnotalpha);
            yalpha = eval(falpha,xnotalpha);
        end
    end
end