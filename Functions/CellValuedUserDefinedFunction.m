% Class CellValuedUserDefinedFunction

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

classdef CellValuedUserDefinedFunction < UserDefinedFunction
    
    methods
        function f = CellValuedUserDefinedFunction(fun,dim,sz)
            % f = CellValuedUserDefinedFunction(fun,dim,sz)
            % fun: function of dim variables
            % dim: number of input variables
            % sz: size of the output of f ([1,1] by default)
            %
            % fun(x) with x of size 1-by-dim returns a cell array of size
            % sz containing arrays of size 1-by-n_i
            % property evaluationAtMultiplePoints is set to false by default,
            % if set to true, fun(x) with x of size N-by-dim returns a cell array
            % sz containing arrays of size N-by-n_i
            
            f@UserDefinedFunction(fun,dim,sz);
        end
        
        function y = eval(h,x)
            n = size(x,1);
            f = h.fun;
            if ismethod(f,'eval')
                f = @(x) eval(f,x);
            end
            sz = prod(h.outputSize);
            y = cell(sz,1);
            if h.evaluationAtMultiplePoints
                [y{:}] = f(x);
            else
                yi = cellfun(@(x) y,cell(1,n),'UniformOutput',false);
                parfor i=1:n
                    yi{i} = f(x(i,:));
                end
                for j=1:sz
                    szj = 0;
                    for i=1:n
                        y{j}(i,:) = yi{i}{j}(:);
                        szj = max(szj,size(yi{i}{j}));
                    end
                    y{j} = reshape(y{j},[n,szj]);
                end
            end
            y = reshape(y,h.outputSize);
        end
    end
end