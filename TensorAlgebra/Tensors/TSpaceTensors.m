% Class TSpaceTensors

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

classdef TSpaceTensors < TSpace
    
    properties
        spaces % cell array of cell array of operator : spaces{i}{j} is the jth operator of the ith TSpace
        order  % number of TSpaces
        dim    % number of elemnts per TSpace
        sz     % matrix that contains the dimension of the operator of the ith TSpace : sz(i,j) = size(spaces{i}{1},j)
        isOrth
    end
    
    methods
        function x = TSpaceTensors(s)
            % TSpaceTensors
            % x = TSpaceTensors(u) creats the tensor space of operator contained in u.
            % u: cell array of cell array of matrices so that u{i}{j} is the jth operator of the ith TSpace
            
            assert(isa(s,'cell'),'The input must be a cell.');
            x.spaces = s;
            x = updateProperties(x);
            x.isOrth=0;
        end
        
        function x = cat(x,y)
            % CAT
            % x = cat(x,y) concatenates the elements of x and y.
            % x,y: two TSpaceTensors of the same order and the same size
            
            x.spaces = cellfun(@(u,v) [u;v] ,x.spaces,y.spaces,'UniformOutput',0);
            x = updateProperties(x);
            x.isOrth=0;
        end
        
        function y = mtimes(x,y,dims)
            if nargin<3
                dims=1:x.order;
            end
            xs=x.spaces;
            if isa(y,'TSpaceTensors')
                for mu = dims
                    [I,J] = ind2sub([y.dim(mu),x.dim(mu)],...
                        1:(x.dim(mu)*y.dim(mu)));
                    y.spaces{mu} = cellfun( @(xs,ys) xs*ys, xs{mu}(J(:)), ...
                        y.spaces{mu}(I(:)),'UniformOutput',0);
                end
            else
                error('Wrong input arguments.');
            end
            y.spaces = y.spaces(dims);
            y.isOrth = 0;
            y=updateProperties(y);
        end
        
        function M = dot(x,y,dims)
            if nargin<3
                dims=1:x.order;
            end
            M = cell(x.order,1);
            for mu = dims
                M{mu} = zeros(x.dim(mu),y.dim(mu));
                for i = 1:x.dim(mu)
                    for j = 1:y.dim(mu)
                        M{mu}(i,j) = dot(x.spaces{mu}{i},y.spaces{mu}{j});
                    end
                end
            end
            M = M(dims);
        end
        
        function x = matrixTimesSpace(x,M,dims)
            % spaceTimesMatrix
            % x = spaceTimesMatrix(x,M,order) compute the linear transformation M of the order-th basis of 'x'
            
            if nargin<3
                dims=1:x.order;
            end
            if ~isa(M,'cell')
                M = {M};
            end
            x.spaces(dims) = cellfun(@(x) cellfun(@(xs,M) M*xs,x,M, ...
                'uniformoutput',false), ...
                x.spaces(dims),'uniformoutput',false);
            x.isOrth = 0;
            x = updateProperties(x);
        end
        
        function x = spaceTimesMatrix(x,M,dims)
            % spaceTimesMatrix
            % x = spaceTimesMatrix(x,M,order) compute the linear transformation M of the order-th basis of 'x'
            
            if nargin<3
                dims=1:x.order;
            end
            if isa(M,'double') && numel(dims) == 1
                M = {M};
            elseif ~isa(M,'cell') || numel(M) ~= numel(dims)
                error('Wrong input arguments.');
            end
            xs = cell(numel(dims),1);
            for mu = dims
                xs{mu} = cell(size(M,2),1);
                for i = 1:size(M,2)
                    for k = 1:size(M,1)
                        xs{mu}{i} = xs{mu}{k}*M(k,i);
                    end
                end
            end
            x.isOrth = 0;
            x = updateProperties(x);
        end
        
        function xalpha = evalInSpace(x,order,alpha)
            xalpha = x.spaces{order}{1} * alpha(1);
            for i=2:x.dim(order)
                xalpha = xalpha + x.spaces{order}{i} * alpha(i);
            end
        end
        
        function x = updateProperties(x)
            % updateProperties
            % x = updateProperties(x) updates the properties of x
            
            x.order=length(x.spaces);
            x.dim=cellfun(@(xs)length(xs),x.spaces)';
            x.sz=zeros(1,x.order);
            for mu=1:x.order
                x.sz(mu) = numel(x.spaces{mu}{1});
            end
        end
    end
    
    methods (Static)
        function x = randn(sz1,sz2,dims)
            d = numel(sz1);
            if nargin < 2
                sz2 = sz1;
            end
            if nargin < 3
                dims = ones(d,1);
            end
            sz1 = sz1(:);
            sz2 = sz2(:);
            u = cell(d,1);
            for mu = 1:d
                u{mu} = cell(dims(mu),1);
                u{mu} = cellfun(@(x) randn(sz1(mu),sz2(mu)),u{mu},'uniformoutput',false);
            end
            x = TSpaceTensors(u);
        end
        
        function x = zeros(sz1,sz2,dims)
            d = numel(sz1);
            if nargin < 2
                sz2 = sz1;
            end
            if nargin < 3
                dims = ones(d,1);
            end
            sz1 = sz1(:);
            sz2 = sz2(:);
            u = cell(d,1);
            for mu = 1:d
                u{mu} = cell(dims(mu),1);
                u{mu} = cellfun(@(x) zeros(sz1(mu),sz2(mu)),u{mu},'uniformoutput',false);
            end
            x = TSpaceTensors(u);
        end
        
        function x = ones(sz1,sz2,dims)
            d = numel(sz1);
            if nargin < 2
                sz2 = sz1;
            end
            if nargin < 3
                dims = ones(d,1);
            end
            sz1 = sz1(:);
            sz2 = sz2(:);
            u = cell(d,1);
            for mu = 1:d
                u{mu} = cell(dims(mu),1);
                u{mu} = cellfun(@(x) ones(sz1(mu),sz2(mu)),u{mu},'uniformoutput',false);
            end
            x = TSpaceTensors(u);
        end
    end
end