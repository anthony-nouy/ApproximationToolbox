% Class OperatorTTTensor

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

classdef OperatorTTTensor
    
    properties
        cores
        order
        sz
        ranks
        isOrth
    end
    
    methods
        function x = OperatorTTTensor(y)
            if isa(y,'cell')
                x.cores = y;
                d = numel(y);
                x.order = d;
                x.ranks = zeros(1,d-1);
                x.sz = zeros(2,d);
                for mu = 1:d-1
                    x.ranks(mu) = y{mu}.sz(4);
                    x.sz(:,mu) = y{mu}.sz(2:3);
                end
                x.sz(:,d) = y{d}.sz(2:3);
                x.isOrth = false;
            elseif isa(y,'OperatorTTTensor')
                x.cores = y.cores;
                x.order = y.order;
                x.ranks = y.ranks;
                x.sz = y.sz;
                x.ranks = y.ranks;
                x.isOrth = y.isOrth;
            elseif isa(y, 'TuckerLikeTensor')
                c = TTTensor(y.core);
                xc = cell(c.order, 1);
                for i = 1:c.order
                    xsi = cellfun(@(x) full(x), y.space.spaces{i},...
                        'uniformoutput', false);
                    xsi = FullTensor(cat(3, xsi{:}), 3, ...
                        [y.sz(1, i), y.sz(2, i), numel(xsi)]);
                    xc{i} = timesTensor(c.cores{i}, xsi, 2, 3);
                    xc{i} = permute(xc{i}, [1, 3, 4, 2]);
                end
                x = OperatorTTTensor(xc);
            else
                error('Wrong arguments');
            end
        end
        
        function n =  storage(x)
            n = sum(cellfun(@numel,x.cores));
        end
        
        function n =  sparseStorage(x)
            n = sum(cellfun(@nnz,x.cores));
        end
        
        
        function z = plus(x,y)
            z = x;
            d = z.order;
            z.ranks = x.ranks + y.ranks;
            z.cores{1} = cat(x.cores{1},y.cores{1},4);
            z.cores{d} = cat(x.cores{d},y.cores{d},1);
            for mu = 2:d-1
                z.cores{mu} = cat(x.cores{mu},y.cores{mu},[1 4]);
            end
        end
        
        function z = uminus(z)
            z.cores{1} = -z.cores{1};
        end
        
        function z = minus(x,y)
            z = x + (-y);
        end
        
        function x = mtimes(x,y)
            if isa(x,'double') || isa(y,'double')
                if isa(x,'double')
                    x = x*y;
                else
                    x.cores{1} = y*x.cores{1};
                end
            elseif isa(y,'TTTensor')
                d = x.order;
                z = cell(d,1);
                for mu = 1:d
                    z{mu} = timesTensor(x.cores{mu},y.cores{mu},3,2);
                    z{mu} = permute(z{mu},[1 4 2 3 5]);
                    newSz = z{mu}.sz;
                    newSz = [newSz(1)*newSz(2) newSz(3) newSz(4)*newSz(5)];
                    z{mu} = reshape(z{mu},newSz);
                end
                x = TTTensor(z);
            elseif isa(y,'OperatorTTTensor')
                d = x.order;
                z = cell(d,1);
                for mu = 1:d
                    z{mu} = timesTensor(x.cores{mu},y.cores{mu},3,2);
                    z{mu} = permute(z{mu},[1 4 2 5 3 6]);
                    newSz = z{mu}.sz;
                    newSz = [newSz(1)*newSz(2) newSz(3) newSz(4) newSz(5)*newSz(6)];
                    z{mu} = reshape(z{mu},newSz);
                end
                x = OperatorTTTensor(z);
            else
                error('Wrong type of input arguments');
            end
        end
        
        function x = ctranspose(x)
            x.cores = cellfun(@(xc) permute(xc,[1 3 2 4]),x.cores, ...
                'uniformoutput',false);
            x.sz = flipud(x.sz);
        end
        
        function x = toVector(x)
            chSz = @(sz) [sz(1) sz(2)*sz(3) sz(4)];
            x = cellfun(@(xc) reshape(xc,chSz(xc.sz)), ...
                x.cores,'uniformoutput',false);
            x = TTTensor(x);
        end
        
        function s = dot(x,y)
            x = toVector(x);
            y = toVector(y);
            s = dot(x,y);
        end
        
        function n = norm(x)
            n = norm(toVector(x));
        end
        
        function [x,dims0] = squeeze(x,varargin)
            d = x.order;
            if nargin == 1
                dims0 = 1:d;
                dims0(prod(x.sz,1) ~= 1) = [];
            else
                dims0 = varargin{1};
            end
            dims0 = sort(dims0);
            dims = dims0;
            remainingDims = 1:d;
            remainingDims(dims) = [];
            if isempty(remainingDims)
                xc = x.cores;
                x = reshape(xc{1}.data,[xc{1}.sz(1) xc{1}.sz(4)]);
                for mu = 2:d
                    x = x*reshape(xc{mu}.data,[xc{mu}.sz(1) xc{mu}.sz(4)]);
                end
            else
                xc = x.cores;
                mu = 1;
                while (mu <= numel(dims)) && (mu == dims(mu))
                    xmu = squeeze(xc{mu},[2 3]);
                    xmu = xmu.data;
                    xc{mu+1} = timesMatrix(xc{mu+1},xmu,1);
                    mu = mu + 1;
                end
                dims(1:mu-1) = [];
                
                mu = d;
                k = numel(dims);
                while (k>0) && (mu == dims(k))
                    xmu = squeeze(xc{mu},[2 3]);
                    xmu = xmu.data;
                    xc{mu-1} = timesMatrix(xc{mu-1},xmu',4);
                    mu = mu - 1;
                    k = k-1;
                end
                dims(numel(dims):-1:k+1) = [];
                
                if ~isempty(dims)
                    mu0 = dims(1)-1;
                    for mu = dims(1):dims(end)
                        if any(remainingDims == mu)
                            mu0 = mu;
                        else
                            xmu = squeeze(x.cores{mu},[2 3]);
                            xmu = xmu.data;
                            xc{mu0} = timesMatrix(xc{mu0},xmu',4);
                        end
                    end
                end
                xc = xc(remainingDims);
                x = OperatorTTTensor(xc);
            end
        end
        
        % function x = full(x)
        %     error('Method not implemented.')
        % end
        
        function x = orth(x,lambda)
            z = x;
            if nargin == 1
                lambda = 1;
            end
            d = x.order;
            U = x.cores;
            % Left to right orth
            for mu = 1:lambda-1
                [U{mu},r] = orth(U{mu},4);
                U{mu+1} = timesMatrix(U{mu+1},{r},1);
            end
            % Right to left orth
            for mu = d:-1:lambda+1
                [U{mu},r] = orth(U{mu},1);
                U{mu-1} = timesMatrix(U{mu-1},{r},4);
            end
            z.cores = U;
            z.isOrth = lambda;
            for mu = 1:d-1
                z.ranks(mu) = z.cores{mu}.sz(4);
            end
        end
        
        % function x = cat(x,y)
        %     error('Method not implemented.')
        % end
        
        % function x = kron(x,y)
        %     error('Method not implemented.')
        % end
        
        % function x = dotWithRankOneMetric(x,M)
        %     error('Method not implemented.')
        % end
        
        % function x = timesDiagMatrix(x,M,varargin)
        %     error('Method not implemented.')
        % end
        
        % function x = timesMatrix(x,M,varargin)
        %     error('Method not implemented.')
        % end
        
        % function x = timesVector(x,M,varargin)
        %     error('Method not implemented.')
        % end
        
        % function x = timesTensorTimesMatrixExceptDim(x,M,dim)
        %     error('Method not implemented.')
        % end
    end
end