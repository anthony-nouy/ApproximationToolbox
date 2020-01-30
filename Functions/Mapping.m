% Class Mapping

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

classdef Mapping
    
    properties
        d
        m
        isIdentity
        storageType
        g
    end
    
    methods
        function map = Mapping(varargin)
            % Class Mapping
            %
            % map = Mapping('c',d,m,g)
            % Creates a mapping from Rd to Rm defined component-wise using
            % cell g
            % d: 1-by-1 double
            % m: 1-by-1 double
            % g: cell of function_handle
            %
            % map = Mapping('m',d,m,A,b)
            % Creates a mapping from Rd to Rm defined in matrix form using
            % the matrix A and the vector b
            % d: 1-by-1 double
            % m: 1-by-1 double
            % A: m-by-d double
            % b: m-by-1 double
            %
            % map = Mapping('f',d,m,g)
            % Creates a mapping from Rd to Rm defined using a
            % function that maps the entire set of variables
            % d: 1-by-1 double
            % m: 1-by-1 double
            % g: function_handle
            
            if length(varargin) == 1
                % An identity mapping is created, from Rd to Rd, with d the
                % input argument
                map.isIdentity = true;
                map.d = varargin{1};
                map.m = varargin{1};
                map.g = [];
                map.storageType = [];
            else
                map.isIdentity = false;
                
                % Storage type determination
                ind = find(cellfun(@(x) ischar(x),varargin),1);
                c = varargin{ind};
                varargin(ind) = [];
                nargin = length(varargin);
                type = {'ComponentWise', 'MatrixForm', 'Function'};
                
                if strncmpi(c,type{1},1) % Mapping stored component-wise
                    if nargin ~= 3
                        error('Wrong number of arguments.');
                    end
                    map.storageType = type{1};
                    map.d = varargin{1};
                    if length(varargin{3}) == 1 && ~iscell(varargin{3})
                        varargin{3} = {varargin(3)};
                    end
                    map.m = varargin{2};
                    map.g = varargin{3};
                elseif strncmpi(c,type{2},1) % Mapping stored in matrix form
                    if nargin ~= 4
                        error('Wrong number of arguments.');
                    end
                    map.storageType = type{2};
                    map.d = varargin{1};
                    map.m = varargin{2};
                    map.g = {varargin{3}, varargin{4}};
                    if size(varargin{3},2) ~= map.d || size(varargin{4},2) ~= 1
                        error('Wrong arguments size.');
                    end
                elseif strncmpi(c,type{3},1) % Mapping stored as a function
                    if nargin ~= 3
                        error('Wrong number of arguments.');
                    end
                    map.storageType = type{3};
                    map.d = varargin{1};
                    map.m = varargin{2};
                    map.g = varargin{3};
                end
            end
        end
        
        function y = eval(m,x)
            % y = eval(m,x)
            % Applies the mapping m to x
            % m: Mapping
            % x: N-by-d double
            % y: N-by-m double
            % See also Mapping
            
            if size(x,2) ~= m.d
                error('Wrong input size.');
            end
            
            if m.isIdentity == true % Identity transformation
                y = x;
            else
                switch m.storageType
                    case 'ComponentWise'
                        y = zeros(size(x,1),m.m);
                        for i = 1:m.m
                            yi = m.g{i}(x);
                            if size(yi,2) ~= 1
                                error('Components must be mappings from Rd to R')
                            end
                            y(:,i) = yi;
                        end
                    case 'MatrixForm'
                        y = x*m.g{1}.' + repmat(m.g{2}.',size(x,1),1);
                    case 'Function'
                        y = m.g(x);
                    otherwise
                        error('Wrong storage type.')
                end
                if (size(y,1) ~= size(x,1) || size(y,2) ~= m.m)
                    error('Incoherent sizes in the mapping, it must be modified.');
                end
            end
        end
        
        function m = keepMapping(m,k)
            % m = keepMapping(m,k)
            % Keeps only the mappings of m associated to the index k
            % m: Mapping
            % k: 1-by-n or n-by-1 double
            
            if ~m.isIdentity
                if min(k) < 1 || max(k) > m.m
                    error('Selected components out of bounds.');
                end
                m.m = length(k);
                switch m.storageType
                    case 'ComponentWise'
                        m.g = m.g(k);
                    case 'MatrixForm'
                        m.g = {m.g{1}(k,:) m.g{2}(k)};
                    case 'Function'
                        error('Method not compatible with Function storage type.')
                    otherwise
                        error('Wrong storage type.')
                end
            end
        end
        
        function m = modifyComponentMap(m,dims,varargin)
            % m = modifyComponentMap(m,dims,g)
            % Modifies the components of the mapping in dimensions in dims
            % using the function_handle in g
            % m: Mapping
            % dims: n-by-1 or 1-by-n double
            % g: 1-by-1 function_handle if n=1, n-by-1 of 1-by-n cell of
            % function_handle otherwise
            %
            % m = modifyComponentMap(m,dims,A,b)
            % Modifies the components of the mapping in dimensions in dims
            % using the matrix A and the vector b
            % m: Mapping
            % dims: n-by-1 or 1-by-n double
            % A: n-by-d double
            % b: n-by-1 double
            
            if nargin == 3 % Component-wise storage
                if ~iscell(varargin{1})
                    varargin{1} = varargin(1);
                end
                if m.isIdentity
                    m.isIdentity = false;
                    m.storageType = 'ComponentWise';
                    m.g = cell(1,m.d);
                    for i = 1:m.d
                        m.g{i} = @(x) x(:,i);
                    end
                end
                if length(varargin{1}) ~= length(dims)
                    error('Wrong dimensions in the cell.')
                end
                m.g(dims) = varargin{1};
            elseif nargin == 4 % Matrix form storage
                if m.isIdentity
                    m.isIdentity = false;
                    m.storageType = 'MatrixForm';
                    m.g = {eye(m.d) zeros(m.d,1)};
                end
                if size(varargin{1},1) ~= length(dims) || ...
                        size(varargin{1},2) ~= m.d
                    error('Wrong dimensions in the matrix.')
                end
                if length(varargin{2}) ~= length(dims)
                    error('Wrong dimensions in the vector.')
                end
                m.g{1}(dims,:) = varargin{1};
                m.g{2}(dims) = varargin{2};
            else
                error('Wrong type for the components.')
            end
        end
        
        function m = removeMapping(m,k)
            % m = keepMapping(m,k)
            % Removes the mappings of m associated to the index k
            % m: Mapping
            % k: 1-by-n or n-by-1 double
            
            if ~m.isIdentity
                kc = 1:m.m;
                kc(k) = [];
                m = keepMapping(m,kc);
            end
        end
        
        function m = permuteMapping(m,num)
            % m = permuteMapping(m,num)
            % Returns m with the mapping permutation num
            % m: Mapping
            % num: m-by-1 or 1-by-n double
            
            if ~m.isIdentity
                if ~isequal(sort(num),1:m.m)
                    error('A permutation must be provided.')
                end
                switch m.storageType
                    case 'ComponentWise'
                        m.g = m.g(num);
                    case 'MatrixForm'
                        m.g = {m.g{1}(num,:) m.g{2}(num)};
                    case 'Function'
                        error('Method not compatible with Function storage type.')
                    otherwise
                        error('Wrong storage type.')
                end
            end
        end
        
        function ok = eq(m1,m2)
            % ok = eq(m1,m2)
            % Returns a logical indicating if two Mapping object are equal
            % m1: Mapping
            % m2: Mapping
            % ok: logical
            
            if m1.isIdentity && m2.isIdentity
                ok = true;
            elseif  m1.isIdentity || m2.isIdentity
                ok = false;
            else
                ok = strcmpi(m1.storageType,m2.storageType) && m1.d == m2.d && ...
                    m1.m == m2.m;
                
                if ok
                    switch m1.storageType
                        case 'ComponentWise'
                            g1 = cellfun(@(x) func2str(x), m1.g,'UniformOutput',false); g1 = [g1{:}];
                            g2 = cellfun(@(x) func2str(x), m2.g,'UniformOutput',false); g2 = [g2{:}];
                            ok = ok && strcmpi(g1,g2);
                        case 'MatrixForm'
                            ok = ok && all(m1.g{1}(:) == m2.g{1}(:)) && all(m1.g{2}(:) == m2.g{2}(:));
                        case 'Function'
                            ok = ok && strcmpi(func2str(m1.g),func2str(m2.g));
                    end
                end
            end
        end
    end
end