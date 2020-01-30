% Class ImprovedInputParser

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

classdef ImprovedInputParser < inputParser
    
    properties
        unmatchedArgs
    end
    
    methods
        function p = ImprovedInputParser()
            p@inputParser();
            p.KeepUnmatched = true;
        end
        function parse(p,varargin)
            parse@inputParser(p,varargin{:});
            makeUnmatchedArgs(p);
        end
        
        function makeUnmatchedArgs(p)
            tmp = [fieldnames(p.Unmatched), ...
                struct2cell(p.Unmatched)];
            p.unmatchedArgs = reshape(tmp',[],1)';
        end
        
        function obj = passMatchedArgsToProperties(p,obj)
            fNames = fieldnames(p.Results);
            for k = 1:numel(fNames)
                obj.(fNames{k}) = p.Results.(fNames{k});
            end
        end
    end
end