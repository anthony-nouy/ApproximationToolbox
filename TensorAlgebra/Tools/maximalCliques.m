% MAXIMALCLIQUES - Find maximal cliques using a recursive Bron-Kerbosch algorithm from a
% graph's boolean adjacency matrix
%
% cliques = MAXIMALCLIQUES(A)
% A: symmetric boolean square matrix representating an undirected graph
% with no self-edges
% C: matrix whose colums are the maximal cliques
%
% Ref: Bron, Coen and Kerbosch, Joep, "Algorithm 457: finding all cliques
% of an undirected graph", Communications of the ACM, vol. 16, no. 9,
% pp: 575?577, September 1973.
%
% Ref: Cazals, F. and Karande, C., "A note on the problem of reporting
% maximal cliques", Theoretical Computer Science (Elsevier), vol. 407,
% no. 1-3, pp: 564-568, November 2008.

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

function cliques = maximalCliques(A)
nok = (size(A,1) ~= size(A,2)) | ~all(all((A==1) | (A==0))) |  ~all(all(A==A.')) | trace(abs(A)) ~= 0;
if nok
    error('Adjacency matrix should be a symmetric boolean matrix with zero diagonal.')
end

n = size(A,2);
cliques = false(n,0);
R = [];
P = 1:n;
X = [];

recursive(R,P,X);

    function recursive(R, P, X )
        if (isempty(P) && isempty(X))
            newClique = false(n,1);
            newClique(R) = true;
            cliques = [cliques , newClique];
        else
            ppivots = union(P,X);
            binP = zeros(1,n);
            binP(P) = 1;
            pcounts = A(ppivots,:)*binP.';
            [~,ind] = max(pcounts);
            u_p = ppivots(ind);
            
            for u = intersect(find(~A(u_p,:)),P)
                P = setxor(P,u);
                Rnew = [R u];
                Nu = find(A(u,:));
                Pnew = intersect(P,Nu);
                Xnew = intersect(X,Nu);
                recursive(Rnew, Pnew, Xnew);
                X = [X u];
            end
        end
    end
end