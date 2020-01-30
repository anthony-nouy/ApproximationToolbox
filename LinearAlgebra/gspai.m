% GSPAI - Generalized sparse approximate inverse
% Minimize ||B-PA|| with respect to P for a given pattern
% A and B must be squared
%
% [P] = GSPAI(A,B,pattern)
% [P] = GSPAI(A,B,'banded',bandwidth)
% [P] = GSPAI(A,B,'auto',maxnz,epsi)
% [P] = GSPAI(A,B,'imposed',pattern,epsi);
% pattern can be equal to
%   'diag' : P is diagonal
%   'tridiag' : P is tridiagonal
%   'band' : P is a band matrix
%       bandwidth : width of the band on the diagonal
%       if bandwidth == 1, P is diag
%       if bandwidth == 2, P is tridiag, etc...
%   'auto' : pattern of P is automatic
%       maxnz : max number of non zeros
%       epsi : tolerance for ||B-PA||_2
%   'imposed'
%       pattern : list of index of the matrix
%   'power'
%       powermax : the pattern is thought as being the pattern of
%       $(I+A)^l$,
%       epsi : tolerance for ||B-PA||_2

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

function [P] = gspai(A,B,pattern,varargin)
n = size(A,1);

switch pattern
    case 'band'
        %% Can be accelerated using sparsity of A
        bw = varargin{1};
        nzmax = (2*bw-1)*n;
        P = spalloc(n,n,nzmax);
        for i = 1:bw
            P(i,1:i+(bw-1)) = B(i,:)/A(1:i+(bw-1),:);
            P(n-(i-1),n-(i-1)-(bw-1):n) = B(n-(i-1),:)/A(n-(i-1)-(bw-1):n,:);
        end
        for i = 1+bw:n-bw
            P(i,i-(bw-1):i+(bw-1)) = B(i,:)/A(i-(bw-1):i+(bw-1),:);
        end
        return
    case 'diag'
        P = gspai(A,B,'band',1);
        %         Why is the following false?
        %         dA = diag(A);
        %         dB = diag(B);
        %         P = spdiags(dB./dA,0,n,n);
        return
    case 'tridiag'
        P = gspai(A,B,'band',2);
        %         nzmax = 3*n-1;
        %         P = spalloc(n,n,nzmax);
        %         P(1,1:2) = B(1,:)/A(1:2,:);
        %         P(n,n-1:n) = B(n,:)/A(n-1:n,:);
        %         for i = 2:n-1
        %             P(i,i-1:i+1) = B(i,:)/A(i-1:i+1,:);
        %         end
        return
    case 'auto'
        %% Initial diagonal sparsity
        P = gspai(A,B,'diag');
        maxnz = varargin{1};
        epsi = varargin{2};
        %         for i=1:maxnz-1
        %             for k = 1:n
        %                 Jk = P(k,:) == 0;
        %                 j = nextIndex(P(k,:),A,B(k,:),Jk);
        %                 Jk = ~Jk;
        %                 Jk(j) = 1;
        %                 P(k,Jk) = B(k,:) / A(Jk,:);
        %             end
        %             if normest(P*A-B) < epsi
        %                 return
        %             end
        %         end
        %nB = norm(B,'fro');
        nB = norm(full(B));
        Pr = mat2cell(P,ones(1,size(P,1)),size(P,2));
        
        nrA = mat2cell(A,ones(1,size(A,1)),size(A,2));
        nrA = cellfun(@norm,nrA);
        nrA = nrA.^2;
        
        
        for i=1:maxnz-1
            parfor k = 1:n
                Jk = Pr{k} == 0;
                j = nextIndex(Pr{k},A,B(k,:),Jk,nrA);
                Jk = ~Jk;
                Jk(j) = 1;
                Pr{k}(Jk) = B(k,:) / A(Jk,:);
            end
            P = cell2mat(Pr);
            err = norm(full(P*A-B))/nB;
            if err < epsi
                return
            end
        end
        return
    case 'imposed'
        pattern = varargin{1};
        nzmax = numel(pattern);
        P = spalloc(n,n,nzmax);
        P(pattern) = 1;
        Pl = cell(n,1);
        J = cell(n,1);
        parfor k = 1:n
            [Pl{k},J{k}] = gspaiLine(P,A,B,k);
        end
        P = full(P);
        for k = 1:n
            P(k,J{k}) = Pl{k};
        end
        P = sparse(P);
        %         for k = 1:n
        %             Jk = P(k,:)~=0;
        %             P(k,Jk) = B(k,:) / A(Jk,:);
        %         end
        return
    case 'power'
        powermax = varargin{1};
        epsi = varargin{2};
        BAl = B;
        patt = find(BAl);
        for l = 0:powermax
            P = gspai(A,B,'imposed',patt);
            if normest(P*A-B) < epsi || l == powermax
                return
            else
                BAl = BAl*A;
                patt = union(patt,find(BAl));
            end
        end
    otherwise
        error('pas programme')
end

end

function [Pl,Jk] = gspaiLine(P,A,B,k)
Jk = P(k,:)~=0;
Pl = B(k,:) / A(Jk,:);
end

function j = nextIndex(Pk,A,Bk,Jk,nrA)
n = size(A,1);
r = Pk*A-Bk;
normr = norm(r);
ind = 1:n;
ind = ind(Jk);
c=1;
%     for i = ind;
% %% Version 1
% %         ei = zeros(1,n);
% %         ei(i) = 1;
% %         eiA = ei*A;
% %         normeiA = norm(eiA);
% %         gamma = -eiA*r'/(normeiA^2);
% %         ri = sqrt(normr^2 - gamma^2*normeiA^2);
% %% Version 2
%          gamma = -r*full(At{i})/nrA(i);
%          ri = sqrt(normr^2 - gamma^2*nrA(i));
%
%         if c==1
%             rj = ri;
%             j = i;
%             c=0;
%         else
%             if ri < rj
%                 rj = ri;
%                 j = i;
%             end
%         end
%     end
%% Version 3
Art = A(Jk,:)*r';
gamma = -Art./nrA(Jk);
rr = sqrt(normr^2-(gamma.^2).*nrA(Jk));
j = find(rr==min(rr));
j = ind(j(1));

end