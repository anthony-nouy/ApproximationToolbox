% Class VectorPCA

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

classdef VectorPCA
    
    properties
        display = false;
        samplingFactor = 1;
        adaptiveSampling = false;
        adaptiveSamplingPCA = false;
        testError = false;
    end
    
    methods
        
        function s = VectorPCA()
            % s = VectorPCA(X,sz)
            % Principal component analysis of a vector-valued random variable
            % s.testError (true or false): error estimation for determining the rank for prescribed tolerance
            % s.adaptiveSampling (true or false): adaptive sampling for determining the principal components with prescribed precision
            % s.samplingFactor: factor for determining the number of samples for the estimation of principal  components (1 by default)
            
        end
        
        function [V,output] = principalComponents(s,X,tol)
            % [V,output] = principalComponents(s,X,t)
            % principal components of a random vector X
            % If t is an integer, t is the number of principal components
            % If t<1, the number of principal components is determined such
            % that the relative reconstruction error after truncation is t.
            % X: object with methods size and random
            %     random(X,m) returns an array of size [m,size(X)]
            % t: number of components or a positive number <1 (tolerance)
            % V: matrix whose columns are the principal components
            % output.sv: corresponding singular values
            % output.numberOfSamples: number Of Samples of X
            
            n = prod(size(X));
             
            if tol<1
                m = s.samplingFactor * n;
            else
                m = s.samplingFactor * tol;
            end
            
            m=ceil(m);
            
            if tol<1 && ~s.adaptiveSampling && s.testError
                A = reshape(random(X,m),[m,n]).';
                A = FullTensor(A);
                [V,sv] = principalComponents(A,n);
                sv = diag(sv)/sqrt(m);
                
                r=0;err=Inf;
                while err > t
                    r=r+1;
                    err = sqrt(1 - sum(sv(1:r).^2)/sum(sv.^2));
                end
                rmin=r;
                
                mtest = 5;
                Atest = FullTensor(reshape(random(X,mtest),[mtest,n]));
                ttest = inf;
                
                for r=rmin:n
                    ttest = norm(Atest-V(:,1:r)*(V(:,1:r)'*Atest),'fro')/norm(Atest,'fro');
                    if ttest < t
                        break
                    end
                end
                
                if ttest >t
                    warning('Precision not reached, should adapt sample.')
                end
                
                V = V(:,1:r);
                sv = sv(1:r);
                output.numberOfSamples = m+mtest;
                
                
            elseif tol<1 && s.adaptiveSampling
                A = FullTensor(zeros(n,0),2,[n,0]);
                for k=1:m
                    A.data = [A.data,reshape(random(X,1),[n,1])];
                    [V,sv] = principalComponents(A,tol);
                    sv = diag(sv)/sqrt(m);
                    if sv(end)/sv(1)<1e-15 || size(V,2)<ceil(k/s.samplingFactor)
                        break
                    end
                end
                output.numberOfSamples = m;
                
            elseif tol<1 && s.adaptiveSamplingPCA && m > 1
                [A, lmold] = random(X,2);
                A = reshape(A,[2,lmold]).';
                A = FullTensor(A);
                errf = Inf;
                k=1;
                while errf > tol && k < m
                    k = k+1;
                    [Anew, lmnew] = random(X,1);
                    lm = max(lmold, lmnew);
                    Aprov = zeros(lm,size(Anew,2));
                    Aprovfull = zeros(lm,size(A.data,2));
                    Aprov(1:lmnew, :) = Anew;
                    Aprovfull(1:size(A.data,1), :) = A.data;
                    A.data = [Aprovfull,reshape(Aprov,[lm,1])];
                    cvp = cvpartition(k,'Leaveout');
                    rmax = min(size(A,1),k-1);
                    errtest = ones(rmax,k);
                    for i = 1:cvp.NumTestSets % for each fold i
                        teIdx = cvp.test(i);
                        learnIdx = setdiff(1:k,find(teIdx'));
                        Alearn = FullTensor(A.data(:,learnIdx));
                        Atest = FullTensor(A.data(:,find(teIdx')).');
                        [V, sv] = principalComponents(Alearn,rmax);
                        sv = diag(sv)./sqrt(k);
                        B = @(r)Atest-V(:,1:r)'*(V(:,1:r)*Atest);
                        for r =1:rmax
                            DEN = B(r);
                            errNum(rmax,i) = norm(DEN.data).^2;
                            errDen(rmax,i) = norm(Atest.data).^2;
                        end
                    end
                    
                    for r =1:rmax
                        err(r)= sqrt(sum(errNum(r,:))/sum(errDen(r,:)));
                        if err(r) < tol
                            ropt = r;
                            break
                        end
                    end
                    errf = err(end);
                    ropt = r;
                    lmold = lm;
                end
                [V, sv] = principalComponents(A,ropt);
%                 mtest = 10;
%                 Atest = reshape(random(X,mtest),[mtest,lm]).';
%                 ttest = norm(Atest-V*(V'*Atest),'fro')/norm(Atest,'fro');
%                 if ttest > tol
%                     warning('Problemos')
%                     errf
%                 end
                sv = diag(sv)/sqrt(m);
                output.errPCA = errf;
                output.numberOfSamples = k;
            else
                [A, lm] = random(X,m); 
                A = reshape(A,[m,lm]).';
                A = FullTensor(A);
                [V,sv] = principalComponents(A,tol);
                % V = V(:,1:numel(nnz(sv)));
                sv = diag(sv)/sqrt(m);
                % sv
                output.numberOfSamples = m;
                %                 if numel(sv) > 10
                %                     sv = sv(1:10);
                %                     V = V(:,1:10);
                %                 end
            end
            output.sv = sv;
            output.alphaRank = nnz(sv);
        end
    end
end
