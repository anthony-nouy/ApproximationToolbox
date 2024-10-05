function ind = backwardGreedySubsampling(A,delta,n,w)
% function backwardGreedySubsampling(F,delta,n,w)

N = size(A,1);
m = size(A,2);
I = eye(m);

if nargin<3 || isempty(n)
    n = m;
end

if nargin==4
    W = spdiags(sqrt(w(:)),0,N,N);
    A = W*A;
end


G = (A'*A)/N;

mask = true(1,N);
K = 1:N;
k = N;
err = 0;

%fprintf('initial = %d\n',norm(G - I));

while k>n && err < delta

        indices = K(mask);
        [q1,w1] = eigs(G - I,1,'largestreal');
        %[qm,wm] = eigs(G - I,1,'smallestreal');
        %lb = (1+ k*w1 - (A(indices,:)*q1).^2)/(k-1);
        lb = - (A(indices,:)*q1).^2;
        %ub = (1 - k*wm + (A(indices,:)*q1).^2)/(k-1)
        ub = (A(indices,:)*q1).^2;
        
        [~, k1] = min(lb);
        [~, k2] = min(ub);
        
        G_k1 = (k*G - A(indices(k1),:)'*A(indices(k1),:))/(k-1);
        %ind1 = setdiff(indices,indices(k1));
        %G_k1 = (A(ind1,:)'*A(ind1,:))/(k-1);
        G_k2 = (k*G - A(indices(k2),:)'*A(indices(k2),:))/(k-1);
        %ind2 = setdiff(indices,indices(k2));
        %G_k2 = (A(ind2,:)'*A(ind2,:))/(k-1);
        err1 = norm(G_k1-I);
        err2 = norm(G_k2-I);
        if err1 < err2
            k_star = k1;
            G = G_k1;
            err=err1;
        else
            k_star = k2;
            G = G_k2;
            err=err2;            
        end
        
        if err < delta
            mask(indices(k_star)) = false;
            k = k-1;
        end
end
ind = K(mask);
%fprintf('final = %d\n',norm((A(ind,:)'*A(ind,:)/length(ind))-I))

