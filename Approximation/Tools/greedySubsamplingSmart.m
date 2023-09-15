function [xnew, deltanew, gstore] = greedySubsamplingSmart(x, basis, delta)
% function  [xnew, deltanew ] = greedySubsampling(x, basis, delta)
% Remove greedily points from a sample xn to provide a sample stable for
% least-squares approximation in the sense of [1]
% x : initial sample of length n
% basis : Approximation basis to build the least-squares approximation
% delta : threshold for the stability bound

% xnew : subsaample of length k <= n
% deltanew : distance beween the empirical Gram matrix computed with xnew
% and delta

% [1] Optimal Weighted Least-Squares. A. Cohen, G. Migliorati
m = cardinal(basis);
% Initialisation
A = basis.eval(x);
W = diag(sqrt(m./sum(A.^2,2)));
WA = W*A;
li = length(x(:,1));
l = length(x(:,1));
G = 1/l*(WA')*WA;
g = norm(G-eye(m));
gstore = g;
while (g < delta && l > m)
    B = l/(l-1)*(G-eye(m));
    [V,D,Q] = eig(B);
    [svG, ind] = sort(diag(D), 'descend'); 
    gmax = max(svG);
    gmin = min(svG);
   %  gmin = min(svG);
    Q = Q(:,ind);
    rho = -1/(l-1)*m;
    QA = (Q'*A')';
    gain = rho.*(QA(:,1)).^2./sum(QA(:,1:end).^2,2);
    f1 = gmax -  gain + 1./(l-1);
    f2 = - gmin +   gain + 1./(l-1);
    [minires1, indres1] = min(f1);
    [minires2, indres2] = min(f2);
    xnew = x;
    WA = W*A;
    WA1 = WA;
    WA1(indres1,:) = [];
    G1 = 1/(l-1)*(WA1')*WA1;
    g1 = norm(G1-eye(m));
    WA2 = WA;
    WA2(indres2,:) = [];
    G2 = 1/(l-1)*(WA2')*WA2;
    g2 = norm(G2-eye(m));
    if g1 > g2
        indres = indres2;
        gprov = g2;
        G = G2;
    else
        indres = indres1;
        gprov = g1;
        G = G1;
    end
    if gprov > delta
        break
    else
        g = gprov;
        xnew(indres,:) = [];
        x = xnew;
        A(indres,:) =[];
        W(indres,:) = [];
        W(:,indres) = [];
        l = length(x(:,1));
    end
end


% outputs
xnew = x;
deltanew= g;