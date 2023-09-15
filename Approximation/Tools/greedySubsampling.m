function [x, deltanew, gstore] = greedySubsampling(x, basis, delta)
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
l = length(x(:,1));
G = 1/l*(WA')*WA;
gstore = norm(G-eye(m));
deltanew = gstore;
while deltanew < delta
    l = length(x(:,1));
    residuals = ones(1,l);
    WA = W*A;
    for i =1:l
        WAnew = WA;
        WAnew(i,:) = [];
        G = 1/(l-1)*(WAnew')*(WAnew);
        residuals(i) = norm(G-eye(m));
    end
    [minires, indres] = min(residuals);
    xnew = x;
    xnew(indres,:) = [];
    g = minires;
    xnew = x;
    xnew(indres,:) = [];
    if g > delta
        % gridalpha = gridalpha;
        break
    else
        x = xnew;
        A(indres,:) =[];
        W(indres,:) = [];
        W(:,indres) = [];
        deltanew = g;
        gstore = [gstore, deltanew];
    end
end
A = basis.eval(x);
W = diag(sqrt(m./sum(A.^2,2)));
WA = W*A;
G = 1/length(x(:,1))*(WA)'*WA;
deltanew = norm(G-eye(m));