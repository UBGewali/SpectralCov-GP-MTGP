function K = cov_SID(X1,X2,par)
    Sigma2 = exp(2*par(end));
    alpha2 = exp(2*par(end-1));

    X1 = X1 ./ sum(X1,2) + 1e-10;
    X2 = X2 ./ sum(X2,2) + 1e-10;

    dist_fun = @(p,Q)  sum((p-Q).*log(p./(Q+eps)),2);
    z =  pdist2(X1,X2,dist_fun);
    K = Sigma2 * exp(-z  * alpha2); 
end
