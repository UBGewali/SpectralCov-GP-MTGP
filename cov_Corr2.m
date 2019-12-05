function K = cov_Corr2(X1,X2,par)
    Sigma2 = exp(2*par(end));
    alpha2 = exp(2*par(end-1));

    z = pdist2(X1,X2,'correlation');
    K = Sigma2 * exp( -  z * alpha2);
end
