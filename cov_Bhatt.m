function K = cov_Bhatt(X1,X2,par)
    n=0.5;
    Sigma2 = exp(2*par(end));
    c2 = exp(2*par(end-1));

    X1 = (X1./sum(X1,1)).^n;
    X2 = (X2./sum(X2,1)).^n;
    K = Sigma2 * (X1*X2')+c2;
end
