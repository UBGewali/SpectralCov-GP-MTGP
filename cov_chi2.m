function K = cov_chi2(X1,X2,par)
    % This function uses VLFeat (http://www.vlfeat.org/).
    Sigma2 = exp(2*par(end));
    c2 = exp(2*par(end-1));
    
    X1 = X1'; 
    X2 = X2';
    X1 = X1./sum(X1,1);
    X2 = X2./sum(X2,1);
    K = Sigma2 * exp(-c2 * vl_alldist2(X1,X2,'CHI2'));
end
