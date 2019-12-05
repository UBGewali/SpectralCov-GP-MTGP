function K = cov_SE(X1,X2,params)
    Sigma2 = exp(2*params(end));
    l = exp(params(end-1));
    se = pdist2(X1/l,X2/l,'squaredeuclidean');
    K = Sigma2*exp(-0.5*se);
end
