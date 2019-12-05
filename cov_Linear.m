function K = cov_Linear(X1,X2,param) 
    Sigma2 = exp(2*param(end));
    c2 = exp(2*param(end-1));

    K = Sigma2 * X1*X2' + c2;
end
