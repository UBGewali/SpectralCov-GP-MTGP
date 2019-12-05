function K = cov_SAM(X1,X2,params)
    Sigma2 = exp(2*params(end));
    alpha2 = exp(2*params(end-1));

    X1X2 = X1*X2';
    XX1 = sum(X1.*X1,2);
    XX2 = sum(X2.*X2,2);

    z = X1X2./(sqrt(XX1*XX2')+10*eps);
    K = Sigma2 * exp( -  acos(z) * alpha2);
    K = real(K);
end
