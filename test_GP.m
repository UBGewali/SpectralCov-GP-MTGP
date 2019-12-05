clear all;
rng('shuffle');
load GP_toy_dataset

covfun = @cov_SAM;
numParams = 2;

gp = GP(covfun, numParams);

gp.train_multistart(Xtrain,ytrain, 30);
[mpred, vpred] = gp.predict(Xtest);
   
md = LinearModel.fit(ytest,mpred);
r2 = md.Rsquared.Ordinary;
rmse = sqrt(mean((ytest-mpred).^2));
disp(sprintf("R^2=%.4f", r2));
disp(sprintf("RMSE=%.4f", rmse));
