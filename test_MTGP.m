clear all;
rng('shuffle');
load MTGP_toy_dataset

covfun = @cov_SAM;
numParams = 2;
numTasks = 2;

gp = GP(covfun, numParams);
mtgp1 = MTGP(covfun, numParams, numTasks);
mtgp2 = MTGP(covfun, numParams, numTasks,1); %correlated noise

idx = ~isnan(ytrain(:,1));
gp.train_multistart(Xtrain(idx,:),ytrain(idx,1), 30);
[mpred, vpred] = gp.predict(Xtest);

disp("GP:")
mdl = fitlm(ytest(:,1), mpred);
disp(sprintf("R2= %.4f", mdl.Rsquared.Ordinary));
rmse = sqrt(mean((ytest(:,1)-mpred).^2));
disp(sprintf("RMSE= %.4f", rmse));

mtgp1.train_multistart(Xtrain,ytrain, 30);
[mpred, vpred] = mtgp1.predict(Xtest);
mpred = mpred(:,1);

disp("MTGP1:")
mdl = fitlm(ytest(:,1), mpred);
disp(sprintf("R2= %.4f", mdl.Rsquared.Ordinary));
rmse = sqrt(mean((ytest(:,1)-mpred).^2));
disp(sprintf("RMSE= %.4f", rmse));

mtgp2.train_multistart(Xtrain,ytrain, 30);
[mpred, vpred] = mtgp2.predict(Xtest);
mpred = mpred(:,1);

disp("MTGP2:")
mdl = fitlm(ytest(:,1), mpred);
disp(sprintf("R2= %.4f", mdl.Rsquared.Ordinary));
rmse = sqrt(mean((ytest(:,1)-mpred).^2));
disp(sprintf("RMSE= %.4f", rmse));
