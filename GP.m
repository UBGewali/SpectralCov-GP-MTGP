% Page-19 of Williams and Rasmussen, Gaussian processes for machine learning, MIT press, 2006.
classdef GP < handle
    properties
        Xtrain;
	ytrain;
	covfun;
	numParams;
	params;
    end
    methods 
	function obj = GP(covfun, numParams, params)
	    obj.covfun = covfun;
	    obj.numParams = numParams;
	    if nargin==3
                obj.params = params;
	    else
		obj.params = log(10.^(2*(rand(1,1+obj.numParams)-0.5)*5));	
	    end
	end	

        function fval = train(obj, Xtrain, ytrain)
	    obj.Xtrain = Xtrain;
	    obj.ytrain = ytrain;
           
            opts = optimset('Display','off','MaxIter',10000, 'TolFun',  1e-50, 'TolX', 1e-50, 'MaxFunEvals', 5000);
            [obj.params,fval,exitflag] = fminunc(@(x) obj.get_nlml(x), obj.params,opts);
	end

	function obj = train_multistart(obj, Xtrain, ytrain, numStarts)
	    listOfParams = {};
	    listOfNlml = [];
	    for i=1:numStarts
	        obj.params = log(10.^(2*(rand(1,1+obj.numParams)-0.5)*5));	
		try
		    nlml = obj.train(Xtrain, ytrain);
		catch
		    i = i - 1;
		    continue;	
		end
		listOfNlml = [listOfNlml, nlml];
		listOfParams = [listOfParams, obj.params];
	    end
	    [~, minIdx] = min(listOfNlml);
	    obj.params = listOfParams{minIdx};
	end	

	function nlml = get_nlml(obj, paramsVal)
	    sigma_n = exp(paramsVal(1));
	    kparams = paramsVal(2:end);
            n = size(obj.Xtrain,1);
            KXX = obj.covfun(obj.Xtrain, obj.Xtrain, kparams);
            L = chol(KXX+ (sigma_n^2)*eye(n)); 
	    alpha = L\(L'\obj.ytrain);   
	    nlml = 0.5*obj.ytrain'*alpha+sum(log(diag(L)))+0.5*n*log(2*pi);
	end	

	function [predMean,predVar] = predict(obj, Xtest)
	    sigma_n = exp(obj.params(1));
	    kparams = obj.params(2:end);
            KXX = obj.covfun(obj.Xtrain, obj.Xtrain, kparams);
            n = size(obj.Xtrain,1);
            L = chol(KXX+ (sigma_n^2)*eye(n)); 
	    alpha = L\(L'\obj.ytrain);   
	    kx = obj.covfun(obj.Xtrain, Xtest, kparams);
            kxx = obj.covfun(Xtest, Xtest, kparams);
            
	    predMean = kx'*alpha;
	    v = L\kx;
            predVar = kxx - v'*v;
	    predVar = diag(predVar);
	end
    end
end
