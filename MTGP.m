classdef MTGP < handle
    properties
        Xtrain;
	ytrain;
	covfun;
	numParams;
	numTasks;
	params;
	useCorrelatedNoise;
	rankKf;
	rankKn;
    end
    methods 
	function obj = MTGP(covfun, numParams, numTasks, useCorrelatedNoise, params, rankKf, rankKn)
	    obj.covfun = covfun;
	    obj.numParams = numParams;
	    obj.numTasks = numTasks;

            if nargin>=4
	        obj.useCorrelatedNoise = useCorrelatedNoise;
	    else
	        obj.useCorrelatedNoise = 0;
	    end	    
            if nargin>=6
	        obj.rankKf = rankKf;
	    else
	        obj.rankKf = numTasks;
	    end	    
            if nargin>=7
	        obj.rankKn = rankKn;
	    else
	        obj.rankKn = numTasks;
	    end	    
	    if nargin>=5
                obj.params = params;
	    else
		if ~obj.useCorrelatedNoise
                    obj.params = log(10.^(2*(rand(1,obj.numParams+obj.numTasks+1+obj.numTasks*obj.rankKf)-0.5)));
		else
                    obj.params = log(10.^(2*(rand(1,obj.numParams+1+obj.numTasks*obj.rankKn+1+obj.numTasks*obj.rankKf)-0.5)));
		end
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
		if ~obj.useCorrelatedNoise
                    obj.params = log(10.^(2*(rand(1,obj.numParams+obj.numTasks+1+obj.numTasks*obj.rankKf)-0.5)));
		else
                    obj.params = log(10.^(2*(rand(1,obj.numParams+1+obj.numTasks*obj.rankKn+1+obj.numTasks*obj.rankKf)-0.5)));
		end
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

	function [kparams, Kf, D] = seperate_params(obj, params)
            kparams = params(1:obj.numParams); 
            if ~obj.useCorrelatedNoise
	        SigmaN = exp(params((obj.numParams+1):(obj.numParams+obj.numTasks)));
		D = diag(SigmaN.^2);
		nextIdx = obj.numParams+obj.numTasks+1;
	    else
	        D = eye(obj.numTasks) * exp(2*params(obj.numParams+1));
	        nextIdx = obj.numParams+2;
		for i = 1:obj.rankKn
		    v = params(nextIdx:(nextIdx+obj.numTasks-1));
		    D = D + v'*v;
		    nextIdx = nextIdx + obj.numTasks;
		end	
	    end
	        
	    Kf = eye(obj.numTasks) * exp(2*params(nextIdx));
	    nextIdx = nextIdx + 1;
	    for i = 1:obj.rankKn
		v = params(nextIdx:(nextIdx+obj.numTasks-1));
	        Kf = Kf + v'*v;
                nextIdx = nextIdx + obj.numTasks;
	    end	
	end

	function nlml = get_nlml(obj, paramsVal)
	    [kparams, Kf, D] = obj.seperate_params(paramsVal);
            Kx = obj.covfun(obj.Xtrain, obj.Xtrain, kparams);
	    I = eye(size(Kx,1));
            y = obj.ytrain(:);
	    valIdx = find(not(isnan(y)));
	    y = y(valIdx);
            n = length(y);
            KXX = kron(Kf, Kx);
	    KN = kron(D,I);
            KXX = KXX(valIdx, valIdx);
	    KN = KN(valIdx, valIdx);

	    L = chol(KXX+ KN); 
	    alpha = L\(L'\y);   
	    nlml = 0.5*y'*alpha+sum(log(diag(L)))+0.5*n*log(2*pi);
	end	

	function [predMean,predVar] = predict(obj, Xtest)
	    [kparams, Kf, D] = obj.seperate_params(obj.params);
            Kx = obj.covfun(obj.Xtrain, obj.Xtrain, kparams);
	    I = eye(size(Kx,1));
            y = obj.ytrain(:);
	    valIdx = find(not(isnan(y)));
	    y = y(valIdx);
            n = length(y);
            KXX = kron(Kf, Kx);
	    KN = kron(D,I);
            KXX = KXX(valIdx, valIdx);
	    KN = KN(valIdx, valIdx);
              
	    L = chol(KXX+ KN); 
	    alpha = L\(L'\y);   
            
            k = obj.covfun(obj.Xtrain, Xtest, kparams);
	    kx = kron(Kf, k);
	    kx = kx(valIdx,:); 

            kxx = obj.covfun(Xtest, Xtest, kparams);
	    kxx = kron(Kf, kxx);
	    predMean = kx'*alpha;
	    v = L\kx;
            predVar = kxx - v'*v;

	    predMean = reshape(predMean, [size(Xtest,1), obj.numTasks]);
	    predVar = reshape(diag(predVar), [size(Xtest,1), obj.numTasks]);
	end
    end
end
