function oobErr = oobErrRF(params,X)
%   oobErrRF trains a random forest of 300 classification trees using the
%   predictor data in X and the parameter specification in params, and then
%   returns an estimate of mean OOB error. X is a table
%   and params is an array of OptimizableVariable objects corresponding to
%   the minimum leaf size and number of predictors to sample at each
%   node.these are our tuning paramaters.
randomForest = TreeBagger(300,X,'Ytrain','Method','classification',...
    'OOBPrediction','on','MinLeafSize',params.minLS,...
    'NumPredictorstoSample',params.numPTS);
oobErr = mean(oobError(randomForest));
end
