% This code runs using matlab R2018B
% This line loads data which is split 80/20 for training/testing using train_test_split_21 file.
% Loading this mat file ensures both models use precisely the same train
% and test data.
load("Train_test_21.mat")
%%

% We used the matlab inbuilt function 'fitcnb' to train our multiclass 
% naive Bayes model. 

% We attempt to optimize Naive Bayes classifier by using OptimizeHyperparameters 
% to minimize our cross-validation loss

% 1. Our model uses the name-value argument to optimize the parameters by 
% setting 'OptimizeHyperparameters' to 'auto'. This means that Matlab provides 
% an automatic search in both distribution name (normal or kernel) and width 
% (kernel smoothing window width) to attempt to minimize the cross-validation error. 


% 2. In Hyperparameter optimization options, the acquisition function name is 
% set to 'expected-improvement-plus' to ensure that the algorithm is not 
% overexploiting any areas. Thus we can escape form a local objective function 
% minimum. 

mdl_nb = fitcnb(Xtrain, Ytrain, 'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    'expected-improvement-plus'));

yfit_nb = predict(mdl_nb,Xtest);
yfit_train = predict(mdl_nb,Xtrain);

% accurary from the training data
acc_train_nb = mean(double(yfit_train == Ytrain)) * 100;
% accurary from the test data
acc_test_nb = mean(double(yfit_nb == Ytest)) * 100;

%%
disp(['Training accuracy: ' num2str(acc_train_nb) '%']);
disp(['Test accuracy: ' num2str(acc_test_nb) '%']);

% Plot the confusion matrix 
figure(3);
cmh = confusionmat(gather(yfit_nb),gather(Ytest));
chlabels = {'0 Unnaceptable car','1 Acceptable car','2 Good car','3 Very good car'};
cmhc = confusionchart(cmh,chlabels)
title('CM for Tuned Naive Bayes model');


%%

