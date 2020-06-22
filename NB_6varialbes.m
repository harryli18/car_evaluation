% This code runs using matlab R2018B
% This line loads data which is split 80/20 for training/testing using train_test_split_6 file.
% Loading this mat file ensures both models use precisely the same train
% and test data.
load("Train_test_6.mat")
%%

rng('default'); % For reproducibility

% Optimize Naive Bayes Classifier by using OptimizeHyperparameters to 
% minimize cross-validation loss, useing the similar method as Naive Bayes
% 21 varialbes (see NB_Optimisation.m)

mdl_nb = fitcnb(Xtrain, Ytrain, 'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    'expected-improvement-plus'));
%%

% Get the model the predict the training set
yfit_train = predict(mdl_nb,Xtrain);
% Get the model the predict the test set
yfit_nb = predict(mdl_nb,Xtest);

% accurary from the training data
acc_train_nb = mean(double(yfit_train == Ytrain)) * 100;
% accurary from the test data
acc_test_nb = mean(double(yfit_nb == Ytest)) * 100;
disp(['Training accuracy: ' num2str(acc_train_nb) '%']);
disp(['Test accuracy: ' num2str(acc_test_nb) '%']);

%%
% Plot the confusion matrix 
figure(3);
cmh = confusionmat(gather(yfit_nb),gather(Ytest));
chlabels = {'0 Unnaceptable car','1 Acceptable car','2 Good car','3 Very good car'};
cmhc = confusionchart(cmh,chlabels)
title('CM for(6 vairable) Naive Bayes model');