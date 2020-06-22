load("Train_test_21.mat")
% Train_test.mat load the training and test dataset. The dataset has been processed by one hot encoder 
% in order to prepare for our machine learning models. 

%%

% apply naive bayes model without any hyper-parameters tuning.
% The same distribution model (kernel) is used for a fair comparison. 

mdl_nb_simple = fitcnb(Xtrain, Ytrain,'DistributionNames','kernel');

% calculating the accuracy rate of prediction for both training and test set 
yfit_nb = predict(mdl_nb_simple,Xtest);
yfit_train = predict(mdl_nb_simple,Xtrain);
acc_train_nb = mean(double(yfit_train == Ytrain)) * 100;
acc_test_nb = mean(double(yfit_nb == Ytest)) * 100;

disp(['Training accuracy: ' num2str(acc_train_nb) '%']);
disp(['Test accuracy: ' num2str(acc_test_nb) '%']);

%%

